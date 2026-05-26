#!/bin/bash
# All 300k probes — corrected ablations + v2 architecture variants
# No 5M runs. ~70 min total (7 runs × ~10 min each).
#
# Ablations (A/B/C) — re-run with fixed train.py, compare vs tfm_base_300k:
#   A: use_pos_embed=false    does PE help?
#   B: use_cross_attn=false   is cross-attention necessary?
#   C: pair_tokens=false      is the cmd[i]↔fine[i] pairing the key win?
#
# v2 architecture variants — targeted improvements:
#   D: mlp_proj=true          MLP (LN+GELU) projections vs single Linear
#   E: use_reactive=true      direct robot_state bypass path
#   F: attn_pool=true         attention-weighted slot pooling vs mean
#   ALL: D+E+F+ffn_mult=4     full v2 architecture
#
# Baseline: tfm_base_300k  MT=27.0mm  CI=5.0mm  F8=6.5mm
# IK:                      MT=38.1mm  CI=12.1mm  F8=7.7mm
#
# Usage:
#   nohup bash run_probes_300k.sh &
#   tail -f results/probes_300k/probes.log

set -euo pipefail
source .venv/bin/activate

OUTROOT="results/probes_300k"
LOG="$OUTROOT/probes.log"
mkdir -p "$OUTROOT"

run_one() {
    local name="$1"
    local cfg="$2"
    local outdir="$OUTROOT/$name"
    local evaldir="results/eval/probes_300k/$name"

    echo "" | tee -a "$LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
    echo " START: $name  [$(date +%H:%M:%S)]" | tee -a "$LOG"

    if [ -f "$outdir/final_model.zip" ]; then
        echo "  SKIPPING (model exists)" | tee -a "$LOG"
    else
        python train.py --config "$cfg" --out "$outdir" 2>&1 | tee -a "$LOG"
    fi

    if [ ! -f "$evaldir/ablation.json" ]; then
        echo "  Evaluating $name ..." | tee -a "$LOG"
        python evaluate.py ablation --model "$outdir/final_model.zip" --out "$evaldir" 2>&1 | tee -a "$LOG"
    fi

    python3 - "$evaldir/ablation.json" "$name" <<'PYEOF' | tee -a "$LOG"
import json, sys, pathlib
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
name = sys.argv[2]
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",{}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",{}).get("residual_settled_rmse_mm", float("nan"))
base = dict(MT=27.0, CI=5.0, F8=6.5)
print(f"  RESULT  MT={mt:.1f}mm ({mt-base['MT']:+.1f})  CI={ci:.1f}mm ({ci-base['CI']:+.1f})  F8={f8:.1f}mm ({f8-base['F8']:+.1f})")
PYEOF

    echo " DONE: $name  [$(date +%H:%M:%S)]" | tee -a "$LOG"
}

echo "════════════════════════════════════════════════════════" | tee -a "$LOG"
echo " All 300k probes — $(date)" | tee -a "$LOG"
echo " Baseline: tfm_base_300k  MT=27.0mm  CI=5.0mm  F8=6.5mm" | tee -a "$LOG"
echo " IK:                      MT=38.1mm  CI=12.1mm  F8=7.7mm" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════" | tee -a "$LOG"

# ── Corrected ablations ──────────────────────────────────────────────────────
run_one "ablation_a_no_pe_300k"    "ee_tracking/configs/transformer/ablation_a_no_pe_300k.yaml"
run_one "ablation_b_no_xattn_300k" "ee_tracking/configs/transformer/ablation_b_no_xattn_300k.yaml"
run_one "ablation_c_unpaired_300k" "ee_tracking/configs/transformer/ablation_c_unpaired_300k.yaml"

# ── v2 architecture variants ─────────────────────────────────────────────────
run_one "tfm_v2d_mlpproj_300k"    "ee_tracking/configs/transformer/tfm_v2d_mlpproj_300k.yaml"
run_one "tfm_v2e_reactive_300k"   "ee_tracking/configs/transformer/tfm_v2e_reactive_300k.yaml"
run_one "tfm_v2f_attnpool_300k"   "ee_tracking/configs/transformer/tfm_v2f_attnpool_300k.yaml"
run_one "tfm_v2_all_300k"         "ee_tracking/configs/transformer/tfm_v2_all_300k.yaml"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════" | tee -a "$LOG"
echo " COMPLETE — $(date)" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "  Run                         MT       CI       F8" | tee -a "$LOG"
echo "  tfm_base (baseline)         27.0mm    5.0mm    6.5mm" | tee -a "$LOG"
echo "  MLP 10M  (champion)         16.0mm    5.3mm    4.7mm" | tee -a "$LOG"

for name in ablation_a_no_pe_300k ablation_b_no_xattn_300k ablation_c_unpaired_300k \
            tfm_v2d_mlpproj_300k tfm_v2e_reactive_300k tfm_v2f_attnpool_300k tfm_v2_all_300k; do
    f="results/eval/probes_300k/$name/ablation.json"
    [ -f "$f" ] || continue
    python3 - "$f" "$name" <<'PYEOF' | tee -a "$LOG"
import json, sys, pathlib
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
name = sys.argv[2]
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",{}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",{}).get("residual_settled_rmse_mm", float("nan"))
print(f"  {name:<28}  {mt:5.1f}mm  {ci:5.1f}mm  {f8:5.1f}mm")
PYEOF
done
