#!/bin/bash
# 2-hour sweep — fixes the train.py/evaluate.py bug and runs the right experiments.
#
# What was wrong: train.py only passed "net_arch" from policy_kwargs, ignoring
# all transformer-specific keys (d_model, use_pos_embed, pair_tokens, etc.).
# Every ablation ran with all defaults → same result. Both bugs are now fixed.
#
# Schedule (~85 min total, 35 min buffer):
#   1. ablation_a_no_pe_300k    — re-run correctly  [~10 min]
#   2. ablation_b_no_xattn_300k — re-run correctly  [~10 min]
#   3. ablation_c_unpaired_300k — re-run correctly  [~10 min]
#   4. tfm_base_5M              — best recipe, 5M   [~55 min]
#
# Reference baselines:
#   MLP rs012_5M:   MT=21.0mm  CI=7.6mm  F8=7.0mm
#   MLP rs012_10M:  MT=16.0mm  CI=5.3mm  F8=4.7mm
#   tfm_base_300k:  MT=27.0mm  CI=5.0mm  F8=6.5mm  (corrected probes)
#   IK:             MT=38.1mm  CI=12.1mm F8=7.7mm
#
# Usage:
#   nohup bash run_2h_sweep.sh &
#   tail -f results/2h_sweep/sweep.log

set -euo pipefail
source .venv/bin/activate

OUTROOT="results/2h_sweep"
LOG="$OUTROOT/sweep.log"
mkdir -p "$OUTROOT"

run_one() {
    local name="$1"
    local cfg="$2"
    local outdir="$OUTROOT/$name"
    local evaldir="results/eval/2h_sweep/$name"

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

    python3 - "$evaldir/ablation.json" <<'PYEOF' | tee -a "$LOG"
import json, sys, pathlib
f = pathlib.Path(sys.argv[1])
d = json.loads(f.read_text())
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",{}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",{}).get("residual_settled_rmse_mm", float("nan"))
print(f"  RESULT  MT={mt:.1f}mm  CI={ci:.1f}mm  F8={f8:.1f}mm")
PYEOF

    echo " DONE: $name  [$(date +%H:%M:%S)]" | tee -a "$LOG"
}

echo "════════════════════════════════════════════════════════" | tee -a "$LOG"
echo " 2-hour sweep — $(date)" | tee -a "$LOG"
echo " Fixes: train.py net_arch passthrough + evaluate.py import" | tee -a "$LOG"
echo " MLP refs: 5M MT=21.0/CI=7.6/F8=7.0 | 10M MT=16.0/CI=5.3/F8=4.7" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════" | tee -a "$LOG"

# ── Ablations (re-run correctly now that train.py is fixed) ──────────────────
run_one "ablation_a_no_pe_300k"    "ee_tracking/configs/transformer/ablation_a_no_pe_300k.yaml"
run_one "ablation_b_no_xattn_300k" "ee_tracking/configs/transformer/ablation_b_no_xattn_300k.yaml"
run_one "ablation_c_unpaired_300k" "ee_tracking/configs/transformer/ablation_c_unpaired_300k.yaml"

# ── tfm_base 5M — main experiment ───────────────────────────────────────────
run_one "tfm_base_5M"              "ee_tracking/configs/transformer/tfm_base_5M.yaml"

# ── Summary ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════" | tee -a "$LOG"
echo " SWEEP COMPLETE — $(date)" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "  Run                         MT       CI       F8" | tee -a "$LOG"
echo "  --------------------------  -------  -------  -------" | tee -a "$LOG"

for name in ablation_a_no_pe_300k ablation_b_no_xattn_300k ablation_c_unpaired_300k tfm_base_5M; do
    f="results/eval/2h_sweep/$name/ablation.json"
    [ -f "$f" ] || continue
    python3 - "$f" "$name" <<'PYEOF' | tee -a "$LOG"
import json, sys, pathlib
f, name = pathlib.Path(sys.argv[1]), sys.argv[2]
d = json.loads(f.read_text())
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",{}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",{}).get("residual_settled_rmse_mm", float("nan"))
print(f"  {name:<28}  {mt:5.1f}mm  {ci:5.1f}mm  {f8:5.1f}mm")
PYEOF
done
