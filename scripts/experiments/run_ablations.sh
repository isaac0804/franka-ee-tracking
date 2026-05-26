#!/bin/bash
# Architecture ablation sweep — 3 ablations × 300k steps ≈ 35 min total
#
# Tests three architectural choices against the tfm_base_300k baseline:
#   ablation_a — use_pos_embed=False     (remove learned positional embeddings)
#   ablation_b — use_cross_attn=False    (remove cross-attention, plain concat)
#   ablation_c — pair_tokens=False       (fine + cmd as independent token seqs)
#
# Baseline results (tfm_base_300k, already run):
#   MT=27.0mm  CI=5.0mm  F8=6.5mm
#
# Usage:
#   nohup bash run_ablations.sh &
#   tail -f results/ablations/ablation.log

set -euo pipefail
source .venv/bin/activate

OUTROOT="results/ablations"
LOG="$OUTROOT/ablation.log"
mkdir -p "$OUTROOT"

# IK baselines and tfm_base reference
IK_MT=38.1
IK_CI=12.1
IK_F8=7.7
BASE_MT=27.0
BASE_CI=5.0
BASE_F8=6.5

run_one() {
    local name="$1"
    local cfg="$2"
    local outdir="$OUTROOT/$name"
    local evaldir="results/eval/ablations/$name"

    echo "" | tee -a "$LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
    echo " START: $name  [$(date +%H:%M:%S)]" | tee -a "$LOG"

    if [ -f "$outdir/final_model.zip" ]; then
        echo "  SKIPPING (model exists)" | tee -a "$LOG"
        return
    fi

    python train.py --config "$cfg" --out "$outdir" 2>&1 | tee -a "$LOG"

    echo "  Evaluating $name ..." | tee -a "$LOG"
    python evaluate.py ablation --model "$outdir/final_model.zip" --out "$evaldir" 2>&1 | tee -a "$LOG"

    if [ -f "$evaldir/ablation.json" ]; then
        python3 - <<EOF | tee -a "$LOG"
import json, pathlib
d = json.loads(pathlib.Path("$evaldir/ablation.json").read_text())
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",{}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",{}).get("residual_settled_rmse_mm", float("nan"))

def delta(val, base):
    if val != val or base != base: return "±?"
    d = val - base
    return f"+{d:.1f}" if d >= 0 else f"{d:.1f}"

print(f"  RESULT  MT={mt:.1f}mm (d={delta(mt,$BASE_MT)})  CI={ci:.1f}mm (d={delta(ci,$BASE_CI)})  F8={f8:.1f}mm (d={delta(f8,$BASE_F8)})")
EOF
    fi

    echo " DONE: $name  [$(date +%H:%M:%S)]" | tee -a "$LOG"
}

echo "════════════════════════════════════════════════════" | tee -a "$LOG"
echo " Architecture ablation sweep — $(date)" | tee -a "$LOG"
echo " Baseline: tfm_base_300k  MT=${BASE_MT}mm  CI=${BASE_CI}mm  F8=${BASE_F8}mm" | tee -a "$LOG"
echo " IK:                      MT=${IK_MT}mm   CI=${IK_CI}mm   F8=${IK_F8}mm" | tee -a "$LOG"
echo " Ablations: A=no_pe  B=no_xattn  C=unpaired" | tee -a "$LOG"
echo "════════════════════════════════════════════════════" | tee -a "$LOG"

run_one "ablation_a_no_pe_300k"    "ee_tracking/configs/transformer/ablation_a_no_pe_300k.yaml"
run_one "ablation_b_no_xattn_300k" "ee_tracking/configs/transformer/ablation_b_no_xattn_300k.yaml"
run_one "ablation_c_unpaired_300k" "ee_tracking/configs/transformer/ablation_c_unpaired_300k.yaml"

echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════" | tee -a "$LOG"
echo " ABLATION SWEEP COMPLETE — $(date)" | tee -a "$LOG"
echo "════════════════════════════════════════════════════" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo " Summary vs tfm_base_300k (MT=${BASE_MT}  CI=${BASE_CI}  F8=${BASE_F8}):" | tee -a "$LOG"
echo " IK baseline:  MT=${IK_MT}  CI=${IK_CI}  F8=${IK_F8}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for name in ablation_a_no_pe_300k ablation_b_no_xattn_300k ablation_c_unpaired_300k; do
    f="results/eval/ablations/$name/ablation.json"
    if [ -f "$f" ]; then
        python3 - <<EOF | tee -a "$LOG"
import json, pathlib
d = json.loads(pathlib.Path("$f").read_text())
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",{}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",{}).get("residual_settled_rmse_mm", float("nan"))
print(f"  {'$name':<30}  MT={mt:5.1f}mm  CI={ci:5.1f}mm  F8={f8:5.1f}mm")
EOF
    fi
done
