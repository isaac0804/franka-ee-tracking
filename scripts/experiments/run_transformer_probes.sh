#!/bin/bash
# Transformer architecture probe — 4 runs × 300k steps ≈ 45 min total
# Runs: tfm_base, tfm_large, tfm_base_lr3e4, mlp_baseline (control)
#
# Usage:
#   nohup bash run_transformer_probes.sh &
#   tail -f results/transformer_probes/probe.log

set -euo pipefail
source .venv/bin/activate

OUTROOT="results/transformer_probes"
LOG="$OUTROOT/probe.log"
mkdir -p "$OUTROOT"

# IK baselines for result display
IK_MT=38.1
IK_CI=12.1
IK_F8=7.7

run_one() {
    local name="$1"
    local cfg="$2"
    local outdir="$OUTROOT/$name"
    local evaldir="results/eval/transformer_probes/$name"

    echo "" | tee -a "$LOG"
    echo " START: $name  [$(date +%H:%M:%S)]" | tee -a "$LOG"

    if [ -f "$outdir/final_model.zip" ]; then
        echo "  SKIPPING (model exists)" | tee -a "$LOG"
        return
    fi

    python train.py --config "$cfg" --out "$outdir" 2>&1 | tee -a "$LOG"

    echo "  Evaluating $name ..." | tee -a "$LOG"
    python evaluate.py ablation --model "$outdir/final_model.zip" --out "$evaldir" 2>&1 | tee -a "$LOG"

    # Parse and print result
    if [ -f "$evaldir/ablation.json" ]; then
        python3 - <<EOF | tee -a "$LOG"
import json, pathlib
d = json.loads(pathlib.Path("$evaldir/ablation.json").read_text())
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",{}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",{}).get("residual_settled_rmse_mm", float("nan"))
print(f"  RESULT  moving_target={mt:.1f}mm  circle={ci:.1f}mm  fig8={f8:.1f}mm  (IK={$IK_MT}mm)")
EOF
    fi

    echo "  DONE: $name  [$(date +%H:%M:%S)]" | tee -a "$LOG"
}

echo "======================================================" | tee -a "$LOG"
echo " Transformer probe sweep — $(date)" | tee -a "$LOG"
echo " Runs: tfm_base, tfm_large, tfm_base_lr3e4, mlp_baseline" | tee -a "$LOG"
echo "======================================================" | tee -a "$LOG"

run_one "mlp_baseline_300k"    "ee_tracking/configs/transformer/mlp_baseline_300k.yaml"
run_one "tfm_base_300k"        "ee_tracking/configs/transformer/tfm_base_300k.yaml"
run_one "tfm_base_lr3e4_300k"  "ee_tracking/configs/transformer/tfm_base_lr3e4_300k.yaml"
run_one "tfm_large_300k"       "ee_tracking/configs/transformer/tfm_large_300k.yaml"

echo "" | tee -a "$LOG"
echo "======================================================" | tee -a "$LOG"
echo " SWEEP COMPLETE — $(date)" | tee -a "$LOG"
echo "======================================================" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo " Summary (IK baselines: MT=${IK_MT}mm  CI=${IK_CI}mm  F8=${IK_F8}mm):" | tee -a "$LOG"
for name in mlp_baseline_300k tfm_base_300k tfm_base_lr3e4_300k tfm_large_300k; do
    f="results/eval/transformer_probes/$name/ablation.json"
    if [ -f "$f" ]; then
        python3 - <<EOF | tee -a "$LOG"
import json, pathlib
d = json.loads(pathlib.Path("$f").read_text())
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",{}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",{}).get("residual_settled_rmse_mm", float("nan"))
print(f"  {\"$name\":<30}  MT={mt:5.1f}mm  CI={ci:5.1f}mm  F8={f8:5.1f}mm")
EOF
    fi
done
