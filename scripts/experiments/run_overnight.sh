#!/bin/bash
# Overnight sweep — 9 runs, sequential, ~6.5 hours
# Run with: bash run_overnight.sh  (or nohup bash run_overnight.sh &)
#
# Order (highest priority first):
#   1. rs012_10M          — THE target model (rs=0.12, 10M)
#   2. rs005_5M           — safe comparison baseline
#   3. rs008_5M           — rs grid gap (0.08)
#   4. rs010_5M           — rs grid intermediate (0.10)
#   5. rs015_5M           — upper limit push (0.15)
#   6. rs012_5M           — rs=0.12 5M checkpoint (compare to run 1)
#   7. rs012_cosine1e5_5M — deeper LR test (→1e-5 at 5M)
#   8. rs012_cosine1e5_10M— deeper LR at full scale (→1e-5 at 10M)
#   9. rs012_seed1_5M     — variance/reproducibility (seed=1)

set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

LOG=results/sweep/overnight.log
mkdir -p results/sweep
exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo " Overnight sweep started: $(date)"
echo "=============================================="

run_one() {
    local name="$1"
    local config="ee_tracking/configs/sweep/${name}.yaml"
    local outdir="results/sweep/${name}"
    local evaldir="results/eval/sweep/${name}"

    echo ""
    echo "----------------------------------------------"
    echo " START: $name  [$(date '+%H:%M:%S')]"
    echo "----------------------------------------------"

    # Skip if already complete
    if [ -f "${outdir}/final_model.zip" ]; then
        echo "  SKIP — model already exists at ${outdir}/final_model.zip"
    else
        mkdir -p "$outdir"
        python train.py --config "$config" --out "$outdir"
    fi

    # Eval
    mkdir -p "$evaldir"
    if [ -f "${evaldir}/ablation.json" ]; then
        echo "  SKIP eval — already done"
    else
        echo "  Running eval..."
        python evaluate.py ablation \
            --model "${outdir}/final_model.zip" \
            --out "$evaldir" \
            2>&1 | tee "${evaldir}/eval.log"
    fi

    # Print quick summary
    echo "  DONE: $name  [$(date '+%H:%M:%S')]"
    python - <<PYEOF
import json, sys
p = "${evaldir}/ablation.json"
try:
    d = json.load(open(p))
    mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", "?")
    ci = d.get("circle",{}).get("residual_settled_rmse_mm", "?")
    f8 = d.get("figure8",{}).get("residual_settled_rmse_mm", "?")
    ik = d.get("moving_target",{}).get("ik_settled_rmse_mm", "?")
    print(f"  RESULT  moving_target={mt:.1f}mm  circle={ci:.1f}mm  fig8={f8:.1f}mm  (IK={ik:.1f}mm)")
except Exception as e:
    print(f"  (could not parse eval: {e})")
PYEOF
}

# ── Execute in order ──────────────────────────────────────────────────────────
run_one rs012_10M
run_one rs005_5M
run_one rs008_5M
run_one rs010_5M
run_one rs015_5M
run_one rs012_5M
run_one rs012_cosine1e5_5M
run_one rs012_cosine1e5_10M
run_one rs012_seed1_5M

echo ""
echo "=============================================="
echo " All runs complete: $(date)"
echo "=============================================="

# ── Final summary table ───────────────────────────────────────────────────────
python - << 'PYEOF'
import json, glob, os

runs = [
    ("rs012_10M",           "rs=0.12  10M  cosine→1e-4"),
    ("rs005_5M",            "rs=0.05  5M   cosine→1e-4"),
    ("rs008_5M",            "rs=0.08  5M   cosine→1e-4"),
    ("rs010_5M",            "rs=0.10  5M   cosine→1e-4"),
    ("rs015_5M",            "rs=0.15  5M   cosine→1e-4"),
    ("rs012_5M",            "rs=0.12  5M   cosine→1e-4"),
    ("rs012_cosine1e5_5M",  "rs=0.12  5M   cosine→1e-5"),
    ("rs012_cosine1e5_10M", "rs=0.12  10M  cosine→1e-5"),
    ("rs012_seed1_5M",      "rs=0.12  5M   cosine→1e-4 seed=1"),
]

print("\n" + "="*72)
print(f"{'Run':<30} {'moving_target':>14} {'circle':>8} {'figure8':>8}")
print("-"*72)
IK_MM = 38.1
for name, label in runs:
    p = f"results/eval/sweep/{name}/ablation.json"
    if not os.path.exists(p):
        print(f"  {label:<28} {'(no eval)':>14}")
        continue
    d = json.load(open(p))
    mt = d.get("moving_target",{}).get("residual_settled_rmse_mm")
    ci = d.get("circle",{}).get("residual_settled_rmse_mm")
    f8 = d.get("figure8",{}).get("residual_settled_rmse_mm")
    mt_s = f"{mt:.1f}mm" if mt else "?"
    ci_s = f"{ci:.1f}mm" if ci else "?"
    f8_s = f"{f8:.1f}mm" if f8 else "?"
    flag = " ✓" if mt and mt < IK_MM else ""
    print(f"  {label:<28} {mt_s+flag:>14} {ci_s:>8} {f8_s:>8}")
print("="*72)
print(f"  IK+delay baseline:             {IK_MM:.1f}mm")
PYEOF
