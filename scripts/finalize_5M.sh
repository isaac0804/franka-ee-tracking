#!/bin/bash
# finalize_5M.sh — run after tfm_no_xattn_5M training completes
#
# 1. Evaluate the 5M model (3 trajectories, deterministic)
# 2. Print the result
# 3. Regenerate all static figures with 5M data
# 4. Generate 3D animations for all trajectories
#
# Usage:
#   bash scripts/finalize_5M.sh
#
# Assumes the model is at: results/main_runs/tfm_no_xattn_5M_s42/final_model.zip

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL="results/main_runs/tfm_no_xattn_5M_s42/final_model.zip"
MLP="results/sweep/rs012_10M/final_model.zip"
EVAL_OUT="results/eval/main_runs/tfm_no_xattn_5M_s42"
FIG_OUT="results/figures"

echo "════════════════════════════════════════════════════"
echo " Finalize 5M run — $(date)"
echo " Model: $MODEL"
echo "════════════════════════════════════════════════════"

# ── 1. Evaluate ───────────────────────────────────────────────────────────────
echo ""
echo "── Evaluating (3 trajectories) ─────────────────────"
python evaluate.py ablation \
    --model "$MODEL" \
    --out "$EVAL_OUT"

# ── 2. Print result ───────────────────────────────────────────────────────────
echo ""
python3 - "$EVAL_OUT/ablation.json" << 'PYEOF'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",       {}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",      {}).get("residual_settled_rmse_mm", float("nan"))
ik_mt = d.get("moving_target",{}).get("ik_settled_rmse_mm", float("nan"))
ik_ci = d.get("circle",       {}).get("ik_settled_rmse_mm", float("nan"))
ik_f8 = d.get("figure8",      {}).get("ik_settled_rmse_mm", float("nan"))
print("════════════════════════════════════════════════════")
print(f" Transformer no_xattn @ 5M:")
print(f"   MT = {mt:.1f} mm   (IK: {ik_mt:.1f} mm,  {(ik_mt-mt)/ik_mt*100:.0f}% improvement)")
print(f"   CI = {ci:.1f} mm   (IK: {ik_ci:.1f} mm,  {(ik_ci-ci)/ik_ci*100:.0f}% improvement)")
print(f"   F8 = {f8:.1f} mm   (IK: {ik_f8:.1f} mm,  {(ik_f8-f8)/ik_f8*100:.0f}% improvement)")
print("════════════════════════════════════════════════════")
PYEOF

# ── 3. Regenerate static figures ──────────────────────────────────────────────
echo ""
echo "── Regenerating figures ────────────────────────────"
python scripts/make_figures.py \
    --tfm "$MODEL" \
    --mlp "$MLP" \
    --out "$FIG_OUT"

# ── 4. Generate 3D animations ────────────────────────────────────────────────
echo ""
echo "── Generating animations (~5 min) ──────────────────"
python scripts/make_animation.py \
    --model "$MODEL" \
    --all \
    --out "$FIG_OUT"

# ── 5. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "Done. Now:"
echo "  1. Fill TBD rows in README.md with MT/CI/F8 results above"
echo "  2. git add results/figures/ results/eval/main_runs/"
echo "  3. git commit -m 'Add 5M results and final figures'"
echo "  4. Consider: git add -f results/main_runs/tfm_no_xattn_5M_s42/final_model.zip"
echo "════════════════════════════════════════════════════"
