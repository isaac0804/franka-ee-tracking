#!/bin/bash
# reproduce.sh — train the best model from scratch and evaluate it
#
# Reproduces the key result:
#   Transformer (no cross-attn), 5M steps
#   Expected: MT~16-19mm  CI~4-5mm  F8~4-5mm
#
# Runtime: ~55 min on a modern CPU with 20 parallel envs
#
# Usage:
#   bash reproduce.sh
#   bash reproduce.sh --out results/my_run   # custom output dir

set -euo pipefail
source .venv/bin/activate

OUT="results/reproduced"
for arg in "$@"; do
  case $arg in
    --out) OUT="$2"; shift 2 ;;
  esac
done

echo "════════════════════════════════════════════════════"
echo " Franka EE Tracking — Reproduce best result"
echo " Model: Transformer (no cross-attn), 5M steps"
echo " Output: $OUT"
echo " $(date)"
echo "════════════════════════════════════════════════════"

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "── Training (~55 min) ──────────────────────────────"
python train.py \
    --config ee_tracking/configs/transformer/tfm_no_xattn_5M.yaml \
    --out "$OUT"

# ── Evaluate ──────────────────────────────────────────────────────────────────
echo ""
echo "── Evaluating ──────────────────────────────────────"
python evaluate.py ablation \
    --model "$OUT/final_model.zip" \
    --out "$OUT/eval"

# ── Print result ──────────────────────────────────────────────────────────────
echo ""
python3 - "$OUT/eval/ablation.json" << 'PYEOF'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
mt = d.get("moving_target",{}).get("residual_settled_rmse_mm", float("nan"))
ci = d.get("circle",       {}).get("residual_settled_rmse_mm", float("nan"))
f8 = d.get("figure8",      {}).get("residual_settled_rmse_mm", float("nan"))
print("════════════════════════════════════════════════════")
print(f" Result:  MT={mt:.1f}mm   CI={ci:.1f}mm   F8={f8:.1f}mm")
print(f" Target:  MT~16-19mm     CI~4-5mm      F8~4-5mm")
print("════════════════════════════════════════════════════")
PYEOF

# ── Generate figures ──────────────────────────────────────────────────────────
echo ""
echo "── Figures ─────────────────────────────────────────"
python scripts/make_figures.py \
    --tfm "$OUT/final_model.zip" \
    --mlp "results/sweep/rs012_10M/final_model.zip" \
    --out "$OUT/figures" 2>/dev/null || \
python scripts/make_figures.py \
    --tfm "$OUT/final_model.zip" \
    --out "$OUT/figures"

echo ""
echo "Done. Model: $OUT/final_model.zip"
echo "Eval:        $OUT/eval/ablation.json"
echo "Figures:     $OUT/figures/"
