# Submission Checklist

---

## 1. Results

- [x] Ablations A/B/C — 2 seeds each, complete
- [x] Final architecture locked: `tfm_no_xattn` (paired slots, self-attention only)
- [x] Main results table: IK / MLP 300k / MLP 5M / MLP 10M / Tfm 300k / Tfm 5M
- [x] Rigorous multi-seed comparison (10 seeds MT, 5 seeds step_target)
- [x] Smoothness metrics table (action roughness + saturation rate)
- [x] OOD evaluation: square, rectangle, fast circle, step target
- [x] Scaling curves (1M / 2M / 3M / 5M checkpoints, both architectures)
- [x] Transformer seed=1 5M run — complete, evals done, TBD rows filled

---

## 2. Code

- [x] `train.py` — YAML-driven training
- [x] `evaluate.py` — IK vs policy table, multi-seed, OOD, smoothness metrics
- [x] `record_video.py` — tracking video recorder
- [x] `sweep.py` — hyperparameter sweep runner
- [x] `scripts/make_training_curves.py` — training convergence figure
- [x] `scripts/make_comparison_bars.py` — per-trajectory RMSE bar chart
- [x] `scripts/scaling_eval.py` — checkpoint scaling curve eval + plot
- [x] `requirements.txt` — range-pinned
- [ ] Fresh venv install test (running in background)

---

## 3. Documentation

- [x] `README.md` — problem, design note, results, quickstart (option A + B), architecture, ablations, OOD, design choices, repo structure
- [x] `REPORT.md` — full project narrative (problem → what didn't work → what did → results)
- [x] `EXPERIMENTS.md` — detailed experiment log
- [x] `docs/architecture.md` — Mermaid architecture diagrams

---

## 4. Figures (all in `results/figures/`)

- [x] `tracking_3d_moving_target.gif` — main tracking animation (README header)
- [x] `tracking_3d_circle.gif` + `tracking_3d_figure8.gif`
- [x] `rmse_comparison.png` — IK / MLP / Transformer bar chart
- [x] `efficiency_curve.png` — sample efficiency comparison
- [x] `ablation_bar.png` — ablation A/B/C results
- [x] `training_curves.png` — convergence curves (s1 shown as dashed in-progress)
- [x] `comparison_5M_bars.png` — 7-trajectory MLP vs Transformer bars
- [x] `scaling_curves.png` — RMSE vs compute at 1M/2M/3M/5M
- [x] `transformer_architecture.png`

---

## 5. Pre-trained weights (`results/canonical/`)

- [x] `transformer_5M.zip` + `transformer_5M_vecnormalize.pkl` + `transformer_5M_config.yaml`
- [x] `mlp_5M.zip` + `mlp_5M_vecnormalize.pkl` + `mlp_5M_config.yaml`
- [x] `.gitignore` updated to include `results/canonical/`

---

## 6. Repo hygiene

- [x] `LICENSE` (MIT)
- [x] `.gitignore` — excludes `.venv/`, `results/*` (except figures + canonical), `assets/`
- [ ] GitHub repo created and made public
- [ ] Repo description + topics: `reinforcement-learning` `mujoco` `franka` `transformer` `robotics` `ppo`
- [ ] Final commit with all results + tag `v1.0`

---

## Post-submission (deferred)

- [ ] Orientation tracking (6-DoF position + quaternion)
- [ ] Oracle lookahead → Kalman/learned predictor
- [ ] Domain randomisation for sim-to-real
- [ ] Curriculum over delay magnitude
