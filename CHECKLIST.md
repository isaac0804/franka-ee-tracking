# Submission / Publication Checklist

Checklist for presenting this as a standalone internship project or public repo.
Check items off as they are completed.

---

## 1. Results

- [x] Corrected ablations A/B/C (2 seeds each) — all complete
- [x] Final architecture locked: `tfm_no_xattn` (paired slots, self-attention only)
- [x] Final results table in README: IK / MLP 300k / MLP 5M / MLP 10M / Tfm 300k / Tfm 5M
- [ ] 5M run completes — fill TBD rows in README (ETA ~15:40 BST, 2026-05-26)

---

## 2. Code cleanup ✅

- [x] Remove one-off bash scripts from repo root → `scripts/experiments/`
- [x] Remove scratch folder `yaml/` (deleted)
- [x] Remove stale probe/sweep configs (`ee_tracking/configs/probe/`, `ee_tracking/configs/sweep/`)
- [x] `reproduce.sh` — trains + evaluates best model end-to-end
- [x] `record_video.py` — simplified (trails removed, target sphere only; explained in header)
- [x] `requirements.txt` — range-pinned with tested version comment block
- [ ] Verify `requirements.txt` in a fresh venv (deferred — needs clean env)

---

## 3. Documentation ✅

### README.md ✅
- [x] Problem statement: Franka EE tracking, MuJoCo, 5-step actuation delay
- [x] Approach overview: Residual PPO, residual control, observation design
- [x] Architecture section with ASCII diagram (paired slot tokens)
- [x] Results table (IK / MLP / Transformer) with concise takeaway
- [x] Ablation table (A/B/C, 2 seeds each, footnotes)
- [x] Quickstart section (5 commands: install, assets, train, eval, tensorboard)
- [x] Design choices section (state, action, reward, trajectories, metrics, uncertainty)
- [x] Acknowledgements + repo structure
- [ ] Replace TODO GIF placeholder with final tracking animation (after 5M run)

### EXPERIMENTS.md ✅
- [x] MLP phase: full probe log, theories, overnight sweep, confirmed recipe
- [x] Transformer phase: phases 1–4, ablation A/B/C, v2 variants, theories T1/T2
- [x] Phase 4 (5M) status: TBD row to be filled when run completes

### REPORT.md ✅
- [x] Updated with transformer architecture section and ablation results
- [x] Full project narrative: what didn't work → what did → current best
- [x] Key design decisions table
- [x] Limitations and future work

---

## 4. Figures

- [x] `rmse_comparison.png` — IK / MLP / Transformer bar chart (static)
- [x] `efficiency_curve.png` — steps vs RMSE for MLP and Transformer
- [x] `ablation_bar.png` — CI/F8 RMSE for base / no-PE / no-xattn / unpaired
- [x] `transformer_architecture.png` — architecture diagram
- [x] `comparison_{traj}.png` — 3-way IK/MLP/Transformer trajectory comparison
- [ ] Update static figures with 5M result (`python scripts/make_figures.py --static-only`)
- [ ] Generate tracking animation for README GIF (`python scripts/make_animation.py --model <5M> --all`)
- [ ] Update `efficiency_curve.png` with 5M data point

---

## 5. Reproducibility

- [x] `reproduce.sh` — trains best model from scratch, evaluates, saves figures
- [x] `ee_tracking/configs/transformer/tfm_no_xattn_5M.yaml` — canonical best config
- [x] `ee_tracking/configs/mlp/mlp_best_10M.yaml` — canonical MLP champion config
- [ ] Pre-trained weights — consider GitHub Release or `results/canonical/` (after 5M finishes)
- [ ] Document compute requirements in README (currently ~55 min on the test machine)

---

## 6. Polish (after 5M run)

- [ ] Add demo GIF to README header (best tracking animation — `make_animation.py`)
- [ ] Final git commit with 5M results + updated figures + animation
- [ ] GitHub repo description + topics: `reinforcement-learning`, `robotics`, `mujoco`, `transformer`, `franka`
- [ ] Add badges: Python version, license
- [ ] Tag release `v1.0`

---

## Deferred (after submission)

- [ ] **Orientation tracking** — extend obs/action to 6-DoF (position + quaternion)
  - Architecture ready: same paired tokens, just wider fine lookahead
  - Deferred to maintain focus on position tracking for submission
- [ ] **Oracle lookahead → real predictor** — Kalman smoother or learned predictor over target history
- [ ] **Post-hoc smoothing** — 2 Hz Butterworth at inference for hardware deployment
- [ ] **Domain randomisation** — inertia, damping, contact params for sim-to-real
