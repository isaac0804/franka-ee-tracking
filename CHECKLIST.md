# Submission / Publication Checklist

Checklist for presenting this as a standalone internship project or public repo.
Check items off as they are completed.

---

## 1. Results (blocker — wait for sweep)

- [ ] 2h sweep finishes: corrected ablations (A/B/C) + `tfm_base_5M`
- [ ] Decide whether to run `tfm_base_10M` for apple-to-apple vs `rs012_10M`
- [ ] Final results table locked in:
  - IK baseline / MLP 300k / MLP 5M / MLP 10M / Transformer 5M (or 10M)
  - All three trajectories: moving_target, circle, figure8

---

## 2. Code cleanup

- [ ] Remove one-off bash scripts from repo root — keep only clean entry points:
  - `train.py`, `evaluate.py`, `sweep.py` → keep
  - `run_transformer_probes.sh`, `run_ablations.sh`, `run_2h_sweep.sh` → delete or move to `scripts/experiments/`
  - `eval_posthoc.py` → move to `scripts/` or remove if superseded
- [ ] Remove or clean scratch folder `yaml/` (appears to be throwaway configs)
- [ ] Remove `nosmooth_resume.log` and other stray log files from `results/sweep/`
- [ ] Consolidate `reproduce.sh` — single script that trains + evaluates the best model end-to-end
- [ ] Audit `record_video.py` — either fix trail visibility (TODO) or remove the trail injection code and document limitation
- [ ] Consistent naming pass: `cmd_delay` vs `act_delay`, `residual_settled_rmse` vs `residual_rmse` — pick one everywhere
- [ ] Verify `requirements.txt` is complete and pinned (test in a fresh venv)

---

## 3. Documentation

### README.md
- [ ] Problem statement: Franka EE tracking, MuJoCo, 5-step actuation delay
- [ ] Approach overview: Residual PPO — what it is, why residual, why PPO
- [ ] Architecture section with diagram (link `results/figures/transformer_architecture.png`)
- [ ] Results table (IK / MLP / Transformer) with concise takeaway
- [ ] Quickstart section:
  ```bash
  pip install -r requirements.txt
  python train.py --config ee_tracking/configs/transformer/tfm_base_5M.yaml --out results/my_run
  python evaluate.py ablation --model results/my_run/final_model.zip
  ```
- [ ] Reproducing best results section
- [ ] License + acknowledgements

### EXPERIMENTS.md
- [ ] Add final transformer probe findings (corrected A/B/C ablations)
- [ ] Add `tfm_base_5M` result when available
- [ ] Add entry for the two bugs found (train.py net_arch passthrough, evaluate.py import)

### REPORT.md
- [ ] Update with transformer architecture section
- [ ] Update with final results and ablation analysis
- [ ] Write the one-paragraph narrative:
  > *"cmd[i] executes when the target is at fine[i] — pairing them as a slot token
  > wires in the delay structure the MLP must discover from scratch. At 300k steps
  > the transformer matches what the MLP needs 10M steps to achieve on periodic
  > trajectories (CI: 5.0 vs 10.7mm at 300k; MLP reaches 5.3mm only at 10M)."*

---

## 4. Figures

- [ ] **Architecture diagram** — regenerate `draw_transformer.py` once architecture is finalised
- [ ] **Training curves** — reward + pos_err vs steps for MLP vs Transformer (pull from TensorBoard)
  - Script: `make_training_curves.py` (to write)
  - Show: transformer reaches CI=5mm faster than MLP
- [ ] **3-way trajectory comparison** — extend `make_figures.py` to plot IK / MLP / Transformer on same axes
- [ ] **Ablation bar chart** — CI and F8 RMSE for: full model / no PE / no cross-attn / unpaired
- [ ] **Per-trajectory RMSE summary bar chart** — clean final-results figure for README
- [ ] **Tracking video** — `record_video.py` side-by-side IK vs Transformer
  - [ ] Fix trail visibility (sphere radius / alpha issue, marked TODO in record_video.py)
  - [ ] Record one video per trajectory type (moving_target, circle, figure8)
  - [ ] Export as GIF for README embed

---

## 5. Reproducibility

- [ ] `requirements.txt` — pinned versions, tested in clean venv
- [ ] `reproduce.sh` — trains best model from scratch, evaluates, saves figures
- [ ] Pre-trained weights — either commit to `models/` (if small enough) or GitHub Release
- [ ] Seed parity: confirm `seed=42` in best config gives deterministic training
- [ ] Document compute requirements: "~55 min on [CPU spec] for tfm_base_5M"

---

## 6. Polish

- [ ] GitHub repo description + topics: `reinforcement-learning`, `robotics`, `mujoco`, `transformer`, `franka`
- [ ] Add a demo GIF to README header (best tracking video)
- [ ] Add badges: Python version, license
- [ ] Tag a release `v1.0` once the above is done

---

## Priority order

| # | Task | Est. time | Blocks |
|---|---|---|---|
| 1 | Wait for 2h sweep results | — | Everything |
| 2 | README (core sections) | 1 day | Publicity |
| 3 | Training curves figure | 4 hrs | Report / README |
| 4 | 3-way comparison figure | 4 hrs | Report / README |
| 5 | Ablation bar chart | 2 hrs | Report |
| 6 | Code cleanup + reproduce.sh | 1 day | Submission |
| 7 | Fix video trails | 1 day | Demo GIF |
| 8 | requirements.txt + fresh venv test | 2 hrs | Reproducibility |
| 9 | EXPERIMENTS.md + REPORT.md update | 4 hrs | Documentation |
| 10 | Architecture diagram update | 2 hrs | README |
