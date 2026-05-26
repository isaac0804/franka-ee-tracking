# Hyperparameter Experiments — Findings & Theories

Running log of all experiments, what we learned, and current best recipe.
Update this file after every new result.

---

## Benchmarks (fixed reference points)

| System | RMSE (mm) | Notes |
|--------|-----------|-------|
| IK only, no delay | ~18 mm | theoretical floor without delay |
| IK only, 100 ms delay | ~37–44 mm | what the policy must beat |
| **IK floor** | **~14.7 mm** | best IK achieves (no delay, low noise) |
| Policy @ 500K steps (single pool) | ~27 mm | early-training reference |
| **Best 5M policy (moving_target)** | **16.64 mm** | `lrdecay_5M` training metric |
| **Best 5M policy eval (moving_target)** | **20.5 mm** | `lrdecay_5M` deterministic eval vs IK=38.1mm |
| **Best 500K probe (mixed pool, training)** | **13.7 mm** | `mixedpool_cosine1e4` / `lookahead_ext` — both tied |
| **Best 500K probe eval (moving_target)** | **26.2 mm** | `mixedpool_cosine1e4` vs IK=38.1mm |
| **Best 500K probe eval (circle)** | **7.4 mm** | `mixedpool_lrdecay_nosmooth` vs IK=12.1mm |
| **Best 500K probe eval (figure8)** | **5.9 mm** | `mixedpool_cosine1e4` vs IK=7.7mm — **beats IK** |

> ⚠️ Training `pos_err_mm` (from TensorBoard) ≠ eval `pos_err`. Training uses
> noisy obs + stochastic actions mid-episode. Eval uses deterministic rollout vs IK.
> Always compare models using the ablation eval, not training metrics alone.

---

## Eval Results (ablation: IK vs residual, deterministic rollout)

### 5M completed runs

| Model | moving_target | circle | figure8 | Notes |
|-------|-------------|--------|---------|-------|
| IK baseline | 38.1 mm | 12.1 mm | 7.7 mm | reference |
| `base_5M` | 22.2 mm (+42%) | 15.5 mm ❌ | 11.3 mm ❌ | single pool — fails on unseen traj |
| `lrdecay_5M` | **20.5 mm (+46%)** | 13.5 mm ❌ | 13.0 mm ❌ | single pool — same issue |

Single-pool models make circle/figure8 **worse** than IK — never seen those trajectories.

### 500K probes (weaker models, but show directions)

| Probe | moving_target | circle | figure8 | Notes |
|-------|-------------|--------|---------|-------|
| IK baseline | 38.1 mm | 12.1 mm | 7.7 mm | — |
| `mixedpool_lrdecay_nosmooth` | 27.8 mm | **7.4 mm** | **6.5 mm** | ✅ beats IK on all 3 at 500K |
| `cosine_lrdecay_nosmooth` (lr→1e-5) | 22.4 mm | **7.5 mm** | **5.1 mm** | ✅ generalises well; LR too deep |
| `mixedpool_cosine1e4` | **26.2 mm** ✅ | 7.9 mm ✅ | **5.9 mm** ✅ | cosine LR → 1e-4; best figure8 at 500K |
| `lookahead_ext` | 26.5 mm ✅ | 8.0 mm | 8.0 mm | coarse_horizon 4→8; helps moving_target, hurts circle/fig8 |

---

## Completed Training Runs

### 5M sweep runs

All use: `AsymGELUNormPolicy`, `gamma=0.97`, `n_epochs=5`, `n_steps=2048`,
`cmd_delay=5`, `obs_pos_noise=0.005`, `obs_jnt_noise=0.002`, `residual_scale=0.05`,
`trajectory_pool=["moving_target"]`.

| Run | Key change | train pos_err (final/best) | EV | Verdict |
|-----|-----------|--------------------------|-----|---------|
| `base_5M` | control, constant LR=1e-3 | 19.1 / 17.0 mm | 0.861 | baseline |
| `lrdecay_5M` | LR 1e-3 → 1e-4 linear | 17.9 / **16.6 mm** | **0.906** | ✅ best confirmed |
| `gamma99_5M` | gamma 0.97 → 0.99 | 21.7 / 20.1 mm | 0.847 | ❌ worse |
| `nosmooth_5M` | w_smooth=0, w_jerk=0 | — | — | ⏳ incomplete |

### Earlier rs/wr grid (~2M steps, constant LR)

| Run | rs | wr | train pos_err | EV |
|-----|----|----|--------------|-----|
| `rs002_wr01` | 0.02 | 0.10 | 17.96 mm | 0.869 |
| `rs005_wr01` | 0.05 | 0.10 | 17.04 mm | 0.863 |
| `rs008_wr01` | 0.08 | 0.10 | 15.99 mm | 0.851 |
| `rs012_wr01` | 0.12 | 0.10 | **14.90 mm** | 0.854 | ← near IK floor at 2M |
| `rs012_wr10` | 0.12 | 1.00 | 15.10 mm | 0.851 | ← wr=1.0 costs 0.2mm |

`residual_scale` is the dominant knob — monotonic improvement rs=0.02→0.12.
`w_residual` has almost no effect (see Theory 4).

---

## 500K Probes Log

### Round 1 — baseline screening

| Probe | Change | train pos_err | EV | clip | Verdict |
|-------|--------|-------------|-----|------|---------|
| `epochs10_lrdecay` | n_epochs 5→10 | 27.3 mm | 0.870 | 0.272 | ❌ clip saturates |
| `nsteps4096_lrdecay` | n_steps 2048→4096 | 28.3 mm | 0.861 | 0.117 | ❌ slow at 500K |
| `lrdecay_nosmooth` | w_smooth=0, w_jerk=0 | 24.9 mm | 0.888 | 0.111 | ✅ best |

### Round 2 — authority scaling (V-series)

rs=0.12 and rs=0.15 converge slowly at 500K — EV still low. Trust 2M rs-grid data instead.

| Probe | rs | train pos_err | EV | Note |
|-------|----|-------------|-----|------|
| `rs012_lrdecay` | 0.12 | 35.1 mm | 0.595 | still warming up at 500K |
| `rs012_lrdecay_nosmooth` | 0.12 | 27.0 mm | 0.729 | nosmooth helps even here |
| `rs015_lrdecay` | 0.15 | 40.2 mm | 0.569 | too slow; needs 2M+ steps |

### Round 3 — mixed pool & LR schedule (Batch 1) ⭐

**Mixed pool is the biggest single finding of the experiment.**

| Probe | Change | train pos_err | EV | clip | Eval: moving_target | circle | fig8 |
|-------|--------|-------------|-----|------|---------------------|--------|------|
| `mixedpool_lrdecay_nosmooth` | pool + nosmooth | **14.0 mm** | **0.982** | 0.114 | 27.8 mm | **7.4 mm** ✅ | **6.5 mm** ✅ |
| `cosine_lrdecay_nosmooth` | cosine LR→1e-5 + nosmooth | 25.1 mm | 0.886 | **0.003** | 22.4 mm | **7.5 mm** ✅ | **5.1 mm** ✅ |

Key observations:
- Mixed pool EV=0.982 at 500K — highest ever seen, higher than 5M lrdecay_5M (0.906)
- Mixed pool beats IK on **all 3 trajectories** at only 500K steps
- Cosine to 1e-5 collapsed (clip_frac→0.003) but still generalised well — LR too deep
- Both generalise completely vs single-pool models which hurt circle/figure8

### Batch 2 — lookahead & cosine refinement ✅

Both probes finished in ~11.6 min (single-probe, no contention).

| Probe | train pos_err | EV | clip | Eval: moving_target | circle | fig8 | Verdict |
|-------|-------------|-----|------|---------------------|--------|------|---------|
| `mixedpool_cosine1e4` | **13.7 mm** | 0.981 | 0.098 | **26.2 mm** | 7.9 mm | **5.9 mm** | ✅ use cosine for overnight |
| `lookahead_ext` (coarse_horizon 4→8) | **13.7 mm** | 0.983 | 0.118 | 26.5 mm | 8.0 mm | 8.0 mm | ⚠️ mixed — skip extended lookahead |

Key findings:
- **Cosine LR→1e-4** beats linear on moving_target (26.2 vs 27.8mm) and figure8 (5.9 vs 6.5mm), marginal difference on circle. Clip_frac 0.098 — healthy, not collapsing. **Use cosine for overnight.**
- **Extended lookahead** (obs 81→93 dims): better on moving_target (+1.3mm) but regresses on circle (+0.6mm) and figure8 (+1.5mm). Net cost outweighs benefit. **Stick with coarse_horizon=4.**
- Both hit same training pos_err (13.7mm) — extra lookahead dims don't improve the policy's ability to learn, they just redistribute error across trajectories.

### Batch 3 — n_envs=20 ✅

| Probe | train pos_err | EV | clip | Eval: moving_target | circle | fig8 | Time | Verdict |
|-------|-------------|-----|------|---------------------|--------|------|------|---------|
| `nenvs20_mixedpool` | 19.6 mm | 0.976 | 0.110 | 27.4 mm ✅ | **6.0 mm** 🏆 | 6.1 mm ✅ | **3.7 min** | ✅ use n_envs=20 |
| `rs012_mixedpool_cosine_n20` | 26.3 mm | 0.922 | 0.085 | **22.9 mm** 🏆 | 12.0 mm | 10.1 mm ❌ | **3.3 min** | ✅ green light for overnight |

Key findings:
- **n_envs=20** runs at 2491–3943 FPS — **3–4× faster** than n_envs=10 solo (730 FPS). Use for all future runs.
- Total gradient steps are **equal** at n_envs=10 and n_envs=20 for the same total_timesteps (both do ~48,800 gradient steps at 5M). Speed gain is pure wall-clock.
- **Best 500K moving_target eval ever: 22.9mm** (rs=0.12). Still under-converged (circle/fig8 regressed) — needs 5M to fully land.
- Mixed pool rescued rs=0.12 convergence: EV 0.595 (single-pool, 500K) → 0.729 (+ nosmooth) → 0.922 (+ mixed pool + cosine)

---

## Theory 1: Why gamma=0.99 Backfired

**Hypothesis:** gamma=0.99 → stronger credit signal through 5-step delay.
`(γλ)^5`: 0.97→0.644, 0.99→0.774 — 20% stronger.

**Result:** 4.5 mm worse (21.65 vs 19.07 mm), EV dropped 0.861→0.847.

**Explanation:** Longer horizon increases return variance faster than it improves
credit assignment. The EV drop confirms the critic got worse. `cmd_delta_history`
already partially restores Markov property — no systematic bias to fix.

**Conclusion:** gamma=0.97 is the right operating point.

---

## Theory 2: Why n_epochs=10 Doesn't Help

**Result:** clip_fraction 0.272→0.506 at 10 epochs. No pos_err improvement.

**Explanation:** At n_epochs=5, clip_fraction is already 0.46 in the 5M runs.
Adding more passes over exhausted data means more clipped/wasted gradient steps.
PPO data is exhausted after ~5 epochs; more passes overfit to stale rollouts.

**Conclusion:** Keep n_epochs=5.

---

## Theory 3: Why LR Decay Helps (EV 0.861 → 0.906)

**Observation:** LR 1e-3→1e-4 linear decay gave the largest single-knob EV gain.
Clip_fraction dropped from 0.46 steady-state to 0.14 at 5M steps.
Weight magnitudes shrank (block1 std: 0.0425→0.0316) reducing Tanh saturation.

**Explanation:** High constant LR keeps overshooting near-converged reward landscape.
Decay allows fine adjustments late in training → sharper minima → better calibrated critic.

**Cosine vs linear:** Cosine LR (probe B, lr→1e-5) collapsed clip_frac to 0.003 at
500K — LR dropped too far too fast. But generalisation was excellent (circle 7.5mm,
figure8 5.1mm). Retesting cosine with lr_final=1e-4 (batch 2B).

---

## Theory 4: The w_residual Penalty Has No Teeth

**Root cause:** `r_residual = -wr × ‖action × rs‖² = -wr × rs² × ‖action‖²`

The `rs²` factor makes the penalty negligible for small rs. With rs=0.05:
effective coefficient = 0.5 × 0.0025 = 0.00125 — nearly zero.

**The uniform action norm:** `‖action‖ ≈ 2.14` is constant regardless of wr or rs.
Policy always saturates; w_residual only matters near threshold `p_eff = wr × rs² ≈ 0.008`.

**Equivalence rule** (same effective penalty across rs values):
```
wr_new = wr_old × (rs_old / rs_new)²
rs=0.12 → wr=0.087  (to match wr=0.5 @ rs=0.05)
```

**Long-term fix (not yet implemented):** `r_residual = -wr × ‖action‖²` (scale-independent).

---

## Theory 5: The Policy is a Bang-Bang Controller

**Observation:** 47–87% of post-Tanh actions are |a|>0.9 in both 5M models.
Pre-Tanh std of 1.5–4.0 — deep in Tanh saturation zone (gradient ≈ 0).

**Why it makes sense:** w_residual has no teeth (Theory 4), so optimal policy is
to always saturate — max correction costs almost nothing, max reward gain.

**Implication:** Policy is **authority-limited**, not strategy-limited. Larger rs
directly improves tracking because each bang = bigger EE correction.

---

## Theory 6: Why Larger residual_scale Monotonically Helps

```
rs=0.02 → 17.9 mm  |  rs=0.05 → 17.0 mm  |  rs=0.08 → 16.0 mm  |  rs=0.12 → 14.9 mm
```

Policy is bang-bang → larger rs = bigger correction per saturated step.
rs=0.12 already hits IK floor (14.9mm) at 2M steps with constant LR.
Combined with lrdecay + nosmooth + 5M steps, target is <14 mm.

---

## Theory 7: Why nosmooth Helps

**Result:** w_smooth=0, w_jerk=0 → 24.9mm (vs ~27mm), EV 0.888 (vs 0.861–0.870).

**Explanation:** Smoothness/jerk penalties conflict with delay compensation. With 5-step
delay, the optimal correction is a **predictive impulse** — inherently jerky. Penalising
jerk directly penalises the main tool for delay compensation.

Lower clip_fraction (0.111 vs 0.272) confirms cleaner gradient signal.

**Risk:** Bang-bang + no smoothness = stressful joint commands. May need post-hoc
filtering before real hardware deployment.

---

## Theory 8: Why Mixed Trajectory Pool is a Major Win

**Result:** mixed pool (moving_target + circle + figure8) at 500K steps:
- Training pos_err: 14.0 mm (vs 24.9mm single pool) — 43% better
- EV: 0.982 — highest ever seen, even above 5M lrdecay_5M (0.906)
- **Eval: beats IK on all 3 trajectories at only 500K steps**

**Explanation (two effects):**

1. **Richer value landscape.** Circle and figure8 have predictable, bounded returns
   (IK nearly solves them). These clean, low-variance episodes give the critic easier
   regression targets, anchoring the value function and raising EV overall. The critic
   becomes better calibrated across the whole state space, not just moving_target states.

2. **Better gradient signal for moving_target.** The policy can't lazily fit to
   moving_target's specific distribution — it must learn generalisable tracking
   corrections that work across trajectory types, forcing more structured representations.

**Warning about EV inflation:** High EV on mixed pool may be partly because
circle/figure8 episodes are trivially predictable for the critic. Use eval (ablation)
to verify actual moving_target performance — don't trust training EV alone.

**Confirmed in eval:** mixed pool beats IK on circle (7.4 vs 12.1mm) and figure8
(6.5 vs 7.7mm) at 500K steps. Single-pool 5M models were *worse* than IK on both.

---

## Theory 9: n_envs Scaling (corrected)

**Initial wrong reasoning:** "12 cores → n_envs>10 causes oversubscription."

**Correction:** We already run 2×10-env probes simultaneously = 22 processes on
12 threads, without crashing. So 1×20-env = 21 processes is *less* oversubscribed.

**Actual result:**
- Wall-clock FPS: **2491–3943** (vs 730 at n_envs=10 solo) — 3–5× speedup confirmed
- Total gradient steps: **equal** for both n_envs at same total_timesteps. Not a quality trade-off.
- Empirical quality: competitive at 500K (nenvs20 circle=6.0mm, best ever); n_envs=20 adopted for all overnight runs.

**Why FPS exceeded prediction:** SB3 vectorized env batching has super-linear throughput
gains beyond 10 envs — avoids Python GIL bottleneck with larger parallel batches.

**Implication for overnight:** A 5M run with n_envs=20 takes ~30 min vs ~115 min with n_envs=10.
8-hour budget allows 9 sequential 5M/10M runs instead of 4.

---

## Current Best Recipe

### Confirmed at 5M steps
```yaml
learning_rate: 0.001
lr_final: 0.0001          # linear decay — biggest confirmed single knob
n_epochs: 5
n_steps: 2048
gamma: 0.97
residual_scale: 0.05
trajectory_pool: ["moving_target"]
# eval: 20.5mm on moving_target; fails on circle/figure8
```

### Confirmed overnight recipe (all knobs locked)
```yaml
trajectory_pool: ["moving_target", "circle", "figure8"]   # ✅ mixed pool
w_jerk: 0.0                    # ✅ nosmooth
w_smooth: 0.0
lr_schedule: "cosine"          # ✅ confirmed better than linear
learning_rate: 0.001
lr_final: 0.0001               # ✅ 1e-4 (1e-5 variant also planned)
n_envs: 20                     # ✅ 3-5× faster, equal gradient steps
lookahead_coarse_horizon: 4    # ✅ extended (=8) gave no net gain
delay_aware_gae: false         # ✅ cmd_delta_history already restores Markov
```

**Remaining sweep dimension: `residual_scale`** (rs-grid outdated; need 5M with full recipe):

See `ee_tracking/configs/sweep/` and `run_overnight.sh` for the 9-run overnight plan.

---

## What NOT to Try Again

| Idea | Why it failed |
|------|--------------|
| `gamma=0.99` | 4.5mm worse; return variance > credit signal gain |
| `n_epochs=10` | clip_fraction hits 0.50; data exhausted after 5 epochs |
| `delay_aware_gae=true` | **Do not enable.** cmd_delta_history already restores Markov property. Shifting rewards creates V/advantage mismatch. Code explicitly documents this. |
| `action_filter_hz=2Hz` | Adds ~5.6 steps of group delay on top of 5-step FIFO — doubles total latency. Currently disabled (0.0). |
| `cosine LR → 1e-5` (at 500K) | clip_frac collapses to 0.003; LR drops too fast for 500K steps. Use 1e-4 or test cosine only at 5M. |
| `rs=0.15` at 500K | EV=0.569 at 500K — still in warmup phase. Trust rs-grid at 2M instead. |

---

## Open Questions

1. ~~**Extended lookahead (coarse_horizon=8):**~~ **CLOSED** — mixed result; skip.
2. ~~**Cosine to 1e-4:**~~ **CLOSED** — cosine is better; use it.
3. ~~**n_envs=20:**~~ **CLOSED** — 3–5× faster, equal gradient steps, adopted permanently.
4. **Can we beat IK floor (14.7mm)?** rs=0.12 probe hit 22.9mm eval at 500K (not converged). At 5M: target <16mm.
5. **Optimal rs at 5M with full recipe?** Old grid at 2M, single-pool, constant LR. Overnight sweep tests rs ∈ {0.05, 0.08, 0.10, 0.12, 0.15}.
6. **Does cosine→1e-5 work at 5M scale?** At 600 gradient updates the LR stays warm for 2.5M steps — no collapse expected. Overnight tests this.
7. **How much do 10M steps add over 5M?** Overnight tests rs=0.12 at both depths.
5. **Scale-invariant w_residual:** `−wr × ‖action‖²` instead of `‖action × rs‖²` — makes
   wr tuning consistent across rs values. Implement before further rs experiments.
6. **Real hardware transfer:** bang-bang wrist control will need post-hoc smoothing.
   Evaluate action filter (2Hz Butterworth) applied post-hoc at inference only.

## Overnight Sweep Plan

9 sequential runs, ~6.5 hours, launched via `bash run_overnight.sh`.
All share: mixed pool + nosmooth + cosine LR + n_envs=20.

| # | Config | rs | lr_final | Steps | Purpose |
|---|--------|----|----------|-------|---------|
| 1 | `rs012_10M` | 0.12 | 1e-4 | 10M | 🎯 primary target model |
| 2 | `rs005_5M` | 0.05 | 1e-4 | 5M | safe comparison baseline |
| 3 | `rs008_5M` | 0.08 | 1e-4 | 5M | rs grid gap |
| 4 | `rs010_5M` | 0.10 | 1e-4 | 5M | rs intermediate |
| 5 | `rs015_5M` | 0.15 | 1e-4 | 5M | upper authority limit |
| 6 | `rs012_5M` | 0.12 | 1e-4 | 5M | 5M checkpoint of run 1 |
| 7 | `rs012_cosine1e5_5M` | 0.12 | **1e-5** | 5M | deeper LR test |
| 8 | `rs012_cosine1e5_10M` | 0.12 | **1e-5** | 10M | deeper LR at full scale |
| 9 | `rs012_seed1_5M` | 0.12 | 1e-4 | 5M | variance / reproducibility (seed=1) |

Results land in `results/sweep/` with auto-eval after each run.
Summary table printed to `results/sweep/overnight.log` at the end.

---

## Eval Bug Fixed

`evaluate.py` was not restoring `residual_scale` from the saved config, causing the
eval env to use the default (0.4 rad) instead of the training value (0.05 rad).
With 8× larger corrections, every episode hit the 300mm failure threshold in ~70 steps.
Fixed: `_env_kwargs_from_cfg` now restores `residual_scale`, `lookahead_coarse_horizon`,
and `lookahead_coarse_dt` alongside the existing obs-space params.
