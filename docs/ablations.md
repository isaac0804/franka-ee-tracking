# Ablation Studies & Architecture Explorations

Full experimental record for the delay-aware transformer policy.
All experiments use the Franka Panda 7-DoF position-tracking task with a **5-step (100 ms) whole-pipeline delay**.
Unless noted, training is **300k steps, seed 42, 20 parallel envs**.
Metrics are settled RMSE (mm) after 0.5 s warmup — lower is better.

---

## 1. Core Architecture Ablation

Each variant removes or replaces **one component** from the canonical architecture.
The canonical model uses: self-attention only (no cross-attention), paired slot tokens, positional embedding.

![Ablation chart](../results/figures/ablation_bar.png)

| Variant | Moving Target | Circle | Figure-8 | Conclusion |
|---|---|---|---|---|
| **Canonical** (self-attn, no xattn) | **24.4 mm†** | **5.7 mm†** | **5.2 mm†** | — baseline |
| A: − Positional Embedding | 26.8 mm | 11.1 mm | 10.7 mm | PE critical for periodic trajectories |
| B: + Cross-Attention layers | 27.0 mm | 5.0 mm | 6.5 mm | xattn hurts MT and F8, marginal CI gain |
| C: Unpaired tokens | 26.6 mm‡ | 6.8 mm‡ | 6.9 mm‡ | pairing helps all trajectories |

† Mean over 2 seeds — seed 42: MT=23.6 / CI=4.9 / F8=4.8 mm; seed 1: MT=25.2 / CI=6.5 / F8=5.5 mm  
‡ Mean over 2 seeds — seed 42: MT=25.9 / CI=5.9 / F8=5.9 mm; seed 1: MT=27.2 / CI=7.7 / F8=7.8 mm

### Why each finding matters

**Positional Embedding (Ablation A)**  
Without PE, `slot[0]` and `slot[4]` are indistinguishable to the encoder — the temporal ordering of the delay queue is lost.
On periodic trajectories this is catastrophic (+5–6 mm on circle/figure-8) because the phase relationship between queued commands is the key signal.
Moving target is less affected because random-walk correction doesn't require slot ordering.

**Cross-Attention (Ablation B)**  
Adding cross-attention *hurts* (+2.6 mm MT, +1.7 mm F8).
The `cmd[i]↔fine[i]` pairing already encodes the temporal alignment that cross-attention was meant to learn.
The extra attention heads add capacity but introduce redundant inductive bias, degrading generalisation.
The canonical architecture (self-attention only) is both smaller and better.

**Paired Slot Tokens (Ablation C)**  
The key structural prior: `slot[i] = W[fine[i] ‖ cmd[i]]` wires each queued command to the target it will execute against.
An unpaired design (separate projections for fine and cmd, then concatenated) loses this alignment.
The penalty is consistent across all trajectories (+2 mm typical), confirming the pairing is always useful.

---

## 2. V2 Architecture Explorations

These variants were probed against the **cross-attention baseline** (MT=27.0, CI=5.0, F8=6.5 mm) —
note this baseline is slightly worse than the canonical (B=no-xattn) which is the final champion.
Single seed (42), 300k steps.

| Variant | Moving Target | Circle | Figure-8 | vs baseline | Conclusion |
|---|---|---|---|---|---|
| **Baseline** (+ xattn, tfm_base) | 27.0 mm | 5.0 mm | 6.5 mm | — | reference |
| v2-D: MLP Projection | 30.0 mm | 9.4 mm | 13.0 mm | −3.0 / −4.4 / −6.5 | MLP proj hurts; linear projection is sufficient |
| v2-E: Reactive Bypass | **25.6 mm** | 6.6 mm | 6.8 mm | **+1.4** / −1.6 / −0.3 | MT improves; periodic slightly worse |
| v2-F: Attention Pooling | 26.0 mm | 9.0 mm | **5.2 mm** | +1.0 / −4.0 / +1.3 | mixed; mean pool is more stable |

### Notes

**v2-D: MLP Projection (LN+GELU)**  
Replaces the single linear slot/state projections with 2-layer MLPs.
Significantly degrades figure-8 (13.0 mm vs 6.5 mm baseline).
Hypothesis: the non-linearity before attention distorts slot representations before self-attention has learned to align them.

**v2-E: Reactive Bypass Path**  
Adds a direct `robot_state → d_model` shortcut that feeds into the actor, bypassing the attention stack.
Hypothesis: gives the policy a fast reflex channel for stochastic tasks where delay prediction is less useful.
Moving target does improve (+1.4 mm), suggesting the bypass helps reactive correction.
However it adds 32k parameters and the gain is not seen at 5M steps (the canonical already handles MT at 20.6 mm).

**v2-F: Attention-Weighted Slot Pooling**  
Replaces mean pooling with a learned scalar salience score per slot (`Linear(d_model→1)` + softmax).
Adds only 65 parameters.
Inconsistent results: good F8 (5.2 mm) but poor CI (9.0 mm).
Mean pooling is more robust; attention pooling may over-weight a single delay slot.

---

## 3. Hyperparameter Sensitivity

### 3a. Residual Scale & Regularization Weight (5M, seed 42)

Varying `residual_scale` (how large a joint correction the policy can apply per step) and
`w_residual` (L2 penalty on the raw action):

| Config | residual_scale | w_residual | Moving Target | Circle | Figure-8 |
|---|---|---|---|---|---|
| rs005_5M | 0.05 | 0.087 | 20.0 mm | 5.1 mm | 5.3 mm |
| **rs012_5M (champion)** | **0.12** | **0.087** | **19.6 mm** | **4.8 mm** | **4.8 mm** |
| rs008_5M | 0.08 | 0.087 | 17.8 mm | 6.5 mm | 6.8 mm |
| rs015_lrdecay | 0.15 | 0.087 | 47.7 mm | 53.2 mm | 33.0 mm | 

`rs015` diverged — at scale=0.15 the policy can push the robot out of its workspace, triggering the early-termination penalty and causing instability.
`rs012` gives the best balance: enough authority to correct delay lag (+26 mm on circle) without destabilising.

### 3b. Learning Rate Schedule (5M)

| Config | LR schedule | LR end | Moving Target | Circle | Figure-8 |
|---|---|---|---|---|---|
| tfm_base_5M | constant 3e-4 | — | 22.2 mm | 15.5 mm | 11.3 mm |
| lrdecay_5M | linear 1e-3→1e-5 | 1e-5 | 20.5 mm | 13.5 mm | 13.0 mm |
| **rs012_5M (champion)** | **cosine 1e-3→5e-5** | **5e-5** | **19.6 mm** | **4.8 mm** | **4.8 mm** |

Constant LR prevents convergence on periodic trajectories.
Cosine decay to 5e-5 dramatically improves circle and figure-8: the policy needs the low final LR to stop oscillating around the fine-correction optimum.

### 3c. Other PPO Hyperparameters (5M)

| Config | Change | Moving Target | Circle | Figure-8 |
|---|---|---|---|---|
| **rs012_5M** | baseline | 19.6 mm | 4.8 mm | 4.8 mm |
| nsteps4096_lrdecay | n_steps 2048→4096 | 25.2 mm | 6.7 mm | 7.4 mm |
| epochs10_lrdecay | n_epochs 5→10 | 25.1 mm | 8.1 mm | 8.8 mm |
| nenvs20_mixedpool | n_envs 20, mixed pool | 27.4 mm | 6.0 mm | 6.1 mm |
| mixedpool_lrdecay_nosmooth | no smooth penalty | 27.8 mm | 7.4 mm | 6.5 mm |

Smaller rollout batches (n_steps=2048) and fewer gradient steps per batch (n_epochs=5) consistently outperform longer rollouts or more epochs.
The mixed trajectory pool without careful tuning also degrades performance.

---

## 4. Action Filter Experiments

Early probes with a baked 2nd-order Butterworth low-pass filter on the policy output.
**Note:** these used a small residual_scale=0.05 (pre-sweep), so the policy has minimal authority regardless of filtering — conclusions are limited to the filter's overhead.

| Config | Filter | residual_scale | w_residual | Policy RMSE | IK RMSE | Δ |
|---|---|---|---|---|---|---|
| baseline | 2 Hz Butterworth | 0.05 | 0.50 | 15.9 mm | 15.6 mm | −1.8% |
| nofilter | none | 0.05 | 0.50 | 15.6 mm | 15.6 mm | ~0% |
| hz5 | 5 Hz Butterworth | 0.05 | 0.50 | 15.4 mm | 15.6 mm | +1.1% |
| wr020 | 2 Hz Butterworth | 0.05 | 0.20 | 15.4 mm | 15.6 mm | +1.5% |
| rs010_wr020 | 2 Hz Butterworth | 0.10 | 0.20 | 15.4 mm | 15.6 mm | +1.2% |

At rs=0.05 the filter makes negligible difference — all configs are within noise of the IK baseline.
The champion config (rs=0.12, no filter, `action_filter_hz=0`) was chosen after the sweep confirmed that no filter is needed when the scale and regularization are tuned correctly.
The online training error (`tracking/pos_err_mm`) for the champion reaches ~9.7 mm (transformer) vs ~11.9 mm (MLP) by 5M steps.

---

## 5. MLP Baseline Sweep

For completeness, the MLP (GELU + LayerNorm) hyperparameter sweep:

| Config | Moving Target | Circle | Figure-8 | Notes |
|---|---|---|---|---|
| mlp_baseline_300k | 25.9 mm | 10.7 mm | 8.7 mm | 300k reference |
| mlp_5M (champion) | 19.6 mm | 7.9 mm | 6.7 mm | 5M, 2-seed mean |

The MLP champion uses the same `residual_scale=0.12` and `w_residual=0.087` as the transformer.
At 300k steps the MLP (25.9 mm MT) matches the canonical transformer (24.4 mm MT) within noise on moving target, but the gap widens at 5M steps — the transformer benefits more from extended training on periodic trajectories.

---

*Configs live in `ee_tracking/configs/transformer/` and `ee_tracking/configs/mlp/`.  
Run logs are in `results/` (2h_sweep, v2_probes, ablations, main_runs).  
Generate the ablation bar chart: `python scripts/make_figures.py`*
