# Architecture Diagrams

> **Current canonical architecture:** `DelayTransformerPolicy` (no cross-attention).
> See `README.md` for the up-to-date architecture diagram and description.
> The transformer architecture diagram is at `results/figures/transformer_architecture.png`.
>
> The Mermaid diagrams below document the **full control pipeline** (§1) and
> the legacy **MLP architecture** (§2–3) which is kept for historical reference.

---

## 1 — Full Control Pipeline

The residual PPO policy outputs a 7-D correction on top of damped-least-squares IK.
Both IK and residual travel through the same FIFO delay buffer.

```mermaid
flowchart TD
    TGT(["🎯 Moving Target\ntarget_pos(t) — known now"])

    subgraph OBS["📋 Observation — 95-D flat vector  (fed to policy)"]
        direction LR

        subgraph STATE["Current State  30-D"]
            S1["q         joint positions  ×7"]
            S2["qd        joint velocities ×7"]
            S3["ee_pos    EE pos + noise   ×3"]
            S4["pos_err   target − ee_pos  ×3"]
            S5["tgt_vel   target velocity  ×3"]
            S6["ik_qdot   DLS output       ×7"]
        end

        subgraph LOOK["Fine Lookahead  15-D  (oracle, covers 5-step delay window)"]
            L1["tgt_pos @ t + 20 ms  ×3"]
            L2["tgt_pos @ t + 40 ms  ×3"]
            L3["tgt_pos @ t + 60 ms  ×3"]
            L4["tgt_pos @ t + 80 ms  ×3"]
            L5["tgt_pos @ t + 100 ms  ×3"]
        end

        subgraph COARSE["Coarse Lookahead  12-D  (trend beyond delay)"]
            C1["tgt_pos @ t + 200 ms  ×3"]
            C2["tgt_pos @ t + 300 ms  ×3"]
            C3["tgt_pos @ t + 400 ms  ×3"]
            C4["tgt_pos @ t + 500 ms  ×3"]
        end

        subgraph CMDDELTA["Cmd-Delta History  35-D  ← Markov fix"]
            D1["setpt_oldest − q  ×7   executes at t+20ms"]
            D2["setpt_...    − q  ×7   executes at t+40ms"]
            D3["setpt_...    − q  ×7   executes at t+60ms"]
            D4["setpt_...    − q  ×7   executes at t+80ms"]
            D5["setpt_newest − q  ×7   executes at t+100ms"]
        end

        TJ["traj_onehot  3-D  (moving_target / circle / figure8)"]
    end

    VN["⚖️ VecNormalize\nobs ← (obs − μ̂) / (σ̂ + 1e-8)\nRunning mean/var updated each rollout"]

    subgraph POLICY["🧠 Transformer Policy  (DelayTransformerPolicy, no cross-attn)"]
        ACT["Actor [256×256] → 7-D action mean  +  log_std"]
        CRIT["Critic [256×256×256] → V̂(s)"]
    end

    subgraph ACTPROC["⚙️ Action Processing"]
        SAMP["Sample  ã ~ N(μ, σ)   or   ã = μ  at eval\nclip(ã, −1, 1)"]
        SCALE["correction = ã × residual_scale (0.12 rad)\nmax |correction| = 0.12 rad per joint"]
        SAMP --> SCALE
    end

    subgraph CTRLSTEP["🤖 Control Step"]
        IKC["DLS IK\nv_des = Kp·pos_err + tgt_vel\nq_ik += Jᵀ(JJᵀ + λ²I)⁻¹ v_des · dt"]
        MERGE["q_setpoint = clip(q_ik + correction, joint_limits)\nIK and residual merged — same FIFO path"]
    end

    subgraph FIFO["⏱️ FIFO Delay Buffer  —  D = 5 slots = 100 ms"]
        direction LR
        F0["slot 0\noldest\n→ executes now"] --> F1["slot 1"] --> F2["slot 2"] --> F3["slot 3"] --> F4["slot 4\nnewest\n← push here"]
    end

    MJC["🌍 MuJoCo\nctrl = q_setpoint(t−5)   ← 100 ms stale\nPD actuator → joint torques → physics\n→ ee_pos(t)   result of decision at t−5"]

    %% ── main data flow ─────────────────────────────────────────
    TGT -->|"future positions"| LOOK
    TGT -->|"future positions"| COARSE
    TGT --> IKC

    OBS --> VN --> POLICY
    POLICY --> ACTPROC
    ACTPROC --> MERGE
    IKC --> MERGE
    MERGE --> F4
    F0 --> MJC

    MJC -->|"noisy q, qd, ee_pos\n(joint noise 2mm · pos noise 5mm)"| STATE
    MERGE -->|"append to deque\n→ next obs cmd_delta"| CMDDELTA

    %% ── delay alignment note ────────────────────────────────────
    L1 -. "fine[0] = target at t+20ms\naligns with cmd[0] (oldest)\nwhich executes at t+20ms" .-> D1
    L5 -. "fine[4] = target at t+100ms\naligns with cmd[4] (newest)\nwhich executes at t+100ms" .-> D5
```

---

## 2 — Transformer Architecture (canonical, no cross-attention)

See `results/figures/transformer_architecture.png` for the rendered diagram.

```mermaid
flowchart TD
    subgraph OBS["Observation (95-D)"]
        RS["robot_state  30-D"]
        FL["fine_lookahead  5×3 = 15-D"]
        CL["coarse_lookahead  4×3 = 12-D"]
        CH["cmd_history  5×7 = 35-D"]
        TJ["traj_onehot  3-D"]
    end

    subgraph SLOT["Slot Tokenizer  (key structural prior)"]
        PAIR["pair: concat(fine[i], cmd[i]) for i=0..4\neach slot = 3+7 = 10 dims"]
        SPROJ["Linear(10 → 64) + pos_embed\n→ (B, 5, 64)  slot tokens"]
        PAIR --> SPROJ
    end

    subgraph STATEENC["State Encoder"]
        SCAT["concat(robot_state, coarse_flat, traj)  45-D"]
        SPROJ2["Linear(45 → 64)\n→ state_enc (B, 64)"]
        SCAT --> SPROJ2
    end

    TENC["TransformerEncoder\nPre-LN, 2 layers, 4 heads, d_model=64\nself-attention over 5 slot tokens"]
    POOL["mean pool → slots_enc (B, 64)"]
    CAT["concat(state_enc, slots_enc) → 128-D"]

    subgraph HEADS["Policy Heads"]
        ACTOR["Actor MLP [256, 256] → 7-D action"]
        CRITIC["Critic MLP [256, 256, 256] → scalar V̂"]
    end

    FL --> PAIR
    CH --> PAIR
    RS --> SCAT
    CL --> SCAT
    TJ --> SCAT

    SPROJ --> TENC --> POOL
    SPROJ2 --> CAT
    POOL --> CAT

    CAT --> ACTOR
    CAT --> CRITIC
```

**Key insight:** `cmd[i]` executes when the target is at `fine[i]`. Pairing them into
one slot token makes this temporal causal link explicit — the encoder can immediately
act on the cmd↔fine residual. Cross-attention was removed (Ablation B) because the
pairing already encodes the alignment cross-attention would have learned.

---

## 3 — Delay Impact: What the Policy Compensates

```mermaid
flowchart LR
    subgraph TIMELINE["Timeline — what happens to a decision made at time t"]
        direction TB

        PT["Policy decides correction at t\nbased on obs(t)"]
        QS["q_setpoint(t) pushed into FIFO"]
        EX["q_setpoint(t) executes at t+5\n(FIFO delay = 5 steps = 100 ms)"]
        EE["ee_pos changes at t+5\nthis error reaches obs at t+5"]

        PT --> QS --> EX --> EE
    end

    subgraph BREAKDOWN["Delay Breakdown (current config)"]
        direction TB
        DLY1["FIFO delay:    5.0 steps = 100 ms"]
        DLY2["Action filter: disabled (action_filter_hz=0.0)"]
        DLY3["Total:         5.0 steps = 100 ms"]
        DLY1 --- DLY2 --- DLY3
    end

    subgraph COVERAGE["Fine Lookahead Coverage (matches delay exactly)"]
        direction TB
        LA0["fine[0]  →  t + 20 ms  (cmd[0] executes here)"]
        LA1["fine[1]  →  t + 40 ms  (cmd[1] executes here)"]
        LA2["fine[2]  →  t + 60 ms  (cmd[2] executes here)"]
        LA3["fine[3]  →  t + 80 ms  (cmd[3] executes here)"]
        LA4["fine[4]  →  t + 100 ms (cmd[4] executes here)"]
        LA0 --- LA1 --- LA2 --- LA3 --- LA4
    end

    BREAKDOWN -->|"fine lookahead covers the\nexact FIFO delay window"| COVERAGE
```
