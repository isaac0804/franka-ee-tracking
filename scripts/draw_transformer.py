"""Draw a clean architecture diagram for DelayTransformerPolicy."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── palette ───────────────────────────────────────────────────────────────────
C_OBS_ROBOT  = "#4e79a7"   # blue   — robot state
C_OBS_FINE   = "#e05252"   # red    — fine lookahead
C_OBS_COARSE = "#f28e2b"   # orange — coarse lookahead
C_OBS_CMD    = "#59a14f"   # green  — cmd history
C_OBS_TRAJ   = "#b07aa1"   # purple — traj one-hot
C_PROJ       = "#76b7b2"   # teal   — linear projections
C_ATTN       = "#edc948"   # yellow — attention blocks
C_XATTN      = "#ff9da7"   # pink   — cross-attention
C_MLP        = "#9c755f"   # brown  — MLP heads
C_OUT        = "#bab0ac"   # grey   — outputs
ARROW        = "#555555"

def box(ax, x, y, w, h, label, sublabel="", color="#cccccc",
        fontsize=9, subfontsize=7.5, alpha=0.85, radius=0.015):
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0.01,rounding_size={radius}",
                           fc=color, ec="white", lw=1.2, alpha=alpha, zorder=3)
    ax.add_patch(patch)
    if sublabel:
        ax.text(x, y + h*0.12, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)
        ax.text(x, y - h*0.22, sublabel, ha="center", va="center",
                fontsize=subfontsize, color="white", alpha=0.9, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)

def arrow(ax, x0, y0, x1, y1, color=ARROW, lw=1.4, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

def curved_arrow(ax, x0, y0, x1, y1, rad=0.3, color=ARROW, lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}"),
                zorder=2)

def dim_label(ax, x, y, text, color="#555555", fontsize=7):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, style="italic", zorder=5)

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#f8f8f8")
ax.set_facecolor("#f8f8f8")

ax.text(7.5, 9.7, "DelayTransformerPolicy — Best Architecture (no cross-attention)",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#333333")

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Observation segments
# ═══════════════════════════════════════════════════════════════════════════════
y_obs = 8.8
obs_segments = [
    (1.3,  1.8, "robot state\n30 dims",         "q  q̇  ee_pos  ee_err\ntarget_vel  ik_qdot",  C_OBS_ROBOT),
    (3.5,  1.4, "fine lookahead\n5 × 3 = 15 d", "target at\nt+20…100 ms",                     C_OBS_FINE),
    (5.4,  1.2, "coarse lookahead\n4 × 3 = 12 d","target at\nt+120…520 ms",                   C_OBS_COARSE),
    (7.5,  1.8, "cmd history\n5 × 7 = 35 d",    "queued joint Δ\noldest → newest",             C_OBS_CMD),
    (9.5,  0.9, "traj\none-hot\n3 d",            "",                                            C_OBS_TRAJ),
]
for (cx, w, label, sub, col) in obs_segments:
    box(ax, cx, y_obs, w, 0.7, label, sub, color=col, fontsize=8, subfontsize=6.5)

ax.text(0.25, y_obs, "obs\n(95 d)", ha="center", va="center",
        fontsize=8, color="#555555", style="italic")

# bracket around obs
ax.plot([0.5, 0.5, 10.4, 10.4], [y_obs-0.45, y_obs-0.55, y_obs-0.55, y_obs-0.45],
        color="#aaaaaa", lw=1.2, zorder=1)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Pairing annotation + separate state path
# ═══════════════════════════════════════════════════════════════════════════════
y_pair = 7.55

# Pairing brace for fine + cmd
for cx in [3.5, 7.5]:
    arrow(ax, cx, y_obs - 0.35, cx, y_pair + 0.18, color=C_OBS_FINE if cx==3.5 else C_OBS_CMD)

# Pairing label
ax.annotate("", xy=(5.5, y_pair + 0.28), xytext=(3.5, y_pair + 0.28),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=1.0,
                            connectionstyle="arc3,rad=0"))
ax.annotate("", xy=(5.5, y_pair + 0.28), xytext=(7.5, y_pair + 0.28),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=1.0,
                            connectionstyle="arc3,rad=0"))
ax.text(5.5, y_pair + 0.48, "pair: fine[i] ↔ cmd[i]\n(same execution slot i)",
        ha="center", va="bottom", fontsize=7.5, color="#555555", style="italic")

# Slot concat box
box(ax, 5.5, y_pair, 3.2, 0.55,
    "concat(fine[i], cmd[i])  ×5 slots",
    "each slot: 3+7 = 10 dims",
    color=C_PROJ, fontsize=8.5)

# State path: robot + coarse + traj flow down
for cx, col in [(1.3, C_OBS_ROBOT), (5.4, C_OBS_COARSE), (9.5, C_OBS_TRAJ)]:
    arrow(ax, cx, y_obs - 0.35, cx, y_pair - 0.55, color=col)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Linear projections
# ═══════════════════════════════════════════════════════════════════════════════
y_proj = 6.6

arrow(ax, 5.5, y_pair - 0.28, 5.5, y_proj + 0.28)

box(ax, 5.5, y_proj, 3.0, 0.50,
    "slot_proj  Linear(10 → 64)  ×5",
    "+ learned pos_embed(slot_idx)",
    color=C_PROJ, fontsize=8.5)

dim_label(ax, 5.5, y_proj - 0.45, "→ (B, 5, 64)")

# State proj (robot + coarse + traj concatenated)
# Draw convergence lines into state_proj
y_state_proj = 6.35
for cx, col in [(1.3, C_OBS_ROBOT), (5.4, C_OBS_COARSE), (9.5, C_OBS_TRAJ)]:
    curved_arrow(ax, cx, y_pair - 0.55, 11.2, y_state_proj + 0.20,
                 rad=-0.15 if cx < 5 else (0.0 if cx < 8 else 0.2), color=col, lw=1.2)

box(ax, 11.2, y_state_proj, 2.8, 0.50,
    "state_proj  Linear(45 → 64)",
    "robot(30) + coarse(12) + traj(3)",
    color=C_PROJ, fontsize=8.5)

dim_label(ax, 11.2, y_state_proj - 0.42, "→ (B, 1, 64)  [state token]")

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Transformer Encoder (slot self-attention)
# ═══════════════════════════════════════════════════════════════════════════════
y_enc = 5.45

arrow(ax, 5.5, y_proj - 0.25, 5.5, y_enc + 0.35)

box(ax, 5.5, y_enc, 3.6, 0.65,
    "TransformerEncoder  (Pre-LN)",
    "2 layers · 4 heads · d_model=64\nself-attention over 5 slot tokens",
    color=C_ATTN, fontsize=9)

dim_label(ax, 5.5, y_enc - 0.50, "→ (B, 5, 64)  slot tokens updated")

# Self-attention loop annotation
ax.annotate("", xy=(7.5, y_enc + 0.05), xytext=(7.5, y_enc + 0.32),
            arrowprops=dict(arrowstyle="->", color=C_ATTN, lw=1.2,
                            connectionstyle="arc3,rad=-0.6"))
ax.text(8.0, y_enc + 0.18, "self-attn\n(slots↔slots)", fontsize=6.5,
        color=C_ATTN, ha="left", va="center")

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Mean pool + Concat  (NO cross-attention — ablation B finding)
# ═══════════════════════════════════════════════════════════════════════════════
y_pool = 4.4

arrow(ax, 5.5, y_enc - 0.33, 5.5, y_pool + 0.28, color=C_ATTN)

box(ax, 5.5, y_pool, 2.8, 0.52,
    "mean pool  (slots dim)",
    "→ slots_enc  (B, 64)",
    color=C_ATTN, fontsize=9)

# State path arrives from state_proj
arrow(ax, 11.2, y_state_proj - 0.25, 11.2, y_pool + 0.10, color=C_PROJ)
ax.text(10.6, y_pool + 0.14, "state_enc\n(B, 64)", ha="right", fontsize=6.5, color=C_PROJ)

dim_label(ax, 5.5, y_pool - 0.43, "→ slots_enc (B, 64)")

# Ablation B annotation
ax.text(7.8, y_pool + 0.05,
        "✦ Ablation B: removing cross-attention\n"
        "  improves all metrics — paired tokens\n"
        "  already encode the cmd↔fine alignment.",
        fontsize=7, color="#555555", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd", alpha=0.85))

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6 — Concat + split to actor / critic
# ═══════════════════════════════════════════════════════════════════════════════
y_cat = 3.25

arrow(ax, 5.5, y_pool - 0.26, 7.0, y_cat + 0.28, color=C_ATTN)
arrow(ax, 11.2, y_pool + 0.10, 8.5, y_cat + 0.28, color=C_PROJ)

box(ax, 7.0, y_cat, 3.2, 0.50,
    "concat(state_enc,  slots_enc)",
    "shared latent  →  128 dims",
    color="#6d9eeb", fontsize=9)

# ── Actor ──
y_actor = 1.95
arrow(ax, 5.5, y_cat - 0.25, 4.5, y_actor + 0.33)
box(ax, 4.5, y_actor, 2.8, 0.62,
    "Actor MLP",
    "Linear(128→128)→LN→GELU\n× 2  +  skip connections",
    color=C_MLP, fontsize=9)
arrow(ax, 4.5, y_actor - 0.31, 4.5, 1.05)
box(ax, 4.5, 0.75, 2.2, 0.55,
    "action  (7 dims)",
    "Tanh → × residual_scale",
    color=C_OUT, alpha=0.9, fontsize=8.5)

# ── Critic ──
y_critic = 1.95
arrow(ax, 8.5, y_cat - 0.25, 9.5, y_critic + 0.33)
box(ax, 9.5, y_critic, 2.8, 0.62,
    "Critic MLP",
    "Linear(128→128)→LN→GELU\n× 3  +  skip connections",
    color=C_MLP, fontsize=9)
arrow(ax, 9.5, y_critic - 0.31, 9.5, 1.05)
box(ax, 9.5, 0.75, 2.2, 0.55,
    "value  (scalar)",
    "for PPO advantage est.",
    color=C_OUT, alpha=0.9, fontsize=8.5)

# ═══════════════════════════════════════════════════════════════════════════════
# Legend
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_OBS_ROBOT,  "robot state"),
    (C_OBS_FINE,   "fine lookahead"),
    (C_OBS_COARSE, "coarse lookahead"),
    (C_OBS_CMD,    "cmd history"),
    (C_OBS_TRAJ,   "traj one-hot"),
    (C_PROJ,       "linear projection"),
    (C_ATTN,       "self-attention (no cross-attn)"),
    (C_MLP,        "MLP head"),
]
lx, ly = 0.3, 4.5
for i, (col, label) in enumerate(legend_items):
    rect = FancyBboxPatch((lx, ly - i*0.38 - 0.13), 0.28, 0.26,
                          boxstyle="round,pad=0.02", fc=col, ec="none", alpha=0.85, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.38, ly - i*0.38, label, va="center", fontsize=7.5, color="#444444")

ax.text(lx + 0.14, ly + 0.28, "Legend", ha="center", fontsize=8,
        fontweight="bold", color="#444444")

# ═══════════════════════════════════════════════════════════════════════════════
# Key insight annotation
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(0.3, 2.5,
        "Key inductive bias:\ncmd[i] executes at t+i·20ms\n= when fine[i] says\ntarget will be there.\nPairing them in one\nslot token makes the\nattention learn\n'what's already queued\nvs what's needed'.",
        fontsize=7, color="#555555", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.8))

plt.tight_layout(pad=0.3)
out = "results/figures/transformer_architecture.png"
import pathlib; pathlib.Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"saved → {out}")
