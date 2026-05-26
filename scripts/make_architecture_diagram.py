#!/usr/bin/env python3
"""Generate transformer architecture diagram PNG.

Usage:
    python scripts/make_architecture_diagram.py
    python scripts/make_architecture_diagram.py --out results/figures/architecture.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Palette ────────────────────────────────────────────────────────────────────
OBS_FC,  OBS_EC  = '#dbeafe', '#1d4ed8'   # blue   — observations
SLOT_FC, SLOT_EC = '#ffe4e6', '#be123c'   # rose   — slot tokens
OP_FC,   OP_EC   = '#d1fae5', '#059669'   # green  — operations (linear, pool)
TFM_FC,  TFM_EC  = '#fef3c7', '#b45309'   # amber  — transformer encoder
ENC_FC,  ENC_EC  = '#e0f2fe', '#0369a1'   # sky    — encoded representations
FUS_FC,  FUS_EC  = '#f1f5f9', '#475569'   # slate  — fusion
ACT_FC,  ACT_EC  = '#fce7f3', '#be185d'   # pink   — actor head
CRT_FC,  CRT_EC  = '#ede9fe', '#6d28d9'   # violet — critic head

ARROW_C = '#555555'


# ── Helpers ────────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, text, fc, ec, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.10', facecolor=fc, edgecolor=ec,
        linewidth=1.5, zorder=4))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal',
            multialignment='center', zorder=5)


def slot_box(ax, cx, cy, w, full_h, top_text, bot_text, fc, ec, fs=6.8):
    """Single rounded box split into top/bottom halves by a dashed rule."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - full_h / 2), w, full_h,
        boxstyle='round,pad=0.08', facecolor=fc, edgecolor=ec,
        linewidth=1.5, zorder=4))
    ax.plot([cx - w / 2 + 0.08, cx + w / 2 - 0.08], [cy, cy],
            color=ec, lw=0.9, ls='--', zorder=5)
    ax.text(cx, cy + full_h / 4, top_text, ha='center', va='center',
            fontsize=fs, multialignment='center', zorder=5)
    ax.text(cx, cy - full_h / 4, bot_text, ha='center', va='center',
            fontsize=fs, multialignment='center', zorder=5)


def arr(ax, x1, y1, x2, y2, c=ARROW_C, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=6)


def note(ax, x, y, t, fs=8, c='#666666', ha='center', **kw):
    ax.text(x, y, t, ha=ha, va='center', fontsize=fs, color=c,
            multialignment='center', **kw)


# ── Main ───────────────────────────────────────────────────────────────────────
def make_figure(out: Path):
    FW, FH = 22, 17
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    BH = 0.60   # standard box height

    # ═════════════════════════════════════════════════════════════════════════
    # Y LEVELS (bottom → top)
    # Key design choice: State Encoding and Slot Encoding share the same Y
    # so fusion arrows are perfectly symmetric.
    # ═════════════════════════════════════════════════════════════════════════
    Y_IN   = 3.3    # obs inputs (left) AND slot tokens (right) — same level
    Y_SLPR = 5.6    # slot projection + positional embedding
    Y_TFMR = 7.6    # transformer encoder
    Y_POOL = 9.4    # mean pool
    Y_ENC  = 11.0   # State Encoding (left) AND Slot Encoding (right) — same level
    Y_FUSE = 12.8   # fusion concat
    Y_HEAD = 14.6   # actor / critic heads

    # ═════════════════════════════════════════════════════════════════════════
    # LEFT PATH — State Encoder
    # ═════════════════════════════════════════════════════════════════════════
    LX = 3.3         # centre of left obs column
    BW = 3.4         # obs box width

    # Three stacked observation boxes
    Y_ROB = Y_IN + 0.95
    Y_COA = Y_IN
    Y_TRJ = Y_IN - 0.95
    for yy, lbl in [(Y_ROB, 'Robot State  (30D)'),
                    (Y_COA, 'Coarse Lookahead  (12D)'),
                    (Y_TRJ, 'Trajectory ID  (3D)')]:
        box(ax, LX, yy, BW, BH, lbl, OBS_FC, OBS_EC, fs=9)

    # Bracket merging the three inputs
    xbr = LX + BW / 2 + 0.14
    ax.plot([xbr] * 2, [Y_TRJ - BH / 2 + 0.08, Y_ROB + BH / 2 - 0.08],
            color='#888888', lw=1.6, zorder=3)
    for yy in (Y_ROB, Y_COA, Y_TRJ):
        ax.plot([LX + BW / 2, xbr], [yy, yy], color='#888888', lw=1.1, zorder=3)

    # Concatenate
    xcat = xbr + 1.60
    arr(ax, xbr, Y_COA, xcat - 0.80, Y_COA)
    box(ax, xcat, Y_COA, 1.60, BH, 'Concatenate\n(45D)', OP_FC, OP_EC, fs=9)

    # Linear + ReLU
    xlin = xcat + 1.80
    arr(ax, xcat + 0.80, Y_COA, xlin - 0.74, Y_COA)
    box(ax, xlin, Y_COA, 1.48, BH, 'Linear\n+ ReLU', OP_FC, OP_EC, fs=9)

    # Long vertical arrow → State Encoding
    arr(ax, xlin, Y_COA + BH / 2, xlin, Y_ENC - 0.37)
    note(ax, xlin + 0.50, (Y_COA + Y_ENC) / 2, '64D', fs=8.5, c='#0369a1')

    # State Encoding box
    box(ax, xlin, Y_ENC, 2.3, 0.75,
        'State Encoding  (64D)', ENC_FC, ENC_EC, fs=10.5, bold=True)

    # Section badge
    note(ax, 0.85, (Y_IN + Y_ENC) / 2, 'State\nEncoder',
         fs=12, c='#1d4ed8', fontweight='bold', ha='left')

    # ═════════════════════════════════════════════════════════════════════════
    # RIGHT PATH — Slot Transformer
    # ═════════════════════════════════════════════════════════════════════════
    N       = 5
    SW      = 1.60                              # slot box width
    SH      = 1.45                              # slot box total height
    slot_xs = np.linspace(11.0, 18.5, N)       # [11.0, 12.875, 14.75, 16.625, 18.5]
    RX      = float(np.mean(slot_xs))           # 14.75

    TIME = ['t + 20 ms', 't + 40 ms', 't + 60 ms', 't + 80 ms', 't + 100 ms']
    for i, sx in enumerate(slot_xs):
        slot_box(ax, sx, Y_IN, SW, SH,
                 f'Fine Lookahead {i}\n(3D)',
                 f'Command History {i}\n(7D)',
                 SLOT_FC, SLOT_EC, fs=7.0)
        note(ax, sx, Y_IN + SH / 2 + 0.25, TIME[i], fs=7.2, c='#be123c')

    note(ax, RX, Y_IN - SH / 2 - 0.35,
         'Fine Lookahead i — target position at delay step i (3D)  ·  '
         'Command History i — queued joint setpoint (7D)',
         fs=7.5, c='#888888')

    # Slot Projection + Positional Embedding
    proj_w = slot_xs[-1] - slot_xs[0] + SW + 0.40
    for sx in slot_xs:
        arr(ax, sx, Y_IN + SH / 2, sx, Y_SLPR - 0.32)
    box(ax, RX, Y_SLPR, proj_w, BH,
        'Linear Projection  (× 5)    →    d_model = 64    +    Positional Embedding',
        OP_FC, OP_EC, fs=9)

    # Transformer Encoder
    arr(ax, RX, Y_SLPR + BH / 2, RX, Y_TFMR - 0.37)
    box(ax, RX, Y_TFMR, proj_w, 0.75,
        'Transformer Encoder  ·  2 Layers  ·  4 Heads  ·  d_model = 64  ·  pre-LayerNorm',
        TFM_FC, TFM_EC, fs=9.5, bold=True)

    # Mean Pool
    arr(ax, RX, Y_TFMR + 0.375, RX, Y_POOL - 0.32)
    box(ax, RX, Y_POOL, 3.0, BH, 'Mean Pool  (over 5 slots)', OP_FC, OP_EC, fs=9)

    # Slot Encoding
    arr(ax, RX, Y_POOL + BH / 2, RX, Y_ENC - 0.37)
    note(ax, RX + 0.65, (Y_POOL + Y_ENC) / 2, '64D', fs=8.5, c='#0369a1')

    box(ax, RX, Y_ENC, 2.3, 0.75,
        'Slot Encoding  (64D)', ENC_FC, ENC_EC, fs=10.5, bold=True)

    # Section badge
    note(ax, FW - 0.85, (Y_IN + Y_ENC) / 2, 'Slot\nTransformer',
         fs=12, c='#b45309', fontweight='bold', ha='right')

    # ═════════════════════════════════════════════════════════════════════════
    # FUSION — symmetric because both encodings sit at Y_ENC
    # ═════════════════════════════════════════════════════════════════════════
    FX = (xlin + RX) / 2     # midpoint  ≈ 11.5

    arr(ax, xlin, Y_ENC + 0.375, FX - 1.60, Y_FUSE - 0.39)
    arr(ax, RX,   Y_ENC + 0.375, FX + 1.60, Y_FUSE - 0.39)

    box(ax, FX, Y_FUSE, 5.0, 0.78,
        'Concatenate  ( State Encoding  +  Slot Encoding )  →  128D',
        FUS_FC, FUS_EC, fs=11, bold=True)

    # ═════════════════════════════════════════════════════════════════════════
    # ACTOR / CRITIC HEADS
    # ═════════════════════════════════════════════════════════════════════════
    ax_cx = FX - 3.5
    cr_cx = FX + 3.5

    arr(ax, FX - 1.30, Y_FUSE + 0.39, ax_cx + 1.45, Y_HEAD - 0.39)
    arr(ax, FX + 1.30, Y_FUSE + 0.39, cr_cx - 1.50, Y_HEAD - 0.39)

    box(ax, ax_cx, Y_HEAD, 4.1, 0.78,
        'Actor MLP  [ 256 → 256 ]  →  7D', ACT_FC, ACT_EC, fs=10, bold=True)
    box(ax, cr_cx, Y_HEAD, 4.6, 0.78,
        'Critic MLP  [ 256 → 256 → 256 ]  →  1D', CRT_FC, CRT_EC, fs=10, bold=True)

    note(ax, ax_cx,  Y_HEAD + 0.67, 'Gaussian policy  (7-DoF joint residuals)', fs=8.5, c='#be185d')
    note(ax, cr_cx, Y_HEAD + 0.67, 'Value estimate', fs=8.5, c='#6d28d9')

    # ═════════════════════════════════════════════════════════════════════════
    # TITLE + SUBTITLE
    # ═════════════════════════════════════════════════════════════════════════
    note(ax, FW / 2, FH - 0.38,
         'Delay-Aware Transformer Policy — Architecture',
         fs=15, c='#1e293b', fontweight='bold')
    note(ax, FW / 2, FH - 0.88,
         'Output = clip( q_IK  +  residual × scale,  q_lim )   ·   '
         'Zero residual collapses to pure IK baseline',
         fs=9.5, c='#94a3b8')

    # ═════════════════════════════════════════════════════════════════════════
    # COLOUR LEGEND
    # ═════════════════════════════════════════════════════════════════════════
    legend_items = [
        (OBS_FC,  OBS_EC,  'Observation inputs'),
        (SLOT_FC, SLOT_EC, 'Slot tokens (Fine Lookahead ‖ Command History)'),
        (OP_FC,   OP_EC,   'Linear / Pooling operations'),
        (TFM_FC,  TFM_EC,  'Transformer Encoder'),
        (ENC_FC,  ENC_EC,  'Encoded representations'),
        (ACT_FC,  ACT_EC,  'Actor head'),
        (CRT_FC,  CRT_EC,  'Critic head'),
    ]
    handles = [mpatches.Patch(facecolor=fc, edgecolor=ec, label=lbl, linewidth=1.2)
               for fc, ec, lbl in legend_items]
    ax.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
              framealpha=0.92, edgecolor='#e2e8f0',
              bbox_to_anchor=(0.5, -0.01))

    # ═════════════════════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════════════════════
    plt.tight_layout(pad=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  saved → {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/figures/architecture.png')
    args = ap.parse_args()
    make_figure(Path(args.out))


if __name__ == '__main__':
    main()
