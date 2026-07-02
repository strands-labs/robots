"""Generate the BED_MAKING_IRL.md §10.9 result charts (run in unitree_deploy env on the robot)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/tmp"
RED, GREEN, DGREEN, TEAL, GRAY = "#c0392b", "#27ae60", "#1f8a4c", "#16a085", "#7f8c8d"

# ── Chart 1: grasp strength before vs after the sequenced-close fix ──────────────
labels = ["Weak grab\n(stale-bridge\ncurled thumb)", "Fixed grab\n(1 layer)", "Fixed handoff\n(index pad)"]
vals = [5, 51, 108]
fig, ax = plt.subplots(figsize=(7, 4.3))
bars = ax.bar(labels, vals, color=[RED, GREEN, DGREEN])
ax.axhline(4, ls="--", color=GRAY, lw=1)
ax.text(2.48, 6, "fabric threshold = 4", color=GRAY, ha="right", va="bottom", fontsize=9)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("strongest finger-pad force rise (u16)")
ax.set_title("Grasp strength: before vs after the sequenced-close fix\n"
             "~10–20× stronger once the fingers stop snagging the thumb")
ax.set_ylim(0, 122)
fig.tight_layout()
fig.savefig(f"{OUT}/grasp_strength_before_after.png", dpi=150)
plt.close(fig)

# ── Chart 2: empty-grab touch separation (calibration) ──────────────────────────
empty = [0, 0, 0, 0, 0, 1, 1]
one = [9, 12, 15, 23, 24, 39]
two = [81]
fig, ax = plt.subplots(figsize=(7, 4.3))
def strip(xc, ys, color, label):
    offs = np.linspace(-0.15, 0.15, len(ys)) if len(ys) > 1 else [0.0]
    ax.scatter([xc + o for o in offs], ys, s=70, color=color, zorder=3,
               edgecolor="white", linewidth=0.8, label=label)
strip(0, empty, RED, "empty / air (n=7)")
strip(1, one, GREEN, "1 layer (n=6)")
strip(2, two, TEAL, "2 layers (n=1)")
ax.axhline(4, ls="--", color=GRAY, lw=1)
ax.text(2.45, 5.5, "threshold = 4", color=GRAY, ha="right", fontsize=9)
ax.axhspan(1, 9, color="gray", alpha=0.08)
ax.text(2.45, 4.0, "clean gap (≤1 vs ≥9)", color=GRAY, ha="right", fontsize=8, style="italic")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["empty / air", "1 layer", "2 layers"])
ax.set_xlim(-0.5, 2.6)
ax.set_ylabel("strongest finger-pad sustained rise (u16)")
ax.set_title("Empty-grab touch separation (held-grip verdict, thumb pad excluded)\n"
             "empty floor ≤1  vs  fabric ≥9 — touch-only separates on this G1")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/empty_grab_separation.png", dpi=150)
plt.close(fig)

# ── Chart 3: draw-resistance signatures (empty ≈ free; only anchored stalls) ─────
cats = ["Empty /\nslipped", "Free sheet\n(autonomous)", "Free sheet,\nfirm (handoff)", "Anchored\ncover (stall)"]
follow = [0.217, 0.239, 0.300, 0.35]
tau = [4.75, 4.38, 7.25, 14.0]
cols = [GRAY, GREEN, DGREEN, RED]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
x = list(range(len(cats)))
b1 = a1.bar(x, follow, color=cols)
a1.axhline(0.28, ls="--", color=GRAY); a1.text(3.45, 0.288, "stall thr 0.28", ha="right", color=GRAY, fontsize=8)
a1.set_xticks(x); a1.set_xticklabels(cats, fontsize=8); a1.set_ylabel("peak following-error (rad)")
a1.set_title("Draw following-error"); a1.set_ylim(0, 0.40)
for bb, v in zip(b1, follow): a1.text(bb.get_x() + bb.get_width() / 2, v + 0.006, f"{v:.2f}", ha="center", fontsize=8)
b2 = a2.bar(x, tau, color=cols)
a2.axhline(11, ls="--", color=GRAY); a2.text(3.45, 11.3, "stall thr 11", ha="right", color=GRAY, fontsize=8)
a2.set_xticks(x); a2.set_xticklabels(cats, fontsize=8); a2.set_ylabel("peak joint torque (est.)")
a2.set_title("Draw joint torque"); a2.set_ylim(0, 15.5)
for bb, v in zip(b2, tau): a2.text(bb.get_x() + bb.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center", fontsize=8)
fig.suptitle("Draw-resistance signatures: empty ≈ free (why the hand, not the arm, must catch a slip) — "
             "only an anchored load stalls", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/draw_signatures.png", dpi=150)
plt.close(fig)

print("wrote 3 charts to /tmp")
