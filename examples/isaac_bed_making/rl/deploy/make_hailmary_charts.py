"""Render the Hail Mary IRL bed-making charts on the robot (matplotlib).
Reads archived approach telemetry + LiDAR clouds from /home/unitree/hailmary_{hero,success}/,
plus draw/grasp telemetry transcribed from the run logs. Writes PNGs to /home/unitree/hailmary_charts/.

Two real runs, both SUCCESSES, wildly different telemetry:
  HERO  (R_final 0.404): touch pads read EMPTY (a side/outside *pinch* the front pads can't sense),
                         waist twisted 23.9 deg as the pinch dragged the sheet.
  CLEAN (R_final 0.41):  touch pad fired (index 17 -> FABRIC), waist barely moved (0.2 deg).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/unitree/hailmary_charts"
os.makedirs(OUT, exist_ok=True)
HERO, CLEAN = "/home/unitree/hailmary_hero", "/home/unitree/hailmary_success"
C_HERO, C_CLEAN = "#d1495b", "#2e86ab"   # hero=red, clean=blue
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

def load_approach(d):
    t, r = [], []
    with open(os.path.join(d, "run_approach_telem.jsonl")) as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("R") is not None:
                t.append(o["t"]); r.append(o["R"])
    return np.array(t), np.array(r)

# ---- Fig 1: LiDAR-gated approach R(t) ---------------------------------------
th, rh = load_approach(HERO); tc, rc = load_approach(CLEAN)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tc, rc, "-o", ms=3, color=C_CLEAN, label="clean run (R_final 0.41 m)")
ax.plot(th, rh, "-o", ms=3, color=C_HERO, label="hero run (R_final 0.40 m)")
ax.axhline(0.425, ls="--", color="gray", lw=1, label="target R (stop)")
ax.set_xlabel("time since walk start (s)"); ax.set_ylabel("LiDAR forward range to bed  R (m)")
ax.set_title("Perception-gated approach — odom-free, LiDAR-range closed loop")
ax.annotate("StopMove\n(arrived)", xy=(th[-1], rh[-1]), xytext=(th[-1]-2.2, 0.62),
            arrowprops=dict(arrowstyle="->", color=C_HERO), color=C_HERO, fontsize=9)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_approach.png"); plt.close(fig)

# ---- draw telemetry (transcribed from run logs) -----------------------------
hero_t   = [0.00,0.71,1.45,2.22,3.03,3.87,4.60,5.36,6.09,6.92]
hero_yaw = [0.0, 0.3, 16.6,23.7,23.7,23.7,23.7,23.7,23.7,23.7]
hero_fol = [0.065,0.325,0.111,0.123,0.119,0.118,0.119,0.117,0.118,0.119]
hero_tau = [2.19,4.38,4.25,5.06,4.81,4.69,4.81,4.44,4.81,4.69]
cln_t    = [0.00,0.72,1.46,2.18,3.04,3.78,4.54,5.27,6.01,6.71]
cln_yaw  = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cln_fol  = [0.065,0.324,0.109,0.089,0.091,0.094,0.092,0.091,0.090,0.092]
cln_tau  = [2.62,3.12,3.50,3.62,3.56,3.44,3.56,3.56,3.62,3.44]

# ---- Fig 2: the MESSY-TELEMETRY headline — waist twist during the draw ------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(cln_t, cln_yaw, "-o", ms=4, color=C_CLEAN, label="clean run — peak 0.2 deg")
ax.plot(hero_t, hero_yaw, "-o", ms=4, color=C_HERO, label="hero run — peak 23.9 deg")
ax.set_xlabel("time since draw start (s)"); ax.set_ylabel("waist-yaw deflection (deg)")
ax.set_title("Same task, two successes — torso twist tells very different stories")
ax.annotate("pinch drags the sheet →\ntorso loads to 23.9 deg", xy=(2.2, 23.7), xytext=(2.6, 14),
            arrowprops=dict(arrowstyle="->", color=C_HERO), color=C_HERO, fontsize=9)
ax.legend(loc="center right", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_waist_twist.png"); plt.close(fig)

# ---- Fig 3: grasp force per finger pad (the false-negative) ------------------
fingers = ["index", "middle", "ring", "pinky"]
hero_pad  = [0, 0, 0, 0]      # sustained force-rise, thumb (self-contact) excluded
clean_pad = [17, 0, 0, 0]
x = np.arange(len(fingers)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(x - w/2, clean_pad, w, color=C_CLEAN, label="clean run → FABRIC")
ax.bar(x + w/2, hero_pad,  w, color=C_HERO,  label="hero run → reads EMPTY")
ax.axhline(6, ls="--", color="gray", lw=1)
ax.text(len(fingers)-0.55, 6.5, "fabric threshold = 6", fontsize=8, color="gray", ha="right")
for xi, v in zip(x - w/2, clean_pad): ax.text(xi, v + 0.4, str(v), ha="center", fontsize=9, color=C_CLEAN)
for xi, v in zip(x + w/2, hero_pad):  ax.text(xi, 0.4, str(v), ha="center", fontsize=9, color=C_HERO, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(fingers); ax.set_ylim(0, 20)
ax.set_ylabel("sustained touch-pad force rise (u16)")
ax.set_title("Touch-confirm false-negative — the hero grab fired NO pad", fontsize=11.5)
ax.text(0.5, -0.20, "hero pinched with the finger SIDES (no front-pad contact) — the 23.9° waist twist is the proof fabric was loaded",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#444")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_grasp_pads.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: LiDAR top-down, start vs arrival (hero) -------------------------
def topdown(ax, pts, title):
    m = (pts[:,2] > -0.6) & (pts[:,2] < 0.4)           # bed-height-ish band
    p = pts[m]
    ax.scatter(p[:,0], p[:,1], s=0.5, c=p[:,2], cmap="viridis", alpha=0.5)
    ax.scatter([0],[0], c="red", marker="^", s=80, label="robot")
    # forward corridor used by the gate
    ax.axhline(0.20, ls=":", color="k", lw=0.8); ax.axhline(-0.20, ls=":", color="k", lw=0.8)
    ax.set_xlabel("x forward (m)"); ax.set_ylabel("y left (m)"); ax.set_title(title)
    ax.set_xlim(-0.5, 3.0); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal"); ax.legend(fontsize=8)
start = np.load(f"{HERO}/run_lidar_start.npy"); arr = np.load(f"{HERO}/run_lidar_arrival.npy")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
topdown(axes[0], start, "LiDAR top-down — START (R 1.14 m)")
topdown(axes[1], arr,   "LiDAR top-down — ARRIVED (R 0.40 m)")
fig.suptitle("Bed seen as the forward obstacle; robot creeps until R hits the stop", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_lidar_topdown.png", bbox_inches="tight"); plt.close(fig)

print("charts written:", sorted(os.listdir(OUT)))
