#!/usr/bin/env python3
"""s2_pickstack_ergo.py - RULA/REBA posture-score figure for the S2 two-handed
block pick-and-stack pilot (IEEE single-column, ~3.4 in).

Panels stacked vertically (RULA over REBA) so each gets full column width. NOT
a spatial "ergonomic boundary": RULA/REBA are joint-angle functions, so what is
drawn is the published category thresholds as horizontal risk bands under the
per-frame grand scores over the task cycle.

Series:
  human capture   real per-frame scores of the human doing the task.
  robot mimic     the robot replaying that capture, scored AS IF human.
  robot native    the robot-native motion, scored AS IF human.
Robot stances use the MEASURED knee bend (mimic ~39 deg default H1-2 stance,
native ~62 deg deliberate crouch; knee_probe.py). The default stance is applied
to BOTH robot series; only the deliberate crouch differs.

MANDATORY CAVEAT (printed on the figure): the robot series are a posture-
similarity statement, NOT a claim about robot injury risk.

Outputs: s2_pickstack_ergo.pdf (vector) + .png.
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import matplotlib.patheffects as pe          # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments", "scripts")
DATA = os.path.join(EXP, "..", "iphone_data", "Blockdata0725", "Block_S2_1", "b2g")
sys.path.insert(0, EXP)
from retarget_arm import ArmChainFK                       # noqa: E402
from angles import compute_frame_angles, legs_support_info  # noqa: E402
from rula_reba_score import rula_grand, reba_grand, score_csv  # noqa: E402

RJ = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
      "right_elbow"]
LJ = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
      "left_elbow"]
UP = (0.0, 0.0, 1.0)

RULA_BANDS = [(1, 2, "acceptable"), (3, 4, "investigate"),
              (5, 6, "change soon"), (7, 7, "change now")]
REBA_BANDS = [(1, 1, "negligible"), (2, 3, "low"), (4, 7, "medium"),
              (8, 10, "high"), (11, 15, "very high")]
BAND_COLORS = ["#e8f5e9", "#fff8e1", "#ffe0b2", "#ffcdd2", "#ef9a9a"]
HUMAN_C, MIMIC_C, NATIVE_C = "#d1495b", "#e69f00", "#00798c"


def robot_scores(traj_csv, urdf, knee_deg):
    """Score a robot arm trajectory as if it were a person (same code path).
    knee_deg models the measured stance: thigh and shin each tilt knee_deg/2
    so the knee interior angle reads 180 - knee_deg through knee_flexion()."""
    fk = {"left": ArmChainFK(urdf, "left"), "right": ArmChainFK(urdf, "right")}
    L1 = L2 = 0.45
    b = np.radians(knee_deg) / 2.0
    rows = list(csv.DictReader(open(traj_csv)))
    out = []
    for x in rows:
        j = {}
        for side, cols in (("left", LJ), ("right", RJ)):
            q = [float(x[c]) for c in cols]
            sh, el, wr = fk[side].positions(q)
            j[f"{side}_shoulder"] = tuple(sh)
            j[f"{side}_elbow"] = tuple(el)
            j[f"{side}_wrist"] = tuple(wr)
        smid = 0.5 * (np.array(j["left_shoulder"]) + np.array(j["right_shoulder"]))
        for side, sgn in (("left", 1), ("right", -1)):
            hip = np.array([smid[0], sgn * 0.09, smid[2] - 0.45])
            knee = hip + [L1 * np.sin(b), 0, -L1 * np.cos(b)]
            ankle = knee + [-L2 * np.sin(b), 0, -L2 * np.cos(b)]
            j[f"{side}_hip"] = tuple(hip)
            j[f"{side}_knee"] = tuple(knee)
            j[f"{side}_ankle"] = tuple(ankle)
        ang = compute_frame_angles(j, up=UP)
        legs = legs_support_info(j, up=UP)
        out.append((float(x["t_s"]), rula_grand(ang, legs), reba_grand(ang, legs)))
    return out


def occupancy(name, scores, bands):
    v = np.array([s for s in scores if s is not None], float)
    parts = [f"{lab} {np.mean((v >= lo) & (v <= hi)) * 100:.0f}%"
             for lo, hi, lab in bands]
    print(f"  {name:14} mean {v.mean():.2f}  " + "  ".join(parts))
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human-joints", default=os.path.join(DATA, "mono_trim.csv"))
    p.add_argument("--human-up", default="0,-1,0")
    p.add_argument("--mimic", default=os.path.join(DATA, "traj_mono_trim.csv"))
    p.add_argument("--native", default=os.path.join(HERE, "s2_bricklay_native.csv"))
    p.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"))
    p.add_argument("--robot-knee-deg", type=float, default=39.0,
                   help="measured default-stance knee bend, both robot series")
    p.add_argument("--native-knee-deg", type=float, default=62.0,
                   help="measured native crouch knee bend (knee_probe.py)")
    p.add_argument("--stem", default=os.path.join(HERE, "s2_pickstack_ergo"))
    p.add_argument("--rula-only", action="store_true",
                   help="single RULA panel only (IROS WS dropped REBA); writes "
                        "<stem>_rula.*")
    args = p.parse_args()

    up_h = tuple(float(v) for v in args.human_up.split(","))
    hrows = score_csv(args.human_joints, 0.5, up_h)
    nat = robot_scores(args.native, args.urdf, args.native_knee_deg)
    mim = robot_scores(args.mimic, args.urdf, args.robot_knee_deg)
    print(f"[ergo] stance knees: mimic {args.robot_knee_deg:.0f}, "
          f"native {args.native_knee_deg:.0f} deg (measured)")
    print("[ergo] band occupancy (posture-only lower bounds, load = 0):")
    print("  RULA:")
    hR = occupancy("human", [r[1] for r in hrows], RULA_BANDS)
    mR = occupancy("mimic", [r[1] for r in mim], RULA_BANDS)
    nR = occupancy("native", [r[1] for r in nat], RULA_BANDS)
    print("  REBA:")
    hB = occupancy("human", [r[2] for r in hrows], REBA_BANDS)
    mB = occupancy("mimic", [r[2] for r in mim], REBA_BANDS)
    nB = occupancy("native", [r[2] for r in nat], REBA_BANDS)

    from matplotlib.lines import Line2D
    if args.rula_only:
        fig, ax0 = plt.subplots(figsize=(4.8, 5.0))
        panels = [(ax0, RULA_BANDS, (hR, mR, nR), "RULA grand score", 7,
                   [1, 3, 5, 7])]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
        panels = [(axes[0], RULA_BANDS, (hR, mR, nR), "RULA grand score", 7,
                   [1, 3, 5, 7]),
                  (axes[1], REBA_BANDS, (hB, mB, nB), "REBA grand score", 12,
                   [1, 3, 5, 7, 9, 11])]
    series = [("human demonstration", HUMAN_C, 1.7, "-", 5),
              ("Human-Derived Motion", MIMIC_C, 1.4, "--", 6),
              ("Robot-Specific Motion", NATIVE_C, 1.7, "-", 4)]
    halo = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]
    for ax, bands, tri, ttl, ymax, yticks in panels:
        for (lo, hi, lab), c in zip(bands, BAND_COLORS):
            ax.axhspan(lo - 0.5, hi + 0.5, color=c, zorder=0)
            ax.text(101, (lo + hi) / 2, lab, fontsize=6.5, va="center",
                    color="#555555")
        means = []
        for v, (lab, col, lw, ls, z) in zip(tri, series):
            t = np.linspace(0, 100, len(v))
            ln, = ax.step(t, v, where="mid", color=col, lw=lw, ls=ls, zorder=z)
            if ls == "--":                       # white halo so it reads on red
                ln.set_path_effects(halo)
            means.append((col, float(v.mean())))
        # per-panel means, colored to each series, in the empty top band
        ax.text(0.30, 0.905, "mean", transform=ax.transAxes, fontsize=8,
                family="monospace", color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999",
                          alpha=0.9))
        for k, (col, m) in enumerate(means):
            ax.text(0.475 + 0.155 * k, 0.905, f"{m:.2f}", transform=ax.transAxes,
                    fontsize=8, family="monospace", color=col)
        ax.set_ylim(0.4, ymax + 0.6)
        ax.set_xlim(0, 100)
        ax.set_yticks(yticks)
        ax.set_ylabel("grand score")
        ax.set_xlabel("task cycle (%)")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.15, axis="x")

    handles = [Line2D([0], [0], color=c, lw=lw + 0.3, ls=ls)
               for (_, c, lw, ls, _) in series]
    if args.rula_only:
        fig.legend(handles, [s[0] for s in series], loc="lower center", ncol=1,
                   fontsize=8, framealpha=0.9, handlelength=1.8, labelspacing=0.3,
                   bbox_to_anchor=(0.5, 0.04))
        fig.suptitle("Block pick-and-stack: RULA posture score "
                     "(posture-only, load = 0)", fontsize=10)
        fig.text(0.5, 0.205, "robot trajectories scored with the human posture "
                 "tables: a posture-similarity\nstatement, not robot injury risk",
                 ha="center", fontsize=6.8, style="italic", color="#444444")
        fig.tight_layout(rect=[0, 0.30, 1, 0.95])
        stem = args.stem + "_rula"
    else:
        fig.legend(handles, [s[0] for s in series], loc="lower center", ncol=3,
                   fontsize=8, framealpha=0.9, handlelength=1.8, columnspacing=1.3,
                   bbox_to_anchor=(0.5, -0.015))
        fig.suptitle("Block pick-and-stack: RULA and REBA posture scores "
                     "(posture-only, load = 0)", fontsize=10)
        fig.text(0.5, 0.072, "robot trajectories scored with the human posture "
                 "tables: a posture-similarity statement, not robot injury risk",
                 ha="center", fontsize=6.8, style="italic", color="#444444")
        fig.tight_layout(rect=[0, 0.12, 1, 0.95], w_pad=2.5)
        stem = args.stem
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[ergo] wrote {os.path.basename(stem)}.pdf + .png")


if __name__ == "__main__":
    main()
