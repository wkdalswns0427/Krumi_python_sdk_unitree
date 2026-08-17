#!/usr/bin/env python3
"""s2_pickstack_paths.py - trajectory figure for the S2 two-handed block
pick-and-stack pilot (IEEE single-column, ~3.4 in).

Single side-view (x-z) panel: block-center path for both conditions, with
faint per-hand paths behind, and the pick + three stack layers marked. The
top-view panel was dropped (a dense unreadable tangle at column width); the
two-handedness it showed is instead quantified by the hand-separation stats
(F1.5), printed and saved to CSV for the caption.

This is NOT the M2 bricklaying capture (a separate motion in the error-budget
study); it is a designed two-handed pick-and-stack task.

Outputs: s2_pickstack_paths.pdf (vector) + .png, and s2_pickstack_handsep.csv.

Usage:
    /usr/bin/python3 s2_pickstack_paths.py
    /usr/bin/python3 s2_pickstack_paths.py --human s2_bricklay_human.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType, editable text
    "font.size": 8, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments"))
from h12_experiments.fk_h12 import H12ArmFK          # noqa: E402
from h12_experiments.joints import ARM_CHAIN         # noqa: E402

RJ = ARM_CHAIN["right"][:4]
LJ = ARM_CHAIN["left"][:4]
URDF = os.path.expanduser("~/mj_ws/assets/h1_2_description/h1_2.urdf")
HUMAN_C, NATIVE_C = "#d1495b", "#00798c"           # colorblind-safe red / teal
STATION_C = "#edae49"
NEW_LABELS = ["Pick", "Layer 1", "Layer 2", "Layer 3"]


def hands(path):
    fkR, fkL = H12ArmFK(URDF, "right"), H12ArmFK(URDF, "left")
    rows = list(csv.DictReader(open(path)))
    qR = np.array([[float(x[j]) for j in RJ] for x in rows])
    qL = np.array([[float(x[j]) for j in LJ] for x in rows])
    R = np.array([fkR.ee_position(list(q) + [0, 0, 0]) for q in qR])
    L = np.array([fkL.ee_position(list(q) + [0, 0, 0]) for q in qL])
    return L, R, 0.5 * (L + R)


def sep_stats(L, R, C, block_w_cm):
    """Hand-separation stats in cm, whole cycle and manipulation window.

    The whole cycle includes the open-hand home/rest pose (no block held), so
    for the graspability claim the manipulation window (block-center out in
    front, x > 0.15 m) is the physically relevant one. A rigid two-hand grasp
    needs the separation near the block width; the mimic averaging well over
    2x the block width is the evidence it was never graspable."""
    d = np.linalg.norm(L - R, axis=1) * 100.0
    manip = C[:, 0] > 0.15
    dm = d[manip]
    return dict(mean=float(d.mean()), min=float(d.min()), max=float(d.max()),
                range=float(d.max() - d.min()),
                manip_mean=float(dm.mean()), manip_sd=float(dm.std()),
                manip_range=float(dm.max() - dm.min()),
                over_block=float(dm.mean() / block_w_cm))


DEFAULT_HUMAN = os.path.join(
    HERE, "..", "experiments", "iphone_data", "Blockdata0725", "Block_S2_1",
    "b2g", "traj_mono_trim.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=DEFAULT_HUMAN,
                    help="human baseline trajectory CSV (default: retargeted "
                         "Block_S2_1 capture)")
    ap.add_argument("--native", default=os.path.join(HERE, "s2_bricklay_native.csv"))
    ap.add_argument("--spec", default=os.path.join(HERE, "s2_bricklay_spec.json"))
    ap.add_argument("--stem", default=os.path.join(HERE, "s2_pickstack_paths"))
    args = ap.parse_args()

    natL, natR, natC = hands(args.native)
    humL, humR, humC = hands(args.human)
    spec = json.load(open(args.spec))
    st = np.array(spec["stations"])
    labels = NEW_LABELS[:len(st)]

    # ---- F1.5 hand-separation stats (printed + CSV) ----
    block_w = spec.get("block_width", 0.238) * 100.0     # cm
    ns = sep_stats(natL, natR, natC, block_w)
    hs = sep_stats(humL, humR, humC, block_w)
    print(f"[handsep] distance between the two hands, cm (block width {block_w:.1f} cm):")
    print(f"    {'condition':16}| whole cycle: {'mean':>5}{'min':>6}{'max':>6}"
          f"{'range':>6} | manip: {'mean':>5}{'sd':>5}{'range':>6}{'/block':>7}")
    for name, s in (("robot-native", ns), ("human-mimic", hs)):
        print(f"    {name:16}|              {s['mean']:>5.1f}{s['min']:>6.1f}"
              f"{s['max']:>6.1f}{s['range']:>6.1f} |        {s['manip_mean']:>5.1f}"
              f"{s['manip_sd']:>5.1f}{s['manip_range']:>6.1f}{s['over_block']:>6.2f}x")
    csvp = os.path.join(HERE, "s2_pickstack_handsep.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "block_width_cm", "whole_mean_cm", "whole_min_cm",
                    "whole_max_cm", "whole_range_cm", "manip_mean_cm",
                    "manip_sd_cm", "manip_range_cm", "manip_mean_over_block"])
        for name, s in (("robot_native", ns), ("human_mimic", hs)):
            w.writerow([name, f"{block_w:.1f}"]
                       + [f"{s[k]:.2f}" for k in ("mean", "min", "max", "range",
                          "manip_mean", "manip_sd", "manip_range", "over_block")])
    print(f"[handsep] wrote {os.path.basename(csvp)}")

    # ---- single side-view (x-z) panel ----
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    i, j = 0, 2                                    # x horizontal, z vertical
    off_side = {"Pick": (-4, -12, "right", "top"),
                "Layer 1": (9, -4, "left", "center"),
                "Layer 2": (9, 0, "left", "center"),
                "Layer 3": (9, 3, "left", "center")}
    for H, col in ((humL, HUMAN_C), (humR, HUMAN_C),
                   (natL, NATIVE_C), (natR, NATIVE_C)):
        ax.plot(H[:, i], H[:, j], "-", color=col, lw=0.6, alpha=0.22, zorder=1)
    ax.plot(humC[:, i], humC[:, j], "-", color=HUMAN_C, lw=2.0, zorder=3,
            label="human baseline (retargeted capture)")
    ax.plot(natC[:, i], natC[:, j], "-", color=NATIVE_C, lw=2.4, zorder=4,
            label="robot-native (direct transfers)")
    for lab, s in zip(labels, st):
        ax.scatter(s[i], s[j], s=70, marker="s" if lab == "Pick" else "o",
                   color=STATION_C, edgecolor="k", lw=0.6, zorder=6)
        dx, dy, ha, va = off_side.get(lab, (9, 3, "left", "center"))
        ax.annotate(lab, (s[i], s[j]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, ha=ha, va=va, zorder=7)
    ax.set_xlabel("x  (forward +, m)")
    ax.set_ylabel("z  (up +, m)")
    ax.set_title("Two-handed block pick-and-stack:\nblock-center and hand "
                 "trajectories (side view)", fontsize=10)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1,
              fontsize=8, framealpha=0.9, handlelength=1.8, labelspacing=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[plot] wrote {os.path.basename(args.stem)}.pdf + .png")


if __name__ == "__main__":
    main()
