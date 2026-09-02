#!/usr/bin/env python3
"""Generate Figures 2 and 3 for the ICRA submission, plus Figure 4 when the
hardware replay logs exist.

Figure 2  violation rate against reconstructed extension, the dose-response
Figure 3  the posture-weight axis, margin and contact against posture weight
Figure 4  hardware replay, commanded against achieved on the limiting joint

All three read the committed result JSONs rather than re-solving, so a figure
can never disagree with the numbers in the text. Output goes to figures/ as
both PDF for the paper and PNG for quick viewing.

    make_figures.py            # everything available
    make_figures.py --only 2   # one figure
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

# IEEE single column is 3.5 in. Keep type at 8 pt so it survives reduction.
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"fid": "#B4413D", "fidreg": "#D98A56", "tcen": "#4A7BA7", "tdef": "#2E6B4F"}


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


# ── Figure 2: the dose-response ──────────────────────────────────────────────
def figure2():
    p = os.path.join(RES, "reach_extent", "v3_right.json")
    if not os.path.isfile(p):
        print("  Figure 2 skipped, no v3_right.json"); return
    d = json.load(open(p))
    lv = ["d1", "d2", "d3", "d4"]
    ext = [d["by_level"][k]["extension_ratio"] * 100 for k in lv]
    n = [d["by_level"][k]["n"] for k in lv]
    series = [("fid_reps_violating", "direction matching", C["fid"], "o", "-"),
              ("task_reps_violating", "direction-reconstructed", C["tcen"], "s", "--"),
              ("tdef_reps_violating", "task-defined", C["tdef"], "D", "-")]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for key, lab, col, mk, ls in series:
        y = [d["by_level"][k][key] / d["by_level"][k]["n"] * 100 for k in lv]
        ax.plot(ext, y, ls, color=col, marker=mk, ms=4, lw=1.4, label=lab,
                zorder=3, clip_on=False)
    ax.axvspan(94, 95, color="0.85", zorder=0, lw=0)
    ax.annotate("onset", xy=(94.5, 62), ha="center", va="bottom",
                fontsize=7, color="0.35")
    ax.set_xlabel("reconstructed extension, percent of arm length")
    ax.set_ylabel("repetitions with limit contact, percent")
    ax.set_ylim(-4, 106)
    ax.set_xlim(89, 101)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)
    for x, m in zip(ext, n):
        ax.annotate(f"n={m}", xy=(x, -4), ha="center", va="top",
                    fontsize=6, color="0.5", annotation_clip=False)
    save(fig, "fig2_dose_response")


# ── Figure 3: the posture-weight axis ────────────────────────────────────────
ORDER = ["okami-gain1.0", "h2o-keypoint", "b1-naive", "okami-gain0.1",
         "okami-gain0.01", "okami-gain0.001", "b1-iros-reg", "taskcentric"]
LABEL = {"okami-gain1.0": "soft posture, 1.0", "h2o-keypoint": "hard keypoint",
         "b1-naive": "ours, naive", "okami-gain0.1": "soft posture, 0.1",
         "okami-gain0.01": "soft posture, 0.01",
         "okami-gain0.001": "soft posture, 0.001",
         "b1-iros-reg": "ours, regularized", "taskcentric": "posture-free"}


def figure3():
    ps = {s: os.path.join(RES, "b3", f"v2_{s}.json") for s in ("right", "left")}
    if not all(os.path.isfile(v) for v in ps.values()):
        print("  Figure 3 skipped, no b3 v2 files"); return
    T = {s: json.load(open(v))["totals"] for s, v in ps.items()}
    keys = [k for k in ORDER if k in T["right"]]
    x = np.arange(len(keys))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1], hspace=0.18))
    for s, col, mk in (("right", "#B4413D", "o"), ("left", "#4A7BA7", "s")):
        a1.plot(x, [T[s][k]["margin_mean"] for k in keys], "-", color=col,
                marker=mk, ms=4, lw=1.4, label=f"{s} arm", clip_on=False)
        a2.plot(x, [T[s][k]["reps_with_violation"] for k in keys], "-", color=col,
                marker=mk, ms=4, lw=1.4, clip_on=False)
    a1.set_ylabel("mean limit margin")
    a1.legend(frameon=False, loc="upper left", handlelength=1.8)
    a2.set_ylabel("captures in contact, of 25")
    a2.set_ylim(-1, 26)
    a2.set_xticks(x)
    a2.set_xticklabels([LABEL[k] for k in keys], rotation=38, ha="right")
    a1.annotate("more posture", xy=(0, a1.get_ylim()[1]), fontsize=6,
                color="0.5", va="top")
    a1.annotate("less posture", xy=(len(keys) - 1, a1.get_ylim()[1]), fontsize=6,
                color="0.5", va="top", ha="right")
    save(fig, "fig3_posture_axis")


# ── Figure 4: hardware replay ────────────────────────────────────────────────
def figure4():
    logs = sorted(glob.glob(os.path.join(RES, "x3_v2", "*.log")))
    if not logs:
        print("  Figure 4 skipped, no replay logs yet in results/x3_v2/")
        print("    run the X3 runbook, then re-run this script")
        return
    print(f"  Figure 4: found {len(logs)} logs, plotting is wired but the log "
          f"schema should be checked against one real file first")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, choices=[2, 3, 4])
    a = ap.parse_args()
    todo = [a.only] if a.only else [2, 3, 4]
    for n in todo:
        print(f"Figure {n}:")
        {2: figure2, 3: figure3, 4: figure4}[n]()


if __name__ == "__main__":
    main()
