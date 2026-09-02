#!/usr/bin/env python3
"""Did the reach sweep actually sweep anything?

Run this after the first two or three levels, while the targets can still be
moved. It answers the one question the sweep exists to answer, which is whether
the target distance is controlling how far the arm extends.

Two quantities per level, both taken in the subject's own torso frame so the
camera angle does not enter:

  fwd      forward component of wrist minus shoulder, in cm. This is what the
           target distance is supposed to control, and it must rise down the
           column.
  ext      reach divided by that frame's own segment sum. Scale-free and
           bounded by 1 through the triangle inequality, so no arm-length
           constant is involved and it cannot report an impossible value.

Everything is measured on the REACH frames only, meaning the top quartile of
forward reach within each capture. This matters more than it sounds. A hanging
arm is nearly straight, so it scores 0.90 to 0.97 on any extension measure, and
a capture that is 90 percent rest reports the resting posture rather than the
reach. The %rest column shows how much of each capture is dead time.

A sweep is working when fwd rises monotonically down the column and ext spans
at least 0.15. Keep %rest under about 60 percent, since a capture dominated by
rest gives too few reach frames to estimate anything from.

    check_sweep_span.py RS_d*_[123].csv
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common.human_directions import torso_frame          # noqa: E402


def load(path, conf_min):
    frames = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row.get("conf", 1.0) or 1.0) < conf_min:
                continue
            fi = int(float(row["frame"]))
            frames.setdefault(fi, {})[row["joint"].strip()] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])])
    return frames


def measure(path, side, conf_min):
    """Per frame forward reach and scale-free extension, paired so the caller
    can select the reach frames and drop the rest frames."""
    sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    out = []
    for j in load(path, conf_min).values():
        if sh not in j or wr not in j or el not in j:
            continue
        R = torso_frame(j)
        if R is None:
            continue
        a = float(np.linalg.norm(j[el] - j[sh]))
        b = float(np.linalg.norm(j[wr] - j[el]))
        if not (0.12 < a < 0.5 and 0.12 < b < 0.5):
            continue
        v = j[wr] - j[sh]
        out.append((float((R.T @ v)[0]), float(np.linalg.norm(v)) / (a + b)))
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--side", default="right", choices=["right", "left"])
    p.add_argument("--conf-min", type=float, default=0.5)
    args = p.parse_args()

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f])
    paths = [f for f in paths if os.path.isfile(f)]
    if not paths:
        sys.exit("no files matched")

    lv = defaultdict(list)
    for f in paths:
        m = re.search(r"(d\d+(?:_frontal)?)", os.path.basename(f))
        key = m.group(1) if m else os.path.basename(f)
        lv[key].extend(measure(f, args.side, args.conf_min))

    print(f"{args.side} arm, torso frame, reach frames only\n")
    print(f"{'level':12}{'n':>7}{'%rest':>7}{'fwd cm':>9}{'ext':>8}   step")
    keys = sorted(lv, key=lambda k: (("frontal" in k), k))
    prev = None
    exts = []
    warn = []
    for k in keys:
        rows = lv[k]
        if len(rows) < 40:
            continue
        A = np.array(rows)
        rest = float((A[:, 0] < 0.05).mean())
        sel = A[A[:, 0] >= np.percentile(A[:, 0], 75)]
        F = float(np.median(sel[:, 0])) * 100
        E = float(np.median(sel[:, 1]))
        step = "" if prev is None else f"{F - prev:+6.1f} cm"
        if prev is not None and F - prev < 2:
            step += "   NOT MONOTONIC"
            warn.append(k)
        if rest > 0.60:
            step += "   MOSTLY REST"
            warn.append(k)
        print(f"{k:12}{len(A):7d}{rest*100:6.0f}%{F:9.1f}{E:8.3f}   {step}")
        prev = F
        if "frontal" not in k:
            exts.append(E)

    print()
    if len(exts) >= 2:
        span = max(exts) - min(exts)
        print(f"extension span across levels: {span:.3f}")
        if span < 0.06:
            print("TOO FLAT. The targets are not controlling arm extension.")
            print("Check that the target sits at CHEST HEIGHT, not on the floor.")
            print("A target near the feet turns the reach downward, which "
                  "extends the arm without extending it forward.")
        elif span < 0.10:
            print("Narrow. Push the nearest target closer and the farthest "
                  "further before shooting the remaining levels.")
        else:
            print("Good span. The levels are distinguishable.")
    if warn:
        print(f"levels needing attention: {', '.join(sorted(set(warn)))}")


if __name__ == "__main__":
    main()
