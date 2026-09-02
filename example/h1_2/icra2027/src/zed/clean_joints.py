#!/usr/bin/env python3
"""clean_joints.py - remove non-physical single-frame outliers from a joints CSV.

BODY_38 sometimes places a keypoint far off for one frame. A jump of a metre or
more in one frame is impossible for a wrist. This drops those samples so the
retargeting never sees the spike. Each sample is compared to a short per-axis
median of its neighbours, and it is dropped when it deviates by more than
--max-jump-mm. Only the offending joint sample is removed, not the whole frame,
and a dropped sample is treated downstream like any missing frame.

    clean_joints.py M1_1.csv --max-jump-mm 400            # writes M1_1_clean.csv
    clean_joints.py M1_1.csv --max-jump-mm 400 --inplace  # overwrites
"""

import argparse
import csv
import os

import numpy as np


def clean(rows, max_jump_mm, win=2):
    """Return kept rows and a per-joint removed-count. A sample is removed when
    it is more than max_jump_mm from the per-axis median of its +/- win present
    neighbours."""
    thr = max_jump_mm / 1000.0
    by_joint = {}
    for r in rows:
        by_joint.setdefault(r["joint"], []).append(r)
    drop = {}
    removed = {}
    for j, rr in by_joint.items():
        rr.sort(key=lambda x: int(x["frame"]))
        f = np.array([int(x["frame"]) for x in rr])
        p = np.array([[float(x["x"]), float(x["y"]), float(x["z"])] for x in rr])
        bad = set()
        for i in range(len(rr)):
            idx = [k for k in range(max(0, i - win), min(len(rr), i + win + 1)) if k != i]
            if len(idx) < 2:
                continue
            med = np.median(p[idx], axis=0)
            if np.linalg.norm(p[i] - med) > thr:
                bad.add(int(f[i]))
        if bad:
            drop[j] = bad
            removed[j] = len(bad)
    kept = [r for r in rows if int(r["frame"]) not in drop.get(r["joint"], ())]
    return kept, removed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--max-jump-mm", type=float, default=400.0,
                    help="drop a sample this far from its local median (mm)")
    ap.add_argument("--win", type=int, default=2, help="neighbours each side for the median")
    ap.add_argument("--out", help="output CSV (default <stem>_clean.csv)")
    ap.add_argument("--inplace", action="store_true", help="overwrite the input CSV")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit("empty CSV")
    kept, removed = clean(rows, args.max_jump_mm, args.win)

    out = args.csv if args.inplace else (
        args.out or os.path.splitext(args.csv)[0] + "_clean.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    total = sum(removed.values())
    print(f"[clean] {os.path.basename(args.csv)}: removed {total} samples "
          f"from {len(removed)} joints -> {os.path.basename(out)}")
    if removed:
        worst = sorted(removed.items(), key=lambda kv: -kv[1])[:4]
        print("  most-removed:", ", ".join(f"{j} {n}" for j, n in worst))


if __name__ == "__main__":
    main()
