#!/usr/bin/env python3
"""Is the skeleton reporting an arm longer than the arm actually is?

Run this between capture blocks, before normalization, while the camera can
still be moved. Normalization clamps the reach to the true arm length, which
hides this fault rather than fixing it, so a saturated capture looks fine
downstream while carrying no usable information about extension.

The failure mode is geometric. When the demonstrator reaches along the camera's
optical axis the arm is foreshortened in the image and the wrist keypoint is
displaced in depth, so the reconstructed shoulder-to-wrist distance exceeds
what the arm can produce. It is not a confidence problem, since the affected
keypoints are reported with high confidence, and it is not an outlier problem,
since the displacement is sustained through the reach.

Reported per file:
  p95, p99   percentiles of |wrist - shoulder| / arm_length
  %full      frames at or past full extension, which a real arm can reach
  %imp       frames past imp-margin of arm length, which it cannot
  azim       body angle to the camera, 0 deg square on, 90 deg profile
  depth      share of the reach vector lying along the camera depth axis

%full and %imp are different questions. A demonstrator genuinely reaching at
full stretch produces a high %full and that is real. Only %imp and a p95 above
the margin indicate the sensor is inventing length.

Tolerance differs by capture type. In the reach sweep extension is the measured
variable, so any impossibility is fatal to that level. In a motion capture
extension is an input the normalization step will clamp, so a few percent is
survivable and merely worth knowing.

    check_saturation.py RS_d4_*.csv --arm 0.52     # shoulder to WRIST
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np


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


def depth_axis(frames):
    """The camera stands off along one axis, so the subject's mean position on
    that axis dominates. That axis is depth."""
    P = [j["right_hip"] for j in frames.values() if "right_hip" in j]
    if not P:
        return 0
    return int(np.argmax(np.abs(np.mean(P, axis=0))))


def stats(path, side, arm, conf_min, IMP=1.05):
    fr = load(path, conf_min)
    if len(fr) < 20:
        return None
    ax = depth_axis(fr)
    ext, share, azim = [], [], []
    sh, wr = f"{side}_shoulder", f"{side}_wrist"
    for j in fr.values():
        if sh in j and wr in j:
            v = j[wr] - j[sh]
            n = float(np.linalg.norm(v))
            if n > 0.15:                       # ignore the arm hanging at rest
                ext.append(n / arm)
                share.append(abs(v[ax]) / n)
        if "left_shoulder" in j and "right_shoulder" in j:
            s = j["left_shoulder"] - j["right_shoulder"]
            lat = [k for k in (0, 1, 2) if k != ax and k != 2]
            if lat:
                d = np.hypot(s[ax], s[lat[0]])
                if d > 1e-6:
                    azim.append(abs(np.degrees(np.arctan2(abs(s[ax]),
                                                          abs(s[lat[0]])))))
    if len(ext) < 20:
        return None
    ext = np.array(ext)
    return dict(n=len(ext), p95=float(np.percentile(ext, 95)),
                p99=float(np.percentile(ext, 99)),
                sat=float((ext >= 0.995).mean()),
                imp=float((ext > IMP).mean()),
                share=float(np.mean(share)),
                azim=float(np.mean(azim)) if azim else float("nan"))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="joint CSVs, before normalization")
    p.add_argument("--arm", type=float, required=True,
                   help="this subject's SHOULDER-TO-WRIST length in metres, "
                        "not shoulder to the back of the hand")
    p.add_argument("--side", default="right", choices=["right", "left"])
    p.add_argument("--conf-min", type=float, default=0.2)
    p.add_argument("--imp-margin", type=float, default=1.05,
                   help="extension above which the reading is physically "
                        "impossible rather than merely full extension")
    p.add_argument("--warn-imp", type=float, default=8.0,
                   help="percent of impossible frames above which to reshoot")
    args = p.parse_args()

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f])
    paths = [f for f in paths if os.path.isfile(f)]
    if not paths:
        sys.exit("no files matched")

    print(f"arm length {args.arm*100:.1f} cm, {args.side} arm, "
          f"conf >= {args.conf_min}\n")
    print("%full = frames at or past full extension, which a real arm can do.")
    print(f"%imp  = frames past {args.imp_margin:.2f} of arm length, which it "
          f"cannot. That column is the one that matters.\n")
    print(f"{'file':26}{'n':>7}{'p95':>8}{'p99':>8}{'%full':>8}{'%imp':>7}"
          f"{'azim':>7}{'depth':>7}   verdict")
    bad = 0
    for f in paths:
        s = stats(f, args.side, args.arm, args.conf_min, args.imp_margin)
        if s is None:
            print(f"{os.path.basename(f):26}{'too few usable frames':>48}")
            continue
        imp = s["imp"] * 100
        if imp > args.warn_imp or s["p95"] > args.imp_margin:
            verdict, bad = "RESHOOT, camera too near the reach axis", bad + 1
        elif imp > 2.0:
            verdict = "marginal, survivable in a motion capture"
        else:
            verdict = "ok"
        print(f"{os.path.basename(f):26}{s['n']:7d}{s['p95']:8.3f}{s['p99']:8.3f}"
              f"{s['sat']*100:7.1f}%{imp:6.1f}%{s['azim']:6.0f}d"
              f"{s['share']*100:6.0f}%   {verdict}")

    print()
    if bad:
        print(f"{bad} of {len(paths)} captures report an arm longer than the "
              f"arm is. Their extension values are the clamp, not a measurement.")
        print("Move the camera further off the reach axis, toward 45 to 50 "
              "degrees on the reaching-arm side, and reshoot that block.")
    else:
        print("Nothing physically impossible. Extension values in this block "
              "are measurements.")


if __name__ == "__main__":
    main()
