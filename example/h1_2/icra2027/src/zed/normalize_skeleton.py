#!/usr/bin/env python3
"""normalize_skeleton.py - enforce anatomically constant arm bone lengths.

ZED BODY_38 degrades when the subject reaches toward the camera. The arm points
along the camera axis, is foreshortened in the image, and the wrist gets pushed
out in depth. Measured on the 2026-08-29 session: at the far reach distances the
shoulder-to-wrist distance reaches 72 to 79 cm while the upper arm plus forearm
only sum to 52 to 55 cm. A rigid arm cannot do that, so those frames are wrong.

Confidence does not catch this. The keypoints are confident and still wrong, and
the error is sustained rather than a single-frame teleport, so the outlier filter
does not catch it either.

The fix keeps what the retargeter actually uses and discards what it should never
have trusted. Direction is the action, magnitude is the artifact:

  1. Keep the shoulder as measured, it is the most stable joint.
  2. Keep the shoulder-to-wrist DIRECTION, which is where the arm is reaching.
  3. Clamp the shoulder-to-wrist DISTANCE to the true arm length.
  4. Re-place the elbow by two-link inverse kinematics, on the side the measured
     elbow was on, so the swivel of the arm is preserved.

The result has exactly correct bone lengths every frame, preserves the reach
direction and the elbow swivel, and is physically realizable.

    normalize_skeleton.py M2_1.csv --arm-length-m 0.6096
    normalize_skeleton.py M2_1.csv --arm-length-m 0.6096 --inplace
"""

import argparse
import csv
import os

import numpy as np

SIDES = ("left", "right")


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def segment_lengths(frames, side):
    """Median measured (upper arm, forearm) for a side, or None."""
    ua, fa = [], []
    for j in frames.values():
        s, e, w = (j.get(f"{side}_shoulder"), j.get(f"{side}_elbow"),
                   j.get(f"{side}_wrist"))
        if s is None or e is None or w is None:
            continue
        ua.append(np.linalg.norm(e - s))
        fa.append(np.linalg.norm(w - e))
    if len(ua) < 5:
        return None
    return float(np.median(ua)), float(np.median(fa))


def normalize_arm(j, side, l_up, l_fo):
    """Rewrite this frame's elbow and wrist so the bones are exactly l_up and
    l_fo. Returns True if the frame was rewritten."""
    s = j.get(f"{side}_shoulder")
    e = j.get(f"{side}_elbow")
    w = j.get(f"{side}_wrist")
    if s is None or e is None or w is None:
        return False

    total = l_up + l_fo
    d = w - s
    L = float(np.linalg.norm(d))
    d_hat = _unit(d)
    if d_hat is None:
        return False

    # Reachable band for a two-link arm. Clamp just inside it so the elbow stays
    # defined and the arm never sits exactly at the singular full extension.
    L = min(L, total * 0.999)
    L = max(L, abs(l_up - l_fo) + 1e-4)

    # Two-link IK. The elbow lies on a circle around the shoulder-wrist axis.
    a = (L * L + l_up * l_up - l_fo * l_fo) / (2.0 * L)
    h2 = l_up * l_up - a * a
    h = float(np.sqrt(h2)) if h2 > 0 else 0.0
    center = s + a * d_hat

    # Keep the measured elbow's side: the component perpendicular to the axis.
    perp = (e - s) - np.dot(e - s, d_hat) * d_hat
    n_hat = _unit(perp)
    if n_hat is None:
        # Degenerate (elbow on the axis). Any perpendicular will do.
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, d_hat)) > 0.9:
            tmp = np.array([0.0, 0.0, 1.0])
        n_hat = _unit(tmp - np.dot(tmp, d_hat) * d_hat)

    j[f"{side}_elbow"] = center + h * n_hat
    j[f"{side}_wrist"] = s + L * d_hat
    return True


def load(path):
    frames, order = {}, []
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        for r in rd:
            fi = int(r["frame"])
            if fi not in frames:
                frames[fi] = {}
                order.append(fi)
            frames[fi][r["joint"]] = np.array(
                [float(r["x"]), float(r["y"]), float(r["z"])])
            frames[fi].setdefault("_conf", {})[r["joint"]] = r["conf"]
    return frames, order, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--arm-length-m", type=float, default=None,
                    help="true SHOULDER-TO-WRIST length in metres. Not shoulder "
                         "to the back of the hand, which runs about 9 cm longer "
                         "and skews every extension ratio. Required.")
    ap.add_argument("--upper-frac", type=float, default=None,
                    help="upper-arm share of the total. Default: the capture's "
                         "own median ratio, which respects ZED keypoint placement")
    ap.add_argument("--out")
    ap.add_argument("--inplace", action="store_true")
    args = ap.parse_args()
    if args.arm_length_m is None:
        raise SystemExit(
            "--arm-length-m is required.\n"
            "It must be shoulder joint centre to the WRIST CREASE, not to the "
            "back of the hand.\nThe two differ by roughly a hand, the tracker "
            "places its wrist keypoint at\nthe wrist, and the hand measurement "
            "therefore inflates the denominator of\nevery extension ratio. S1 "
            "was recorded as 0.6096 m to the back of the hand\nand has not yet "
            "been re-measured at the wrist.")

    frames, order, cols = load(args.csv)
    report = []
    for side in SIDES:
        meas = segment_lengths(frames, side)
        if meas is None:
            continue
        mu, mf = meas
        frac = args.upper_frac if args.upper_frac else mu / (mu + mf)
        l_up = args.arm_length_m * frac
        l_fo = args.arm_length_m * (1.0 - frac)

        before = [np.linalg.norm(j[f"{side}_wrist"] - j[f"{side}_shoulder"])
                  for j in frames.values()
                  if f"{side}_wrist" in j and f"{side}_shoulder" in j]
        n = sum(normalize_arm(j, side, l_up, l_fo) for j in frames.values())
        after = [np.linalg.norm(j[f"{side}_wrist"] - j[f"{side}_shoulder"])
                 for j in frames.values()
                 if f"{side}_wrist" in j and f"{side}_shoulder" in j]
        report.append(
            f"  {side:5} bones {mu*100:.1f}+{mf*100:.1f}={mu*100+mf*100:.1f} cm "
            f"-> {l_up*100:.1f}+{l_fo*100:.1f}={args.arm_length_m*100:.1f} cm | "
            f"reach p95 {np.percentile(before,95)*100:.1f} -> "
            f"{np.percentile(after,95)*100:.1f} cm | {n} frames")

    out = args.csv if args.inplace else (
        args.out or os.path.splitext(args.csv)[0] + "_norm.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fi in order:
            j = frames[fi]
            confs = j.get("_conf", {})
            for name, xyz in j.items():
                if name == "_conf":
                    continue
                w.writerow([fi, name, f"{xyz[0]:.6f}", f"{xyz[1]:.6f}",
                            f"{xyz[2]:.6f}", confs.get(name, "1.0")])
    print(f"[norm] {os.path.basename(args.csv)} -> {os.path.basename(out)}")
    for line in report:
        print(line)


if __name__ == "__main__":
    main()
