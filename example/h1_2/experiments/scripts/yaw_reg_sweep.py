#!/usr/bin/env python3
"""yaw_reg_sweep.py - pick the shoulder-yaw continuity weight for retarget_arm.

M2 (bricklaying) and M3 (lifting) are not replay-ready: the warm-started
Gauss-Newton IK hops shoulder-yaw solution branches (single-frame flips at
9-16 rad/s, occasionally parking at the yaw limit). retarget_arm.py --yaw-reg
adds a quadratic continuity penalty on shoulder yaw only. This sweeps that
weight and reports, per capture, the tradeoff:

  branch flips + peak yaw velocity + %-at-limit   (want these to vanish)
  vs
  EE tracking error of the retargeted trajectory  (must NOT degrade)

EE tracking error is retarget_arm's own metric: achieved FK wrist vs the
wrist implied by the human limb directions and the robot's link lengths,
printed per arm. We parse it from the tool's stdout so there is one source
of truth. The winner is NOT hardcoded: this prints the sweep so a human
freezes the smallest weight that kills the flips without moving EE error.

Usage:
    /usr/bin/python3 yaw_reg_sweep.py \
        --caps M1_R_1 M1_R_2 M1_R_3 M2_1 M2_2 M2_3 M3_1 M3_2 M3_3 \
        --weights 0 0.02 0.05 0.1 0.2 0.5 \
        --out b2g_sweep/yaw_sweep.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)          # scripts/ -> experiments/ (data root)
FLIP_RAD_S = 6.0          # single-frame velocity above this = a branch flip
YAW_LIMIT_DEG = (-172.0, 152.0)


def retarget(cap, weight, out_csv):
    """Run retarget_arm at a given yaw-reg; return per-arm EE-track (cm)."""
    inp = os.path.join(EXP_DIR, "iphone_data", cap, "b2g", "iph_mono.csv")
    cmd = [sys.executable, os.path.join(HERE, "retarget_arm.py"), inp,
           "--yaw-reg", str(weight), "--out", out_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"retarget failed for {cap} w={weight}:\n{r.stderr}")
    ee = {}
    for side in ("left", "right"):
        m = re.search(rf"{side}:.*EE-track mean=([\d.]+) cm", r.stdout)
        ee[side] = float(m.group(1)) if m else float("nan")
    return ee


def traj_yaw_health(path):
    """Per arm: (#flips>6rad/s, peak |yaw vel| rad/s, %-frames at limit)."""
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t_s"]) for r in rows])
    out = {}
    for side in ("left", "right"):
        y = np.array([float(r[f"{side}_shoulder_yaw"]) for r in rows])
        v = np.abs(np.diff(y) / np.diff(t))
        deg = np.degrees(y)
        at_lim = np.mean((deg < YAW_LIMIT_DEG[0] + 12) |
                         (deg > YAW_LIMIT_DEG[1] - 12)) * 100
        out[side] = (int(np.sum(v > FLIP_RAD_S)), float(v.max()), float(at_lim))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caps", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="yawsweep_")
    rows = []
    print(f"{'cap':7} {'w':>5} {'side':5} {'flips':>6} {'peakVel':>8} "
          f"{'%@lim':>6} {'EEtrack cm':>10}")
    for cap in args.caps:
        for w in args.weights:
            out_csv = os.path.join(tmp, f"{cap}_w{w}.csv")
            ee = retarget(cap, w, out_csv)
            health = traj_yaw_health(out_csv)
            for side in ("right", "left"):
                fl, pk, lim = health[side]
                rows.append(dict(cap=cap, weight=w, side=side, flips=fl,
                                 peak_yaw_vel=round(pk, 2),
                                 pct_at_limit=round(lim, 1),
                                 ee_track_cm=round(ee[side], 3)))
                mark = " <-flips" if (side == "right" and fl > 0) else ""
                print(f"{cap:7} {w:>5} {side:5} {fl:>6} {pk:>8.2f} "
                      f"{lim:>6.1f} {ee[side]:>10.3f}{mark}")
        print("-" * 52)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")

    # concise recommendation aid (right arm is where the flips live)
    print("\nSmallest flip-free weight per capture (right arm), and its EE cost:")
    for cap in args.caps:
        cr = [r for r in rows if r["cap"] == cap and r["side"] == "right"]
        base = next(r for r in cr if r["weight"] == 0.0)
        flipfree = [r for r in cr if r["flips"] == 0]
        if not flipfree:
            print(f"  {cap:7}: NO weight eliminated flips (max tried "
                  f"{max(args.weights)})")
            continue
        best = min(flipfree, key=lambda r: r["weight"])
        d_ee = best["ee_track_cm"] - base["ee_track_cm"]
        print(f"  {cap:7}: w={best['weight']:<5} flips {base['flips']}->0  "
              f"EE {base['ee_track_cm']:.2f}->{best['ee_track_cm']:.2f} cm "
              f"(delta {d_ee:+.2f})")


if __name__ == "__main__":
    main()
