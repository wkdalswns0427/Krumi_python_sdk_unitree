#!/usr/bin/env python3
"""X2 (robot-optimal side): produce ONLY the robot-optimal solutions.

Same reader as x2_costoffidelity.py but writes a plain trajectory CSV that
the IROS replayer can consume unchanged. Useful for figures that need the
robot-optimal per-frame joint angles alongside the human capture, and for
setting up X3 hardware runs on the generated (not hand-designed) motion.

Output columns match experiments/scripts/retarget_arm.py::OUT_JOINTS.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4, SIDES  # noqa: E402
from common.ik_taskcentric import solve_taskcentric  # noqa: E402
from common.human_directions import (  # noqa: E402
    load_frames, per_frame_inputs, robot_scaled_wrist_target)
from common.data_paths import ACTIVE_DATA_ROOT as DATA_ROOT, joints_csv_for  # noqa: E402

REST = dict(left=np.array([0.31, 0.24, -0.12, 0.67]),
            right=np.array([0.30, -0.24, 0.12, 0.69]))

OUT_JOINTS = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    "waist_yaw",
]


def solve_arm_series(joints_csv, side, urdf, fps=60.0):
    fk = ArmChainFK(urdf, side)
    lims = load_limits_4(urdf)[side]
    q_prev = REST[side].copy()
    frames = load_frames(joints_csv, fps=fps)
    ts, qs = [], []
    for t, u_hat, f_hat, _wrist_torso in per_frame_inputs(frames, side, fps):
        # Rebuild from THIS frame's directions. Hoisting this out of the
        # loop freezes the target at frame 0, which makes the task-centric
        # arm hold a pose instead of tracking the motion.
        build_target = robot_scaled_wrist_target(fk, u_hat, f_hat)
        target = build_target(q_prev)
        q, _res, _it = solve_taskcentric(
            fk, target, q_prev, lims,
            lambda_limit=0.05, lambda_cont=0.02, lambda_manip=0.0)
        ts.append(t); qs.append(q); q_prev = q
    return np.asarray(ts), np.asarray(qs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", help="capture folder name under MotionsDataset0713/")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--out", help="default: results/x2/traj_<capture>.csv")
    args = p.parse_args()

    joints_csv = joints_csv_for(args.capture)
    if not os.path.isfile(joints_csv):
        sys.exit(f"not found: {joints_csv}")
    per_side = {s: solve_arm_series(joints_csv, s, args.urdf, args.fps)
                for s in SIDES}
    # merge on a common time grid (per-side inputs may differ in usable frames)
    t_start = max(per_side[s][0][0] for s in SIDES)
    t_end = min(per_side[s][0][-1] for s in SIDES)
    t_out = np.arange(t_start, t_end, 1.0 / args.rate)
    cols = {}
    for s in SIDES:
        ts, qs = per_side[s]
        for k in range(4):
            cols[(s, k)] = np.interp(t_out, ts, qs[:, k])
    out = args.out or os.path.join(
        HERE, "..", "results", "x2", f"traj_{args.capture}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + OUT_JOINTS)
        for i, t in enumerate(t_out):
            full = np.zeros(15)
            for k in range(4):
                full[k] = cols[("left", k)][i]
                full[7 + k] = cols[("right", k)][i]
            w.writerow([f"{t - t_out[0]:.4f}"] + [f"{v:.6f}" for v in full])
    print(f"[x2] wrote {out}  frames={len(t_out)}  duration={t_out[-1]-t_out[0]:.1f}s")


if __name__ == "__main__":
    main()
