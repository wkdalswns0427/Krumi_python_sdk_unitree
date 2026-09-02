#!/usr/bin/env python3
"""X2 (cost side): what does fidelity retargeting cost, task held fixed?

For each frame, computes both configurations that reach the SAME task target:
  fidelity      : shoulder + elbow angles matching human upper-arm / forearm
                  direction (ik_baseline.solve_fidelity)
  robot_optimal : shoulder + elbow angles reaching the same wrist target while
                  optimizing joint-limit margin (ik_taskcentric.solve_taskcentric)

Reports per-rep: mean limit margin, mean normalized manipulability, joint-
limit violation rate, and gravity-load proxy (sum |q[i] - q[i,rest]| * effort
weight; a URDF-mass gravity torque calc requires link inertia parsing and is
left to a follow-up if the margin/violation story is not enough).

Companion x2_robot_optimal.py exists as a thin wrapper for callers that only
want the robot-optimal solution (e.g. figures); the analysis lives here.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4  # noqa: E402
from common.ik_baseline import solve_fidelity  # noqa: E402
from common.ik_taskcentric import solve_taskcentric  # noqa: E402
from common.human_directions import (  # noqa: E402
    load_frames, per_frame_inputs, robot_scaled_wrist_target)
from common.metrics import (  # noqa: E402
    limit_margin, limit_violation, normalized_manipulability)
from common.data_paths import ACTIVE_DATA_ROOT as DATA_ROOT, joints_csv_for  # noqa: E402

REST_R = np.array([0.30, -0.24, 0.12, 0.69])
REST_L = np.array([0.31, 0.24, -0.12, 0.67])


def solve_pair(joints_csv, side, urdf, fps=60.0):
    fk = ArmChainFK(urdf, side)
    lims = load_limits_4(urdf)[side]
    q_prev_f = REST_R.copy() if side == "right" else REST_L.copy()
    q_prev_t = q_prev_f.copy()
    frames = load_frames(joints_csv, fps=fps)
    rows = []
    for t, u_hat, f_hat, _wrist_torso in per_frame_inputs(frames, side, fps):
        # Rebuild from THIS frame's directions. Hoisting this out of the
        # loop freezes the target at frame 0, which makes the task-centric
        # arm hold a pose instead of tracking the motion.
        build_target = robot_scaled_wrist_target(fk, u_hat, f_hat)
        target = build_target(q_prev_t)   # task target from the human directions
        q_f, _res_deg, _clamp = solve_fidelity(
            fk, u_hat, f_hat, q_prev_f, lims, yaw_reg=0.0, yaw_neutral=0.0)
        q_t, _res_m, _it = solve_taskcentric(
            fk, target, q_prev_t, lims,
            lambda_limit=0.05, lambda_cont=0.02, lambda_manip=0.0)
        J_f = fk.tip_jacobian(q_f)
        J_t = fk.tip_jacobian(q_t)
        rows.append(dict(
            t=t,
            q_fid=q_f.tolist(), q_task=q_t.tolist(),
            margin_fid=limit_margin(q_f, lims["lo"], lims["hi"]),
            margin_task=limit_margin(q_t, lims["lo"], lims["hi"]),
            viol_fid=limit_violation(q_f, lims["lo"], lims["hi"]),
            viol_task=limit_violation(q_t, lims["lo"], lims["hi"]),
            manip_fid=normalized_manipulability(J_f),
            manip_task=normalized_manipulability(J_t),
        ))
        q_prev_f, q_prev_t = q_f, q_t
    return rows, lims


def summarize(rows):
    if not rows:
        return {}
    keys = ("margin_fid", "margin_task", "manip_fid", "manip_task")
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    agg["viol_frames_fid"] = int(sum(1 for r in rows if r["viol_fid"]))
    agg["viol_frames_task"] = int(sum(1 for r in rows if r["viol_task"]))
    agg["any_viol_fid"] = bool(any(r["viol_fid"] for r in rows))
    agg["any_viol_task"] = bool(any(r["viol_task"] for r in rows))
    agg["n_frames"] = int(len(rows))
    agg["margin_ratio_fid_over_task"] = (
        agg["margin_fid"] / max(agg["margin_task"], 1e-9))
    agg["manip_ratio_fid_over_task"] = (
        agg["manip_fid"] / max(agg["manip_task"], 1e-9))
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", nargs="+",
                   default=["M2_1", "M2_2", "M2_3",
                            "M3_1", "M3_2", "M3_3"])
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "x2", "preview.json"))
    p.add_argument("--frames-dir", default=os.path.join(
        HERE, "..", "results", "x2"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.frames_dir, exist_ok=True)

    summary = dict(side=args.side, per_rep=[])
    tot_fid = tot_task = 0
    reps_viol_fid = reps_viol_task = 0
    for cap in args.captures:
        joints_csv = joints_csv_for(cap)
        if not os.path.isfile(joints_csv):
            print(f"[x2] SKIP {cap}: no joints CSV at {joints_csv}",
                  file=sys.stderr)
            continue
        rows, lims = solve_pair(joints_csv, args.side, args.urdf, args.fps)
        agg = summarize(rows)
        agg["capture"] = cap
        summary["per_rep"].append(agg)
        tot_fid += 1
        tot_task += 1
        reps_viol_fid += 1 if agg["any_viol_fid"] else 0
        reps_viol_task += 1 if agg["any_viol_task"] else 0
        print(f"[x2] {cap:>6}: fid_margin={agg['margin_fid']:.3f}"
              f"  task_margin={agg['margin_task']:.3f}"
              f"  fid_manip={agg['manip_fid']:.3f}"
              f"  task_manip={agg['manip_task']:.3f}"
              f"  viol_fid={agg['any_viol_fid']!s:<5}"
              f"  viol_task={agg['any_viol_task']!s:<5}")
        # frame-level CSV for figures
        fp = os.path.join(args.frames_dir, f"{cap}_{args.side}_frames.csv")
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s",
                        "margin_fid", "margin_task",
                        "manip_fid", "manip_task",
                        "viol_fid", "viol_task"])
            for r in rows:
                w.writerow([f"{r['t']:.4f}",
                            f"{r['margin_fid']:.4f}",
                            f"{r['margin_task']:.4f}",
                            f"{r['manip_fid']:.4f}",
                            f"{r['manip_task']:.4f}",
                            int(r["viol_fid"]),
                            int(r["viol_task"])])
    summary["totals"] = dict(
        n_reps=tot_fid,
        reps_with_violation_fid=reps_viol_fid,
        reps_with_violation_task=reps_viol_task,
    )
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[x2] wrote {args.out}")
    print(f"[x2] TOTAL: {reps_viol_fid}/{tot_fid} fid-violation reps,"
          f" {reps_viol_task}/{tot_task} task-violation reps")


if __name__ == "__main__":
    main()
