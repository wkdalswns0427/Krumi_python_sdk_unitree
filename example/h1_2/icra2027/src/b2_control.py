#!/usr/bin/env python3
"""B2: does a textbook DLS + null-space limit avoidance also fix the flips?

Runs the same reader as x6_taskcentric.py but with ik_dls.solve_dls instead
of solve_taskcentric. The purpose is to check whether the flip fix belongs to
the objective (drop the posture-fidelity term) or to some novelty in the
task-centric solver. B2 was intended to show the former: this baseline is
"generic" and should also remove most flips, at the cost of approach-
direction error at contact events (~12 deg per the plan's B2 result).

Reports flips, tip error, and approach-direction error against the human
forearm unit vector (u_hat, f_hat from the reader).
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4  # noqa: E402
from common.ik_dls import solve_dls  # noqa: E402
from common.human_directions import (  # noqa: E402
    load_frames, per_frame_inputs, robot_scaled_wrist_target)
from common.metrics import (  # noqa: E402
    flip_count, limit_violation, limit_margin,
    approach_direction_error_deg)
from common.data_paths import ACTIVE_DATA_ROOT as DATA_ROOT, joints_csv_for  # noqa: E402

REST = dict(left=np.array([0.31, 0.24, -0.12, 0.67]),
            right=np.array([0.30, -0.24, 0.12, 0.69]))


def run(joints_csv, side, urdf, fps=60.0):
    fk = ArmChainFK(urdf, side)
    lims = load_limits_4(urdf)[side]
    q_prev = REST[side].copy()
    frames = load_frames(joints_csv, fps=fps)
    ts, qs, tip_errs, viols, margins, appr_errs = [], [], [], [], [], []
    t_solve = 0.0
    for t, u_hat, f_hat, _wrist_torso in per_frame_inputs(frames, side, fps):
        # Rebuild from THIS frame's directions. Hoisting this out of the
        # loop freezes the target at frame 0, which makes the task-centric
        # arm hold a pose instead of tracking the motion.
        build_target = robot_scaled_wrist_target(fk, u_hat, f_hat)
        target = build_target(q_prev)
        t0 = time.perf_counter()
        q, _res, _it = solve_dls(fk, target, q_prev, lims,
                                 lambda_limit=0.05)
        t_solve += time.perf_counter() - t0
        ts.append(t); qs.append(q)
        tip_errs.append(float(np.linalg.norm(fk.tip(q) - target)))
        viols.append(limit_violation(q, lims["lo"], lims["hi"]))
        margins.append(limit_margin(q, lims["lo"], lims["hi"]))
        appr_errs.append(approach_direction_error_deg(q, fk, f_hat))
        q_prev = q
    ts = np.asarray(ts); qs = np.asarray(qs)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0 / fps
    return dict(
        n_frames=int(len(ts)),
        flips_yaw=int(flip_count(qs, dt, joint_index=2)),
        tip_err_mean_mm=float(np.mean(tip_errs) * 1000.0),
        tip_err_p95_mm=float(np.percentile(tip_errs, 95) * 1000.0),
        limit_violation_frames=int(np.sum(viols)),
        any_violation=bool(np.any(viols)),
        margin_mean=float(np.mean(margins)),
        approach_err_mean_deg=float(np.nanmean(appr_errs)),
        approach_err_p95_deg=float(np.nanpercentile(appr_errs, 95)),
        solver_ms_per_frame=float(1000.0 * t_solve / max(1, len(ts))),
    ), ts, qs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", nargs="+",
                   default=["M2_1", "M2_2", "M2_3",
                            "M3_1", "M3_2", "M3_3"])
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "b2", "preview.json"))
    p.add_argument("--frames-dir", default=os.path.join(
        HERE, "..", "results", "b2"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.frames_dir, exist_ok=True)

    summary = dict(side=args.side, per_rep=[])
    tot_flips = 0; tot_viol_reps = 0; tot_reps = 0
    for cap in args.captures:
        joints_csv = joints_csv_for(cap)
        if not os.path.isfile(joints_csv):
            print(f"[b2] SKIP {cap}: no joints CSV at {joints_csv}",
                  file=sys.stderr); continue
        stats, ts, qs = run(joints_csv, args.side, args.urdf, args.fps)
        stats["capture"] = cap
        summary["per_rep"].append(stats)
        tot_flips += stats["flips_yaw"]; tot_reps += 1
        if stats["any_violation"]:
            tot_viol_reps += 1
        fp = os.path.join(args.frames_dir, f"{cap}_{args.side}.csv")
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "q_pitch", "q_roll", "q_yaw", "q_elbow"])
            for t, q in zip(ts, qs):
                w.writerow([f"{t:.4f}"] + [f"{v:.6f}" for v in q])
        print(f"[b2] {cap:>6}: flips={stats['flips_yaw']:>2}"
              f"  viol_frames={stats['limit_violation_frames']:>3}"
              f"  tip_mm={stats['tip_err_mean_mm']:>4.1f}"
              f"  approach_deg={stats['approach_err_mean_deg']:>5.1f}"
              f"  margin={stats['margin_mean']:.2f}")
    summary["totals"] = dict(total_flips=tot_flips,
                             reps_with_violation=tot_viol_reps,
                             reps_total=tot_reps)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[b2] wrote {args.out}")
    print(f"[b2] TOTAL: {tot_flips} flips across {tot_reps} reps,"
          f" {tot_viol_reps}/{tot_reps} reps had a violation")


if __name__ == "__main__":
    main()
