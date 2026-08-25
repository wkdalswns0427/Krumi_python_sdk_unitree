#!/usr/bin/env python3
"""X6: does the OBJECTIVE flip fix shoulder-yaw flips, without learning?

Compares two IKs on the same human capture, same target reach, same solver
class (damped GN / DLS, no multi-start):

  fidelity   : ik_baseline.solve_fidelity  (direction-matching, no yaw regs)
  taskcentric: ik_taskcentric.solve_taskcentric  (tip-position + robot regs)

Reports per-rep flip count on the shoulder-yaw channel (single-frame |dq/dt|
> --flip-thresh, default 6 rad/s), tip tracking error, and any joint-limit
violations. Right arm is the shoulder-yaw flip case documented for M2/M3.

Usage:
    /usr/bin/python3 src/x6_taskcentric.py --captures M2_1 M2_2 M2_3 M3_1 M3_2 M3_3 \
        --side right --out results/x6/preview.json

Outputs: one JSON summary + a per-frame CSV per (rep, method) for figures.
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
from common.ik_baseline import solve_fidelity  # noqa: E402
from common.ik_taskcentric import solve_taskcentric  # noqa: E402
from common.human_directions import (  # noqa: E402
    load_frames, per_frame_inputs, robot_scaled_wrist_target)
from common.metrics import flip_count, limit_violation, limit_margin  # noqa: E402
from common.data_paths import ACTIVE_DATA_ROOT as DATA_ROOT, joints_csv_for  # noqa: E402

METHODS = ("fidelity", "taskcentric")


def solve_rep(joints_csv, side, urdf, fps=60.0, method="fidelity"):
    fk = ArmChainFK(urdf, side)
    lims = load_limits_4(urdf)[side]
    q_prev = np.array([0.30 if side == "right" else 0.31,
                       -0.24 if side == "right" else 0.24,
                       0.12 if side == "right" else -0.12,
                       0.69 if side == "right" else 0.67])
    frames = load_frames(joints_csv, fps=fps)
    ts, qs, tips_err, viols, margins, targets = [], [], [], [], [], []
    t_solve = 0.0
    build_target = None
    for t, u_hat, f_hat, _wrist_torso in per_frame_inputs(frames, side, fps):
        if build_target is None:
            build_target = robot_scaled_wrist_target(fk, u_hat, f_hat)
        target = build_target(q_prev)
        t0 = time.perf_counter()
        if method == "fidelity":
            q, _res_deg, _clamp = solve_fidelity(
                fk, u_hat, f_hat, q_prev, lims,
                yaw_reg=0.0, yaw_neutral=0.0)
        elif method == "taskcentric":
            q, _res_m, _it = solve_taskcentric(
                fk, target, q_prev, lims,
                lambda_limit=0.05, lambda_cont=0.02, lambda_manip=0.0)
        else:
            raise ValueError(method)
        t_solve += time.perf_counter() - t0
        ts.append(t)
        qs.append(q)
        tip_err = float(np.linalg.norm(fk.tip(q) - target))
        tips_err.append(tip_err)
        viols.append(limit_violation(q, lims["lo"], lims["hi"]))
        margins.append(limit_margin(q, lims["lo"], lims["hi"]))
        targets.append(target)
        q_prev = q
    ts = np.asarray(ts); qs = np.asarray(qs)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0 / fps
    flips = flip_count(qs, dt, threshold_rad_per_s=6.0, joint_index=2)
    return dict(
        n_frames=int(len(ts)),
        dt_median=dt,
        flips_yaw=int(flips),
        tip_err_mean_mm=float(np.mean(tips_err) * 1000.0),
        tip_err_p95_mm=float(np.percentile(tips_err, 95) * 1000.0),
        limit_violation_frames=int(np.sum(viols)),
        any_violation=bool(np.any(viols)),
        margin_mean=float(np.mean(margins)),
        solver_ms_per_frame=float(1000.0 * t_solve / max(1, len(ts))),
    ), ts, qs, np.asarray(targets)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", nargs="+",
                   default=["M2_1", "M2_2", "M2_3",
                            "M3_1", "M3_2", "M3_3"],
                   help="capture folder names under MotionsDataset0713/")
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "x6", "preview.json"))
    p.add_argument("--frames-dir", default=os.path.join(
        HERE, "..", "results", "x6"),
        help="write per-rep joint trajectories here (one CSV per rep+method)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.frames_dir, exist_ok=True)

    summary = dict(side=args.side, urdf=args.urdf, per_rep=[])
    tot = {m: dict(flips=0, viol_frames=0, viols=0, n=0) for m in METHODS}
    for cap in args.captures:
        joints_csv = joints_csv_for(cap)
        if not os.path.isfile(joints_csv):
            print(f"[x6] SKIP {cap}: no joints CSV at {joints_csv}",
                  file=sys.stderr)
            continue
        rep_row = dict(capture=cap)
        for m in METHODS:
            stats, ts, qs, targets = solve_rep(joints_csv, args.side,
                                                args.urdf, args.fps, m)
            rep_row[m] = stats
            tot[m]["flips"] += stats["flips_yaw"]
            tot[m]["viol_frames"] += stats["limit_violation_frames"]
            tot[m]["viols"] += 1 if stats["any_violation"] else 0
            tot[m]["n"] += 1
            fp = os.path.join(args.frames_dir, f"{cap}_{args.side}_{m}.csv")
            with open(fp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "q_pitch", "q_roll", "q_yaw", "q_elbow",
                            "target_x", "target_y", "target_z"])
                for t, q, tgt in zip(ts, qs, targets):
                    w.writerow([f"{t:.4f}"] + [f"{v:.6f}" for v in q]
                               + [f"{v:.6f}" for v in tgt])
        summary["per_rep"].append(rep_row)
        line = f"[x6] {cap:>6} side={args.side}"
        for m in METHODS:
            s = rep_row[m]
            line += (f"  {m}: flips={s['flips_yaw']:>2}"
                     f" viol_frames={s['limit_violation_frames']:>3}"
                     f" tip_mm={s['tip_err_mean_mm']:>5.1f}"
                     f" margin={s['margin_mean']:.2f}"
                     f" ms/f={s['solver_ms_per_frame']:.1f}")
        print(line)

    summary["totals"] = {m: dict(
        total_flips=tot[m]["flips"],
        total_violation_frames=tot[m]["viol_frames"],
        reps_with_violation=tot[m]["viols"],
        reps_total=tot[m]["n"]) for m in METHODS}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[x6] wrote {args.out}")
    for m in METHODS:
        t = tot[m]
        print(f"[x6] TOTAL {m}: {t['flips']} flips across {t['n']} reps,"
              f" {t['viols']}/{t['n']} reps had a limit violation")


if __name__ == "__main__":
    main()
