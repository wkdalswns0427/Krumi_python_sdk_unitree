#!/usr/bin/env python3
"""B3: published retargeting methods as baselines, on one axis with ours.

Answers the strawman objection the plan anticipates. Rather than asserting
that B1 direction-matching represents "the field's default", this runs the
retargeting objectives of two published humanoid methods and measures what
each costs in robot feasibility on the identical task.

Conditions, ordered by how strongly they constrain arm posture:

  h2o        H2O-style keypoint position matching on (elbow, wrist), equal
             weights. Posture fully pinned: 6 residuals on 4 DOF.
  okami-*    OKAMI-style weighted IK, wrist position dominant with posture as
             a weak regulariser. Swept over posture weight, since OKAMI's
             published 0.04 / 0.08 ratio was tuned for Pink's residual
             scaling and does not transfer verbatim (see ik_keypoint.py).
  b1         Our IROS direction-matching retargeter, with and without the
             yaw regularisers, for continuity with the workshop paper.
  taskcentric  No posture term at all. Wrist position plus robot-native
             regularisers.

The point is the ORDERING. If limit margin falls monotonically as posture
weight rises, the mechanism claim is a dose-response curve across published
methods, not a comparison against a baseline we built to lose.

Usage:
    /usr/bin/python3 src/b3_published_baselines.py --side right
    /usr/bin/python3 src/b3_published_baselines.py --captures M2_1 --quick
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
from common.ik_keypoint import solve_h2o_style, solve_okami_style  # noqa: E402
from common.human_directions import (  # noqa: E402
    load_frames, per_frame_inputs, robot_scaled_arm_targets)
from common.metrics import (  # noqa: E402
    flip_count, limit_violation, limit_margin, normalized_manipulability,
    approach_direction_error_deg)
from common.data_paths import joints_csv_for  # noqa: E402

REST = {"right": np.array([0.30, -0.24, 0.12, 0.69]),
        "left": np.array([0.31, 0.24, -0.12, 0.67])}

# (label, family, kwargs, effective posture weight relative to wrist position).
#
# The OKAMI rows sweep a gain on its published posture weights (shoulder 0.04,
# elbow 0.08) while holding wrist position at 1.0. gain=1.0 is OKAMI as
# published; lower gains walk toward a pure task-space solve. This is the
# dose-response axis: posture weight in, feasibility out.
def _okami(gain):
    return dict(w_shoulder=0.04 * gain, w_elbow=0.08 * gain, pos_scale=1.0)


CONDITIONS = [
    ("h2o-keypoint",      "h2o",   dict(),            "hard"),
    ("okami-gain1.0",     "okami", _okami(1.0),       "0.04 / 0.08"),
    ("okami-gain0.1",     "okami", _okami(0.1),       "0.004 / 0.008"),
    ("okami-gain0.01",    "okami", _okami(0.01),      "4e-4 / 8e-4"),
    ("okami-gain0.001",   "okami", _okami(0.001),     "4e-5 / 8e-5"),
    ("b1-naive",          "b1",    dict(yaw_reg=0.0, yaw_neutral=0.0),   "hard"),
    ("b1-iros-reg",       "b1",    dict(yaw_reg=0.05, yaw_neutral=0.02), "hard"),
    ("taskcentric",       "task",  dict(),            "none"),
]


def solve_rep(joints_csv, side, urdf, fps, label, family, kw):
    fk = ArmChainFK(urdf, side)
    lims = load_limits_4(urdf)[side]
    lo, hi = lims["lo"], lims["hi"]
    q_prev = REST[side].copy()

    frames = load_frames(joints_csv, fps=fps)
    ts, qs, margins, viols, manips, tip_errs, appr = [], [], [], [], [], [], []

    for t, u_hat, f_hat, _w in per_frame_inputs(frames, side, fps):
        elbow_t, wrist_t = robot_scaled_arm_targets(fk, u_hat, f_hat)(q_prev)

        if family == "h2o":
            q, _r, _i = solve_h2o_style(fk, elbow_t, wrist_t, q_prev, lims, **kw)
        elif family == "okami":
            q, _r, _i = solve_okami_style(fk, wrist_t, u_hat, f_hat, q_prev,
                                          lims, **kw)
        elif family == "b1":
            q, _r, _c = solve_fidelity(fk, u_hat, f_hat, q_prev, lims, **kw)
        elif family == "task":
            q, _r, _i = solve_taskcentric(fk, wrist_t, q_prev, lims,
                                          lambda_limit=0.05, lambda_cont=0.02)
        else:
            raise ValueError(family)

        ts.append(t)
        qs.append(q)
        margins.append(limit_margin(q, lo, hi))
        viols.append(limit_violation(q, lo, hi))
        manips.append(normalized_manipulability(fk.tip_jacobian(q)))
        tip_errs.append(float(np.linalg.norm(fk.tip(q) - wrist_t)))
        appr.append(approach_direction_error_deg(q, fk, f_hat))
        q_prev = q

    ts = np.asarray(ts)
    qs = np.asarray(qs)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0 / fps
    return dict(
        n_frames=int(len(ts)),
        flips_yaw=int(flip_count(qs, dt, threshold_rad_per_s=6.0, joint_index=2)),
        margin_mean=float(np.mean(margins)),
        margin_min=float(np.min(margins)),
        violation_frames=int(np.sum(viols)),
        any_violation=bool(np.any(viols)),
        manip_mean=float(np.mean(manips)),
        tip_err_mean_mm=float(np.mean(tip_errs) * 1000.0),
        approach_err_mean_deg=float(np.nanmean(appr)),
    ), qs


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", nargs="+",
                   default=["M2_1", "M2_2", "M2_3", "M3_1", "M3_2", "M3_3"])
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--quick", action="store_true",
                   help="h2o, okami-w1.0 and taskcentric only")
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "b3", "published_baselines.json"))
    args = p.parse_args()

    conds = CONDITIONS
    if args.quick:
        keep = {"h2o-keypoint", "okami-gain1.0", "taskcentric"}
        conds = [c for c in CONDITIONS if c[0] in keep]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary = dict(side=args.side, captures=args.captures, per_rep={},
                   totals={})

    for label, family, kw, posture_w in conds:
        rows = []
        for cap in args.captures:
            jc = joints_csv_for(cap)
            if not os.path.isfile(jc):
                print(f"[b3] SKIP {cap}: no joints CSV", file=sys.stderr)
                continue
            stats, _qs = solve_rep(jc, args.side, args.urdf, args.fps,
                                   label, family, kw)
            stats["capture"] = cap
            rows.append(stats)
        summary["per_rep"][label] = rows
        summary["totals"][label] = dict(
            posture_weight=posture_w,
            total_flips=int(sum(r["flips_yaw"] for r in rows)),
            reps_with_violation=int(sum(r["any_violation"] for r in rows)),
            reps_total=len(rows),
            margin_mean=float(np.mean([r["margin_mean"] for r in rows])),
            manip_mean=float(np.mean([r["manip_mean"] for r in rows])),
            tip_err_mean_mm=float(np.mean([r["tip_err_mean_mm"] for r in rows])),
            approach_err_mean_deg=float(
                np.mean([r["approach_err_mean_deg"] for r in rows])),
        )
        t = summary["totals"][label]
        print(f"[b3] {label:<16s} flips={t['total_flips']:>3d}  "
              f"viol={t['reps_with_violation']}/{t['reps_total']}  "
              f"margin={t['margin_mean']:.3f}  "
              f"manip={t['manip_mean']:.3f}  "
              f"tip_mm={t['tip_err_mean_mm']:.2f}  "
              f"appr_deg={t['approach_err_mean_deg']:.1f}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "posture_weight", "flips", "viol_reps",
                    "reps", "margin", "manip", "tip_mm", "approach_deg"])
        for label, _f, _k, _pw in conds:
            t = summary["totals"][label]
            w.writerow([label, t["posture_weight"], t["total_flips"],
                        t["reps_with_violation"], t["reps_total"],
                        f"{t['margin_mean']:.4f}", f"{t['manip_mean']:.4f}",
                        f"{t['tip_err_mean_mm']:.3f}",
                        f"{t['approach_err_mean_deg']:.2f}"])
    print(f"[b3] wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
