#!/usr/bin/env python3
"""X9 reach-extent sweep: does the limit-violation rate rise with reach extent?

Turns the C1 evidence from a count into a trend. For each reach-sweep capture we
measure how far the subject reached, then solve the same rep two ways, fidelity
and task-centric, and report violations and margin against that reach.

Reach is reported two ways, because they answer different questions:

  human_reach  : p95 shoulder-to-wrist distance in the subject's own torso
                 frame. This is what the subject did, in metres.
  target_reach : p95 distance from the robot shoulder to the ROBOT-SCALED wrist
                 target, which is what the arm is actually asked to achieve.
                 Built from the robot's own limb lengths, so it is capped near
                 the robot's arm length. When it sits at that cap the arm is
                 near full extension and near-singular, which is the regime
                 where every method violates and the metric loses its power.

The second one is the honest x-axis for the curve. Copying the subject's arm
EXTENSION is a configuration property, not a task property. A task-centric
retargeter is free to accomplish the same task at a less extended, better
conditioned posture. This script reports extension_ratio so that saturation is
visible rather than hidden.

    reach_extent_sweep.py --side right
    reach_extent_sweep.py --captures RS_d1_1 RS_d1_2 --out results/reach_extent/sweep.json
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4      # noqa: E402
from common.ik_baseline import solve_fidelity                       # noqa: E402
from common.ik_taskcentric import solve_taskcentric                 # noqa: E402
from common.human_directions import (                               # noqa: E402
    load_frames, per_frame_inputs, robot_scaled_wrist_target,
    task_defined_wrist_target, human_extension_ratio)
from common.metrics import limit_violation, limit_margin            # noqa: E402
from common.data_paths import joints_csv_for                        # noqa: E402

DEFAULT_CAPTURES = [f"RS_d{d}_{r}" for d in (1, 2, 3, 4) for r in range(1, 6)]


def solve_rep(joints_csv, side, urdf, fps=60.0, ext_cap=0.88):
    """Solve one rep three ways. Returns per-method stats plus reach measures.

    fidelity     direction matching, the field baseline
    taskcentric  chases the direction-reconstructed wrist target, which carries
                 the human's extension
    taskdefined  chases the same DIRECTION but at a task-appropriate extension
    """
    fk = ArmChainFK(urdf, side)
    lims = load_limits_4(urdf)[side]
    q0 = np.array([0.30 if side == "right" else 0.31,
                   -0.24 if side == "right" else 0.24,
                   0.12 if side == "right" else -0.12,
                   0.69 if side == "right" else 0.67])
    frames = load_frames(joints_csv, fps=fps)
    rows = list(per_frame_inputs(frames, side, fps))
    if not rows:
        return None

    human_reach = [float(np.linalg.norm(w)) for _t, _u, _f, w in rows]

    # Per-capture rescale so the whole reach profile lands in the robot's usable
    # band, rather than clipping only the far end.
    ext_p95 = human_extension_ratio(fk, rows, q0)
    ext_scale = (ext_cap / ext_p95) if ext_p95 > 1e-6 else 1.0

    out = {}
    target_reach = []
    for method in ("fidelity", "taskcentric", "taskdefined"):
        q_prev = q0.copy()
        viols, margins, tip_err = [], [], []
        for _t, u_hat, f_hat, _w in rows:
            if method == "taskdefined":
                build = task_defined_wrist_target(
                    fk, u_hat, f_hat, ext_scale=ext_scale, ext_cap=ext_cap)
            else:
                build = robot_scaled_wrist_target(fk, u_hat, f_hat)
            target = build(q_prev)
            if method == "fidelity":
                q, _r, _c = solve_fidelity(fk, u_hat, f_hat, q_prev, lims,
                                           yaw_reg=0.0, yaw_neutral=0.0)
                shoulder = fk.positions(q_prev)[0]
                target_reach.append(float(np.linalg.norm(target - shoulder)))
            else:
                q, _r, _i = solve_taskcentric(fk, target, q_prev, lims,
                                              lambda_limit=0.05,
                                              lambda_cont=0.02,
                                              lambda_manip=0.0)
                tip_err.append(float(np.linalg.norm(fk.tip(q) - target)))
            viols.append(limit_violation(q, lims["lo"], lims["hi"]))
            margins.append(limit_margin(q, lims["lo"], lims["hi"]))
            q_prev = q
        out[method] = dict(
            viol_frames=int(np.sum(viols)),
            viol_rate=float(np.mean(viols)),
            any_violation=bool(np.any(viols)),
            margin_mean=float(np.mean(margins)),
            tip_err_mean_mm=(float(np.mean(tip_err) * 1000.0) if tip_err else None),
        )
    out["ext_scale"] = ext_scale
    out["ext_cap"] = ext_cap

    # Robot arm length at the rest pose, for the extension ratio.
    a, e, w = fk.positions(q0)
    arm_len = float(np.linalg.norm(e - a) + np.linalg.norm(w - e))
    tr = float(np.percentile(target_reach, 95)) if target_reach else float("nan")
    out["n_frames"] = len(rows)
    out["human_reach_p95_m"] = float(np.percentile(human_reach, 95))
    out["target_reach_p95_m"] = tr
    out["robot_arm_len_m"] = arm_len
    out["extension_ratio"] = tr / arm_len if arm_len else float("nan")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", nargs="+", default=DEFAULT_CAPTURES)
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--ext-cap", type=float, default=0.88,
                   help="extension ceiling for the task-defined target")
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "reach_extent", "sweep.json"))
    args = p.parse_args()

    per_rep = []
    print(f"{'capture':11}{'ext%':>7}"
          f"{'fid_v%':>8}{'task_v%':>9}{'tdef_v%':>9}"
          f"{'fid_marg':>10}{'task_marg':>11}{'tdef_marg':>11}{'tdef_mm':>9}")
    for cap in args.captures:
        path = joints_csv_for(cap)
        if not os.path.isfile(path):
            print(f"{cap:11}  MISSING {path}")
            continue
        r = solve_rep(path, args.side, args.urdf, args.fps, args.ext_cap)
        if r is None:
            print(f"{cap:11}  no usable frames")
            continue
        r["capture"] = cap
        r["d_level"] = cap.split("_")[1] if "_" in cap else ""
        per_rep.append(r)
        td = r["taskdefined"]
        print(f"{cap:11}{100*r['extension_ratio']:7.1f}"
              f"{100*r['fidelity']['viol_rate']:8.1f}"
              f"{100*r['taskcentric']['viol_rate']:9.1f}"
              f"{100*td['viol_rate']:9.1f}"
              f"{r['fidelity']['margin_mean']:10.3f}"
              f"{r['taskcentric']['margin_mean']:11.3f}"
              f"{td['margin_mean']:11.3f}"
              f"{(td['tip_err_mean_mm'] or 0):9.1f}")

    # Aggregate by d-level, which is the controlled variable.
    by_level = {}
    for r in per_rep:
        by_level.setdefault(r["d_level"], []).append(r)
    print("\nBY TARGET DISTANCE   (reps violating out of n)")
    print(f"{'level':7}{'n':>3}{'human_m':>9}{'ext%':>7}"
          f"{'fidelity':>11}{'taskcentric':>13}{'taskdefined':>13}"
          f"{'fid_marg':>10}{'tdef_marg':>11}")
    summary = {}
    for lvl in sorted(by_level):
        g = by_level[lvl]
        s = dict(
            n=len(g),
            human_reach_m=float(np.mean([x["human_reach_p95_m"] for x in g])),
            extension_ratio=float(np.mean([x["extension_ratio"] for x in g])),
            fid_viol_rate=float(np.mean([x["fidelity"]["viol_rate"] for x in g])),
            task_viol_rate=float(np.mean([x["taskcentric"]["viol_rate"] for x in g])),
            tdef_viol_rate=float(np.mean([x["taskdefined"]["viol_rate"] for x in g])),
            fid_reps_violating=int(sum(x["fidelity"]["any_violation"] for x in g)),
            task_reps_violating=int(sum(x["taskcentric"]["any_violation"] for x in g)),
            tdef_reps_violating=int(sum(x["taskdefined"]["any_violation"] for x in g)),
            fid_margin=float(np.mean([x["fidelity"]["margin_mean"] for x in g])),
            task_margin=float(np.mean([x["taskcentric"]["margin_mean"] for x in g])),
            tdef_margin=float(np.mean([x["taskdefined"]["margin_mean"] for x in g])),
        )
        summary[lvl] = s
        print(f"{lvl:7}{s['n']:3d}{s['human_reach_m']:9.3f}"
              f"{100*s['extension_ratio']:7.1f}"
              f"{s['fid_reps_violating']:>8}/{s['n']:<2}"
              f"{s['task_reps_violating']:>10}/{s['n']:<2}"
              f"{s['tdef_reps_violating']:>10}/{s['n']:<2}"
              f"{s['fid_margin']:10.3f}{s['tdef_margin']:11.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(side=args.side, urdf=args.urdf,
                   per_rep=per_rep, by_level=summary),
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
