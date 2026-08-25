#!/usr/bin/env python3
"""X1 (PRIMARY): automated Robot-Specific generation across M1/M2/M3.

Runs the same task-centric IK as x6/x2 on all three captured motion classes,
produces one replay-ready trajectory per rep, and reports the IROS-style
efficiency measures (execution time, block-center trajectory length, two-arm
mechanical work) so the generated motion can be compared against the IROS
hand-designed baseline on the SAME axes.

STATUS: skeleton only. This is the primary claim (C2) and it needs the
refilmed captures before its numbers are worth reporting. What is here now
is (a) the loop over M1/M2/M3, (b) the trajectory writer (delegates to
x2_robot_optimal.solve_arm_series), and (c) placeholders for the three
efficiency measures.

Efficiency measures (definitions match IROS §III, so numbers are comparable):
  execution_time_s     : t_out[-1] - t_out[0] (trajectory duration)
  trajectory_length_m  : sum |dp_center| where p_center is the midpoint of
                         the two wrist tips (matches the IROS "block center")
  mech_work_two_arm_J  : sum over both arms of int |tau * dq| where tau is
                         a static gravity torque proxy (URDF-derived, or the
                         effort-weighted |q - q_rest| stand-in used in X2).
                         NEEDS a real gravity/inertia model before publishing.

Reviewer objection to anticipate: the IROS "hand-designed" motion is a
different construct (pick-and-stack on a specific block layout), so a like-
for-like comparison requires reusing the IROS pick-and-stack targets, not
running M1/M2/M3 through this pipeline. Two ways to close this:
  (a) run x1 on the IROS block sequence exactly (needs the s2_bricklay spec)
  (b) report M1/M2/M3 efficiency independently and say the IROS comparison
      lives in the workshop paper. Decide with advisors.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4, SIDES  # noqa: E402
from x2_robot_optimal import solve_arm_series, OUT_JOINTS  # noqa: E402
from common.data_paths import ACTIVE_DATA_ROOT as DATA_ROOT, joints_csv_for  # noqa: E402


def two_arm_center_length(t_left, q_left, t_right, q_right, urdf):
    """Trajectory length of the midpoint of the two wrist tips (metres)."""
    fkL = ArmChainFK(urdf, "left")
    fkR = ArmChainFK(urdf, "right")
    t0 = max(t_left[0], t_right[0])
    t1 = min(t_left[-1], t_right[-1])
    n = max(2, int((t1 - t0) * 50))
    ts = np.linspace(t0, t1, n)
    def sample(t, q, tq):
        return np.stack([np.interp(t, tq, q[:, k]) for k in range(4)], axis=1)
    qL = sample(ts, q_left, t_left)
    qR = sample(ts, q_right, t_right)
    tips_center = np.array([
        0.5 * (fkL.tip(qL[i]) + fkR.tip(qR[i])) for i in range(n)])
    return float(np.sum(np.linalg.norm(np.diff(tips_center, axis=0), axis=1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", nargs="+", default=[
        "M1_R_1", "M1_R_2", "M1_R_3",
        "M2_1", "M2_2", "M2_3",
        "M3_1", "M3_2", "M3_3"])
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--out-dir", default=os.path.join(HERE, "..", "results", "x1"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[x1] STATUS: preview run on OLD captures; refilm is a prerequisite.")
    for cap in args.captures:
        joints_csv = joints_csv_for(cap)
        if not os.path.isfile(joints_csv):
            print(f"[x1] SKIP {cap}: no joints CSV at {joints_csv}",
                  file=sys.stderr); continue
        try:
            tL, qL = solve_arm_series(joints_csv, "left", args.urdf, args.fps)
            tR, qR = solve_arm_series(joints_csv, "right", args.urdf, args.fps)
        except Exception as e:
            print(f"[x1] SKIP {cap}: {e}", file=sys.stderr); continue
        exec_s = min(tL[-1] - tL[0], tR[-1] - tR[0])
        length_m = two_arm_center_length(tL, qL, tR, qR, args.urdf)
        # placeholder mechanical-work proxy: sum |dq| * effort
        lims = load_limits_4(args.urdf)
        work_J = 0.0
        for side, ts, qs in (("left", tL, qL), ("right", tR, qR)):
            dq = np.abs(np.diff(qs, axis=0))
            eff = lims[side]["effort"]
            work_J += float(np.sum(dq * eff))
        print(f"[x1] {cap:>8}: exec_s={exec_s:>5.2f}"
              f"  center_len_m={length_m:>5.2f}"
              f"  work_proxy_J={work_J:>7.1f}")


if __name__ == "__main__":
    main()
