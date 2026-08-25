#!/usr/bin/env python3
"""X7: same motion, two modes (replica vs replacement) to demonstrate C4.

STATUS: skeleton only. Takes one capture (default M2_1) and produces two
trajectories:

  replica     : ik_baseline.solve_fidelity with the IROS regularizers
                (yaw_reg=0.05, yaw_neutral=0.02). Should land in the human's
                posture band per the IROS RULA finding.
  replacement : ik_taskcentric.solve_taskcentric. Does the same wrist targets
                from a configuration a worker would not adopt.

Report: RULA grand score per frame for both, side-by-side; two-arm mechanical
work; execution time and center-of-hands trajectory length. If the two modes
land in overlapping posture bands, C4 loses evidence and the section needs
to argue rather than demonstrate.

Depends on the RULA scoring code in
experiments/scripts/rula_reba_score.py (do not reimplement).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4, SIDES  # noqa: E402
from common.ik_baseline import solve_fidelity  # noqa: E402
from common.ik_taskcentric import solve_taskcentric  # noqa: E402
from common.human_directions import (  # noqa: E402
    load_frames, per_frame_inputs, robot_scaled_wrist_target)
from common.data_paths import ACTIVE_DATA_ROOT as DATA_ROOT, joints_csv_for  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", default="M2_1")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out-dir", default=os.path.join(HERE, "..", "results", "x7"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    joints_csv = joints_csv_for(args.capture)
    if not os.path.isfile(joints_csv):
        sys.exit(f"not found: {joints_csv}")

    print("[x7] STATUS: skeleton only. TODO:")
    print(" 1) solve REPLICA mode with solve_fidelity(yaw_reg=0.05,"
          " yaw_neutral=0.02) for both arms.")
    print(" 2) solve REPLACEMENT mode with solve_taskcentric for both arms.")
    print(" 3) score both with experiments/scripts/rula_reba_score.py and"
          " compare against the human capture bands from the IROS §II.A output.")
    print(" 4) write two trajectory CSVs (traj_replica.csv, traj_replacement.csv)"
          " and one RULA comparison CSV to results/x7/.")


if __name__ == "__main__":
    main()
