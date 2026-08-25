#!/usr/bin/env python3
"""X3 (NON-NEGOTIABLE): hardware replay of generated Robot-Specific motion.

STATUS: skeleton only. Runs on the real robot; requires the interface flag
and human presence at the e-stop. Reuses replay_arm.py + sdk_replay_logger.py
from experiments/scripts/, so this file is a thin shim that:
  1. picks a trajectory (produced by x2_robot_optimal.py or x1) for a rep,
  2. calls replay_arm.py with --gravity-ff at gain 1.0 (IROS-frozen),
  3. calls sdk_replay_logger.py to capture cmd/exe/torque logs at 50 Hz,
  4. runs experiments/scripts/success_check-style FK to grade tip tracking,
  5. checks whether the X2 limit-violation frames actually manifest on
     hardware or whether the low-level controller clamps them silently.

Plan quote: "The specific thing to check: do the X2 limit violations
actually manifest during replay, or does the controller clamp them silently?
Either answer matters and the second needs reporting."

DO NOT run this without an e-stop operator present.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_SCRIPTS = os.path.expanduser(
    "~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/scripts")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectory_csv", help="output of x2_robot_optimal.py or x1")
    p.add_argument("--interface", required=True,
                   help="network interface for unitree_sdk2py (e.g. enp128s31f6)")
    p.add_argument("--gravity-gain", type=float, default=1.0)
    p.add_argument("--log-dir", default=os.path.join(HERE, "..", "results", "x3"))
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands that would run and exit")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.trajectory_csv))[0]
    log_path = os.path.join(args.log_dir, f"{tag}.replay.csv")

    replay = [
        "/usr/bin/python3", os.path.join(EXP_SCRIPTS, "replay_arm.py"),
        args.trajectory_csv,
        "--interface", args.interface,
        "--gravity-ff", "--gravity-gain", str(args.gravity_gain),
    ]
    logger = [
        "/usr/bin/python3", os.path.join(EXP_SCRIPTS, "sdk_replay_logger.py"),
        "--interface", args.interface,
        "--out", log_path,
    ]

    print("[x3] WILL RUN (e-stop operator required):")
    print("  logger: " + " ".join(logger))
    print("  replay: " + " ".join(replay))
    if args.dry_run:
        return
    # start the logger in the background, then run the replayer in the foreground
    log_proc = subprocess.Popen(logger)
    try:
        subprocess.run(replay, check=True)
    finally:
        log_proc.terminate()
        log_proc.wait()
    print(f"[x3] logs at {log_path}")
    print("[x3] TODO: pass through success_check-style FK to grade tip"
          " tracking and report any limit-violation frames from the replay log.")


if __name__ == "__main__":
    main()
