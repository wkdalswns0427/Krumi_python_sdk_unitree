#!/usr/bin/env python3
"""replay_logger.py - Block 2 commanded-vs-executed joint logger for H1-2.

Subscribes to the arm command stream (rt/arm_sdk, LowCmd_) and the robot
state (rt/lowstate, LowState_) and writes one synchronized CSV row per tick
at a fixed rate (default 50 Hz) for the duration of a Block 2 replay trial.

This is passive: it does NOT command the robot. Run it alongside whatever
node is replaying the retargeted trajectory through rt/arm_sdk.

Logged per row: wall time, relative time, and for each of the 15 arm/waist
joints the commanded q and executed q, plus the arm_sdk blend weight and
mode_machine (a safety-stop proxy). The commanded-vs-executed joint error
(the "free" secondary fidelity metric) is summarized on exit.

The success criterion (5 cm end-effector via FK, no safety stop, IK
singularity counter) is evaluated OFFLINE from these logs - see README.

Usage:
    # real trial (robot running, a replay node publishing to rt/arm_sdk):
    python3 replay_logger.py enp128s31f6 --out replay_M1_T1.csv --duration 20

    # no robot, exercise the sampler + CSV + summary:
    python3 replay_logger.py --self-test --out /tmp/replay_selftest.csv
"""

import argparse
import csv
import math
import os
import sys
import threading
import time


# ── H1-2 arm_sdk control vector (must match wave_on_person.py) ───────────────
class J:
    WaistYaw = 12
    LShoulderPitch = 13
    LShoulderRoll = 14
    LShoulderYaw = 15
    LElbow = 16
    LWristRoll = 17
    LWristPitch = 18
    LWristYaw = 19
    RShoulderPitch = 20
    RShoulderRoll = 21
    RShoulderYaw = 22
    RElbow = 23
    RWristRoll = 24
    RWristPitch = 25
    RWristYaw = 26
    Weight = 27  # arm_sdk blend (0..1)


ARM_JOINTS = [
    ("L_shoulder_pitch", J.LShoulderPitch), ("L_shoulder_roll", J.LShoulderRoll),
    ("L_shoulder_yaw", J.LShoulderYaw), ("L_elbow", J.LElbow),
    ("L_wrist_roll", J.LWristRoll), ("L_wrist_pitch", J.LWristPitch),
    ("L_wrist_yaw", J.LWristYaw),
    ("R_shoulder_pitch", J.RShoulderPitch), ("R_shoulder_roll", J.RShoulderRoll),
    ("R_shoulder_yaw", J.RShoulderYaw), ("R_elbow", J.RElbow),
    ("R_wrist_roll", J.RWristRoll), ("R_wrist_pitch", J.RWristPitch),
    ("R_wrist_yaw", J.RWristYaw),
    ("waist_yaw", J.WaistYaw),
]
JOINT_NAMES = [name for name, _ in ARM_JOINTS]
JOINT_IDX = [idx for _, idx in ARM_JOINTS]


class Logger:
    """Holds the latest command/state samples and writes synchronized rows.

    DDS callbacks (real mode) or the self-test feeder both just update
    ``latest_cmd`` / ``latest_exe``; the fixed-rate tick snapshots them.
    """

    def __init__(self, out_path, rate_hz):
        self.out_path = out_path
        self.dt = 1.0 / rate_hz
        self._lock = threading.Lock()
        self.latest_cmd = None      # list[15] commanded q
        self.latest_cmd_weight = float("nan")
        self.latest_exe = None      # list[15] executed q
        self.latest_mode = -1
        self.t0 = None
        self.rows = []              # kept for the exit summary

        self._fh = open(out_path, "w", newline="")
        self._w = csv.writer(self._fh)
        header = ["t_wall", "t_rel", "weight", "mode_machine"]
        header += [f"cmd_{n}" for n in JOINT_NAMES]
        header += [f"exe_{n}" for n in JOINT_NAMES]
        self._w.writerow(header)

    # ── sample setters (called by DDS callbacks or self-test) ────────────────
    def set_cmd(self, q15, weight):
        with self._lock:
            self.latest_cmd = list(q15)
            self.latest_cmd_weight = float(weight)

    def set_exe(self, q15, mode_machine):
        with self._lock:
            self.latest_exe = list(q15)
            self.latest_mode = int(mode_machine)

    # ── one synchronized row ─────────────────────────────────────────────────
    def tick(self):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        with self._lock:
            cmd = list(self.latest_cmd) if self.latest_cmd else [float("nan")] * 15
            exe = list(self.latest_exe) if self.latest_exe else [float("nan")] * 15
            weight, mode = self.latest_cmd_weight, self.latest_mode
        t_rel = now - self.t0
        row = [f"{now:.6f}", f"{t_rel:.4f}", f"{weight:.4f}", mode]
        row += [f"{v:.6f}" for v in cmd]
        row += [f"{v:.6f}" for v in exe]
        self._w.writerow(row)
        self.rows.append((cmd, exe))

    def close(self):
        self._fh.flush()
        self._fh.close()

    # ── exit summary: commanded-vs-executed error (deg) ──────────────────────
    def summary(self):
        import numpy as np
        paired = [(np.array(c), np.array(e)) for c, e in self.rows
                  if not (any(map(math.isnan, c)) or any(map(math.isnan, e)))]
        n = len(paired)
        print("=" * 60)
        print("Block 2 replay log summary")
        print("=" * 60)
        print(f"  rows written        : {len(self.rows)}  -> {self.out_path}")
        print(f"  fully-paired rows   : {n}")
        if self.t0 is not None and self.rows:
            dur = len(self.rows) * self.dt
            print(f"  duration (approx)   : {dur:.2f} s  "
                  f"(target {1.0 / self.dt:.0f} Hz)")
        if n == 0:
            print("  (no paired cmd+exe rows - was a replay node publishing?)")
            print("=" * 60)
            return
        err = np.array([np.abs(c - e) for c, e in paired])  # n x 15, radians
        err_deg = np.degrees(err)
        print("-" * 60)
        print(f"  {'joint':<18}{'mean(deg)':>10}{'max(deg)':>10}")
        for k, name in enumerate(JOINT_NAMES):
            print(f"  {name:<18}{err_deg[:, k].mean():>10.3f}"
                  f"{err_deg[:, k].max():>10.3f}")
        print("-" * 60)
        print(f"  overall mean |cmd-exe| : {err_deg.mean():.3f} deg")
        print(f"  overall max  |cmd-exe| : {err_deg.max():.3f} deg")
        print("=" * 60)


# ── Real DDS wiring ──────────────────────────────────────────────────────────
def run_live(logger, interface, duration):
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelSubscriber)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    if interface:
        ChannelFactoryInitialize(0, interface)
    else:
        ChannelFactoryInitialize(0)

    def on_cmd(msg: "LowCmd_"):
        q = [msg.motor_cmd[i].q for i in JOINT_IDX]
        logger.set_cmd(q, msg.motor_cmd[J.Weight].q)

    def on_state(msg: "LowState_"):
        q = [msg.motor_state[i].q for i in JOINT_IDX]
        logger.set_exe(q, msg.mode_machine)

    cmd_sub = ChannelSubscriber("rt/arm_sdk", LowCmd_)
    cmd_sub.Init(on_cmd, 10)
    state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    state_sub.Init(on_state, 10)

    print(f"[replay_logger] logging rt/arm_sdk + rt/lowstate @ "
          f"{1.0 / logger.dt:.0f} Hz -> {logger.out_path}")
    print("[replay_logger] waiting for first samples...")
    t_start = time.time()
    while logger.latest_cmd is None and logger.latest_exe is None:
        if time.time() - t_start > 10.0:
            print("[replay_logger] WARNING: no cmd/state after 10 s - "
                  "logging NaNs. Is the robot up and a replay node publishing?")
            break
        time.sleep(0.05)

    _tick_loop(logger, duration, "Ctrl+C to stop")


def _tick_loop(logger, duration, stop_hint):
    print(f"[replay_logger] recording ({stop_hint})...")
    next_t = time.time()
    try:
        while True:
            logger.tick()
            if duration and (len(logger.rows) * logger.dt) >= duration:
                print(f"[replay_logger] reached --duration {duration}s")
                break
            next_t += logger.dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()  # fell behind; resync
    except KeyboardInterrupt:
        print("\n[replay_logger] stopped by user")


# ── Self-test (no DDS) ───────────────────────────────────────────────────────
def run_self_test(logger, duration):
    """Feed synthetic commanded (sinusoid) + executed (lagged) samples."""
    print("[replay_logger] SELF-TEST: synthetic cmd/exe, no DDS")
    stop = threading.Event()

    def feeder():
        t0 = time.time()
        exe_state = [0.0] * 15
        while not stop.is_set():
            t = time.time() - t0
            cmd = [0.5 * math.sin(2 * math.pi * 0.3 * t + 0.1 * k)
                   for k in range(15)]
            # first-order lag executed toward commanded
            for k in range(15):
                exe_state[k] += 0.25 * (cmd[k] - exe_state[k])
            logger.set_cmd(cmd, 1.0)
            logger.set_exe(exe_state, 4)  # mode_machine=4 nominal
            time.sleep(0.005)  # 200 Hz feeder, faster than the 50 Hz tick

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    _tick_loop(logger, duration or 2.0, f"auto-stop {duration or 2.0}s")
    stop.set()
    th.join(timeout=1.0)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("interface", nargs="?",
                   help="robot network interface, e.g. enp128s31f6")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--rate", type=float, default=50.0,
                   help="logging rate in Hz (default: 50)")
    p.add_argument("--duration", type=float, default=None,
                   help="auto-stop after this many seconds (default: until Ctrl+C)")
    p.add_argument("--self-test", action="store_true",
                   help="run without a robot using synthetic samples")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    logger = Logger(args.out, args.rate)
    try:
        if args.self_test:
            run_self_test(logger, args.duration)
        else:
            run_live(logger, args.interface, args.duration)
    finally:
        logger.close()
        logger.summary()


if __name__ == "__main__":
    main()
