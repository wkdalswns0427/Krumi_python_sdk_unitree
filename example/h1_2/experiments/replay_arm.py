#!/usr/bin/env python3
"""replay_arm.py - play a retargeted arm trajectory on the H1-2 (Block 2).

Reads the trajectory CSV from retarget_arm.py and publishes it to the
rt/arm_sdk overlay at 50 Hz, same pattern as wave_on_person.py: the robot
must already be STANDING in balance (FSM 204); this script only blends the
arm overlay in and out and never touches locomotion.

Sequence (every step operator-gated or interruptible):
  1. Wait for rt/lowstate, snapshot the current arm pose.
  2. ENTER -> fade arm_sdk weight 0 -> 1 (2 s) while holding the snapshot.
  3. Ramp from the snapshot to the trajectory start pose (3 s).
  4. ENTER -> play the trajectory, time-stretched by --speed-scale
     (0.5 = half speed, first trial per protocol).
  5. Hold the final pose 1 s, ramp back to the snapshot pose (3 s),
     fade weight 1 -> 0 (2 s). Loco takes the arms back.
  Ctrl+C at ANY point: freeze current command, ramp weight out, exit.

Safety (protocol): clear workspace, arms-only, second person on the e-stop,
first trial of each motion at --speed-scale 0.5.

Usage:
    # pre-flight, no robot, no DDS:
    /usr/bin/python3 replay_arm.py TRAJ.csv --dry-run
    # on the robot (conda env with unitree_sdk2py, robot standing):
    python3 replay_arm.py TRAJ.csv --interface enp128s31f6 --speed-scale 0.5

Logging: run the h12_experiments replay_logger in another terminal to record
cmd vs exe for success_check.py.
"""

import argparse
import csv
import os
import sys
import time

# ── H1-2 arm_sdk motor indices (matches wave_on_person.py / joints.py) ───────
SDK_INDEX = {
    "left_shoulder_pitch": 13, "left_shoulder_roll": 14,
    "left_shoulder_yaw": 15, "left_elbow": 16,
    "left_wrist_roll": 17, "left_wrist_pitch": 18, "left_wrist_yaw": 19,
    "right_shoulder_pitch": 20, "right_shoulder_roll": 21,
    "right_shoulder_yaw": 22, "right_elbow": 23,
    "right_wrist_roll": 24, "right_wrist_pitch": 25, "right_wrist_yaw": 26,
    "waist_yaw": 12,
}
WEIGHT_IDX = 27
KP = {"shoulder_pitch": 80, "shoulder_roll": 80, "shoulder_yaw": 60,
      "elbow": 60, "wrist_roll": 30, "wrist_pitch": 30, "wrist_yaw": 30,
      "waist_yaw": 150}
KD = {"shoulder_pitch": 2.0, "shoulder_roll": 2.0, "shoulder_yaw": 1.5,
      "elbow": 1.5, "wrist_roll": 1.0, "wrist_pitch": 1.0, "wrist_yaw": 1.0,
      "waist_yaw": 2.0}


def gains_for(name):
    seg = name.split("_", 1)[1] if name != "waist_yaw" else "waist_yaw"
    return KP[seg], KD[seg]


def load_traj(path):
    """(names, t array, Q array rows aligned to names)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[0] != "t_s":
            sys.exit(f"{path}: expected 't_s' first column, got {header[0]}")
        names = header[1:]
        unknown = [n for n in names if n not in SDK_INDEX]
        if unknown:
            sys.exit(f"{path}: unknown joints {unknown}")
        t, Q = [], []
        for row in reader:
            t.append(float(row[0]))
            Q.append([float(v) for v in row[1:]])
    return names, t, Q


class ArmSdk:
    """rt/arm_sdk publisher + lowstate snapshot (wave_on_person pattern)."""

    CTRL_DT = 0.02

    def __init__(self, names, interface, dry):
        self.names = names
        self.dry = dry
        self.weight = 0.0
        self.pose = None                     # current commanded pose (list)
        if dry:
            return
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        if interface:
            ChannelFactoryInitialize(0, interface)
        else:
            ChannelFactoryInitialize(0)
        self._msg = unitree_hg_msg_dds__LowCmd_()
        self._state = None
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        print("[replay] waiting for rt/lowstate...")
        t0 = time.time()
        while self._state is None:
            if time.time() - t0 > 10:
                sys.exit("[replay] no lowstate after 10 s; robot up? interface?")
            time.sleep(0.05)

    def _on_state(self, msg):
        self._state = msg

    def snapshot(self):
        if self.dry:
            return [0.0] * len(self.names)
        return [self._state.motor_state[SDK_INDEX[n]].q for n in self.names]

    def publish(self, pose):
        self.pose = list(pose)
        if self.dry:
            return
        for n, q in zip(self.names, pose):
            mc = self._msg.motor_cmd[SDK_INDEX[n]]
            kp, kd = gains_for(n)
            mc.mode, mc.q, mc.dq, mc.tau = 1, float(q), 0.0, 0.0
            mc.kp, mc.kd = kp, kd
        self._msg.motor_cmd[WEIGHT_IDX].q = self.weight
        self._pub.Write(self._msg)

    def ramp_weight(self, target, seconds):
        steps = max(1, int(seconds / self.CTRL_DT))
        for i in range(steps):
            self.weight += (target - self.weight) / (steps - i)
            self.publish(self.pose)
            if not self.dry:
                time.sleep(self.CTRL_DT)
        self.weight = target

    def ramp_pose(self, target, seconds):
        steps = max(1, int(seconds / self.CTRL_DT))
        start = list(self.pose)
        for i in range(1, steps + 1):
            a = i / steps
            self.publish([s + a * (t - s) for s, t in zip(start, target)])
            if not self.dry:
                time.sleep(self.CTRL_DT)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traj_csv", help="trajectory from retarget_arm.py")
    p.add_argument("--interface", help="robot network interface, e.g. enp128s31f6")
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="time-stretch playback (0.5 = half speed; protocol: "
                        "first trial of each motion at 0.5)")
    p.add_argument("--max-vel", type=float, default=4.0,
                   help="per-joint rate limit rad/s during playback (default 4)")
    p.add_argument("--dry-run", action="store_true",
                   help="no DDS: validate the file and print the pre-flight")
    args = p.parse_args()

    if not os.path.isfile(args.traj_csv):
        sys.exit(f"not found: {args.traj_csv}")
    if args.speed_scale <= 0 or args.speed_scale > 1.5:
        sys.exit("--speed-scale must be in (0, 1.5]")
    names, t, Q = load_traj(args.traj_csv)
    n = len(t)
    dur = t[-1] - t[0]
    play_dur = dur / args.speed_scale

    # pre-flight: peak commanded velocity after speed scaling
    peak = [0.0] * len(names)
    for i in range(1, n):
        dt = (t[i] - t[i - 1]) / args.speed_scale
        if dt <= 0:
            continue
        for k in range(len(names)):
            v = abs(Q[i][k] - Q[i - 1][k]) / dt
            peak[k] = max(peak[k], v)
    worst = sorted(zip(peak, names), reverse=True)[:3]
    print(f"[replay] {args.traj_csv}: {n} rows, {dur:.1f} s "
          f"-> {play_dur:.1f} s at speed {args.speed_scale:g}")
    print("[replay] peak commanded velocity: "
          + "  ".join(f"{nm}={v:.2f} rad/s" for v, nm in worst))
    sat = [nm for v, nm in zip(peak, names) if v > args.max_vel]
    if sat:
        print(f"[replay] WARNING: {sat} exceed --max-vel {args.max_vel}; the "
              f"rate limit will distort them. Lower --speed-scale instead.")
    if args.dry_run:
        print("[replay] dry-run OK (no DDS, nothing published)")
        return

    print("=" * 66)
    print("H1-2 ARM TRAJECTORY REPLAY (rt/arm_sdk overlay)")
    print("Robot must be STANDING IN BALANCE (FSM 204), workspace clear,")
    print("second person on the e-stop. Loco is never commanded here.")
    print("=" * 66)

    sdk = ArmSdk(names, args.interface, dry=False)
    sdk.pose = sdk.snapshot()
    rest = list(sdk.pose)
    print(f"[replay] rest pose captured: {[round(v, 2) for v in rest]}")

    try:
        input("ENTER to fade the arm overlay in (Ctrl+C aborts)... ")
        sdk.ramp_weight(1.0, 2.0)
        print("[replay] moving to trajectory start (3 s)...")
        sdk.ramp_pose(Q[0], 3.0)
        input(f"ENTER to play ({play_dur:.1f} s at speed "
              f"{args.speed_scale:g})... ")
        t_start = time.time()
        i = 1
        max_step = args.max_vel * ArmSdk.CTRL_DT
        while i < n:
            now = (time.time() - t_start) * args.speed_scale + t[0]
            while i < n and t[i] < now:
                i += 1
            if i >= n:
                break
            # interpolate the target at 'now', then rate-limit toward it
            span = max(t[i] - t[i - 1], 1e-6)
            a = (now - t[i - 1]) / span
            target = [q0 + a * (q1 - q0) for q0, q1 in zip(Q[i - 1], Q[i])]
            cmd = [c + max(-max_step, min(max_step, tg - c))
                   for c, tg in zip(sdk.pose, target)]
            sdk.publish(cmd)
            time.sleep(ArmSdk.CTRL_DT)
        print("[replay] trajectory done; holding 1 s")
        for _ in range(50):
            sdk.publish(sdk.pose)
            time.sleep(ArmSdk.CTRL_DT)
        print("[replay] returning to rest pose (3 s)")
        sdk.ramp_pose(rest, 3.0)
        sdk.ramp_weight(0.0, 2.0)
        print("[replay] overlay released; loco has the arms again")
    except KeyboardInterrupt:
        print("\n[replay] INTERRUPT: freezing pose, fading overlay out")
        try:
            sdk.ramp_weight(0.0, 2.0)
        finally:
            print("[replay] overlay released")
        sys.exit(1)


if __name__ == "__main__":
    main()
