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

Logging: run sdk_replay_logger.py in another terminal (same conda env,
same --interface) to record cmd vs exe for success_check.py.
"""

import argparse
import csv
import os
import sys
import time

# Frozen gravity feedforward gain (2026-07-14 M1 T1 sweep: off 12.35 cm ->
# gain 1.4 mean EE 2.59 cm, 86% within 5 cm; 1.4 is the best tested and still
# slightly under-compensates, so it does not over-drive the arm). Single
# source of truth for the default; override per run with --gravity-gain.
FROZEN_GRAVITY_GAIN = 1.4
# Frozen Block 2 binary-success tolerance (secondary metric; continuous EE
# error is primary). At gain 1.4, 5 cm gives 86.3% of frames within tolerance.
FROZEN_TOLERANCE_M = 0.05


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


# ── gravity feedforward (--gravity-ff) ───────────────────────────────────────
ARM_SEGS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw")


class GravityFF:
    """Per-joint gravity holding torque for one arm, from the URDF inertials.

    Motivation (pilot M1 T1, 2026-07-13): with tau=0 and kp=80 the shoulders
    sagged a systematic ~10 deg (97% one-signed error, present during static
    holds) = ~12 cm mean EE deviation. tau_i = dU/dq_i cancels that sag at
    the source instead of cranking kp.

    Torques include everything distal to each joint (branch subtrees too,
    e.g. the hand below the wrist, branch joints held at q=0), are computed
    in the torso frame with z up (robot standing in balance, per protocol),
    capped per joint, and the caller scales them by the arm_sdk weight.
    Correctness is self-checked at init: the cross-product torque must match
    the finite-difference gradient of the potential energy at random poses,
    else we refuse to run.
    """

    G = 9.81
    TAU_CAP = {"shoulder_pitch": 18.0, "shoulder_roll": 18.0,
               "shoulder_yaw": 6.0, "elbow": 10.0,
               "wrist_roll": 2.0, "wrist_pitch": 2.0, "wrist_yaw": 2.0}

    def __init__(self, urdf_path, side):
        import xml.etree.ElementTree as ET
        import numpy as np
        sys.path.insert(0, os.path.expanduser(
            "~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments"))
        from h12_experiments.fk_h12 import (
            UrdfChain, rpy_to_matrix, axis_angle_matrix, homogeneous,
            BASE_LINK, DEFAULT_TIP)
        self.np = np
        self._rpy, self._aa, self._hom = (rpy_to_matrix, axis_angle_matrix,
                                          homogeneous)
        self.side = side
        self.urdf = UrdfChain(urdf_path)
        self.chain = self.urdf.chain(BASE_LINK, DEFAULT_TIP[side])
        self.chain_set = set(self.chain)
        self.actuated = [j for j in self.chain
                         if self.urdf.joints[j]["type"] in ("revolute",
                                                            "continuous")]
        self.base = BASE_LINK
        self.caps = [self.TAU_CAP[seg] for seg in ARM_SEGS]

        root = ET.parse(urdf_path).getroot()
        self.inertial = {}
        for link in root.findall("link"):
            inr = link.find("inertial")
            if inr is None or inr.find("mass") is None:
                continue
            m = float(inr.find("mass").get("value"))
            org = inr.find("origin")
            xyz = ([float(v) for v in org.get("xyz").split()]
                   if org is not None and org.get("xyz") else [0.0, 0.0, 0.0])
            if m > 0:
                self.inertial[link.get("name")] = (m, np.array(xyz))
        self.kids = {}
        for jn, jd in self.urdf.joints.items():
            self.kids.setdefault(jd["parent"], []).append(jn)

        self._self_check()

    def _walk(self, q):
        """(joints, masses): per actuated chain joint (p, axis) in torso
        frame, and (depth, m, com) for every link at/below it."""
        np = self.np
        qmap = dict(zip(self.actuated, q))
        joints, masses = [], []

        def descend(link, T, depth):
            if depth >= 0 and link in self.inertial:
                m, com = self.inertial[link]
                masses.append((depth, m, (T @ np.append(com, 1.0))[:3]))
            for jn in self.kids.get(link, []):
                jd = self.urdf.joints[jn]
                in_chain = jn in self.chain_set
                if depth < 0 and not in_chain:
                    continue                  # other subtrees off the torso
                To = T @ self._hom(self._rpy(*jd["rpy"]), jd["xyz"])
                d = depth
                if in_chain and jd["type"] in ("revolute", "continuous"):
                    d = len(joints)
                    joints.append((To[:3, 3].copy(),
                                   To[:3, :3] @ np.array(jd["axis"], float)))
                    Tn = To @ self._hom(self._aa(jd["axis"], qmap[jn]),
                                        [0, 0, 0])
                else:                         # fixed or branch joint at q=0
                    Tn = To
                descend(jd["child"], Tn, d)

        descend(self.base, np.eye(4), -1)
        return joints, masses

    def tau(self, q):
        """Holding torques (Nm), capped, for the 7 chain joints at pose q."""
        np = self.np
        joints, masses = self._walk(q)
        ez = np.array([0.0, 0.0, 1.0])
        out = []
        for i, (p, a) in enumerate(joints):
            t = sum(m * self.G * float(ez @ np.cross(a, c - p))
                    for d, m, c in masses if d >= i)
            out.append(float(np.clip(t, -self.caps[i], self.caps[i])))
        return out

    def _potential(self, q):
        _, masses = self._walk(q)
        return sum(m * self.G * c[2] for _, m, c in masses)

    def _self_check(self):
        """tau must equal dU/dq (finite differences) or we refuse to run."""
        np = self.np
        rng = np.random.default_rng(0)
        for _ in range(4):
            q = rng.uniform(-1.2, 1.2, 7)
            t = self.tau(q)
            eps = 1e-6
            for i in range(7):
                qp, qm = q.copy(), q.copy()
                qp[i] += eps
                qm[i] -= eps
                fd = (self._potential(qp) - self._potential(qm)) / (2 * eps)
                if abs(fd) < self.caps[i] and abs(t[i] - fd) > 1e-3:
                    sys.exit(f"[gravity-ff] SELF-CHECK FAILED {self.side} "
                             f"joint {i}: tau {t[i]:.4f} vs dU/dq {fd:.4f}")
        total_m = sum(m for _, m, _ in self._walk([0.0] * 7)[1])
        print(f"[gravity-ff] {self.side}: self-check OK, "
              f"{total_m:.2f} kg distal of shoulder")


def wrist_paths(names, Q, urdf):
    """FK wrist positions per frame: (n, 2, 3) array, [left, right] wrists
    in the torso frame (x forward, z up). Shared by the trim checks."""
    import numpy as np
    from h12_experiments.fk_h12 import H12ArmFK
    fks = [H12ArmFK(urdf, side=s) for s in ("left", "right")]
    idx = [[names.index(f"{s}_{seg}") for seg in ARM_SEGS]
           for s in ("left", "right")]
    out = np.zeros((len(Q), 2, 3))
    for i, row in enumerate(Q):
        for a in (0, 1):
            out[i, a] = fks[a].ee_position([row[k] for k in idx[a]])
    return out


def safe_window(wp, x_min):
    """First/last index where BOTH wrists are at least x_min forward of the
    torso. Leading/trailing frames with a wrist behind the body are
    retargeting artifacts (subject stepping into/out of position); driving
    the go-to-start ramp there swings the arm behind the robot. Interior
    frames untouched. wp = wrist_paths array. Returns (i0, i1) inclusive."""
    behind = (wp[:, :, 0] < x_min).any(axis=1)
    n = len(wp)
    i0 = 0
    while i0 < n - 1 and behind[i0]:
        i0 += 1
    i1 = n - 1
    while i1 > i0 and behind[i1]:
        i1 -= 1
    return i0, i1


def task_window(wp, t, signal, thresh, sustain_s, margin_s):
    """Trim the human's PREPARATION (walk-in, settling) and post-task idling
    from the clip ends, keeping every interior frame.

    Prep is not low-speed (walking swings the arms), so speed cannot find
    it; what separates task from prep is WHERE the wrists are. signal
    selects the per-frame statistic: 'z' = max wrist height (M1 overhead
    work), 'x' = max wrist forward reach (M2/M3 forward work). The signal
    is rolling-median filtered over sustain_s so brief interpolation
    wobbles at the clip head cannot fake a task start, then the window is
    the first..last sustained crossing of thresh, padded by margin_s.
    Returns (i0, i1) inclusive, or None if the signal never crosses."""
    import numpy as np
    sig = wp[:, :, 2].max(axis=1) if signal == "z" else wp[:, :, 0].max(axis=1)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    k = max(1, int(round(sustain_s / dt)) | 1)
    pad = k // 2
    ext = np.pad(sig, pad, mode="edge")
    rolled = np.array([np.median(ext[i:i + k]) for i in range(len(sig))])
    active = np.where(rolled > thresh)[0]
    if len(active) == 0:
        return None
    m = int(round(margin_s / dt))
    return max(0, active[0] - m), min(len(sig) - 1, active[-1] + m)


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

    def __init__(self, names, interface, dry, gravity=None, gravity_gain=1.0):
        self.names = names
        self.dry = dry
        self.weight = 0.0
        self.pose = None                     # current commanded pose (list)
        self.gain_override = {}              # name -> (kp, kd), optional
        # gravity feedforward: {side: GravityFF} + per-side name->slot map
        self.gravity = gravity or {}
        self.gravity_gain = gravity_gain
        self._grav_slots = {}                # side -> [7 indices into names]
        for side in self.gravity:
            self._grav_slots[side] = [
                names.index(f"{side}_{seg}") for seg in ARM_SEGS]
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

    def joint_q(self, name):
        """Latest measured position of one joint (rad) from lowstate."""
        if self.dry:
            return 0.0
        return float(self._state.motor_state[SDK_INDEX[name]].q)

    def _gravity_tau(self, pose):
        """{name: tau_ff} for the arm joints, scaled by weight * gain."""
        out = {}
        if not self.gravity:
            return out
        for side, gff in self.gravity.items():
            slots = self._grav_slots[side]
            q = [pose[i] for i in slots]
            taus = gff.tau(q)
            for i, seg in enumerate(ARM_SEGS):
                out[f"{side}_{seg}"] = taus[i] * self.weight * self.gravity_gain
        return out

    def publish(self, pose):
        self.pose = list(pose)
        if self.dry:
            return
        gtau = self._gravity_tau(pose)
        for n, q in zip(self.names, pose):
            mc = self._msg.motor_cmd[SDK_INDEX[n]]
            kp, kd = self.gain_override.get(n) or gains_for(n)
            mc.mode, mc.q, mc.dq = 1, float(q), 0.0
            mc.tau = float(gtau.get(n, 0.0))
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
    p.add_argument("--gravity-ff", action="store_true",
                   help="gravity feedforward torque from the URDF inertials "
                        "(cancels the ~10 deg shoulder sag; sign verified on "
                        "the M1 T1 pilot). Torques are capped and scaled by "
                        "the overlay weight so they fade with it.")
    p.add_argument("--gravity-gain", type=float, default=FROZEN_GRAVITY_GAIN,
                   help=f"scale on the gravity torque (frozen default "
                        f"{FROZEN_GRAVITY_GAIN}: best of the M1 T1 on-robot "
                        f"sweep, mean EE 2.59 cm; 1.0 = pure physics, "
                        f"under-cancels)")
    p.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"),
                   help="URDF for gravity feedforward + the safe-start check")
    p.add_argument("--wrist-behind-m", type=float, default=0.0,
                   help="trim leading/trailing frames whose wrist FK is behind "
                        "this torso-frame x (m); default 0.0 = do not start/"
                        "end with a wrist behind the torso plane (those are "
                        "clip-end retargeting artifacts that swing the arm "
                        "behind the robot during go-to-start). Only contiguous "
                        "leading/trailing frames are trimmed, never interior "
                        "motion; pass a large negative value to disable")
    p.add_argument("--task-signal", choices=["x", "z"], default="x",
                   help="wrist statistic for the prep/tail trim: z = height "
                        "(overhead work, M1), x = forward reach (M2/M3)")
    p.add_argument("--task-thresh", type=float, default=0.0,
                   help="prep/tail trim: play only the first..last sustained "
                        "crossing of the task signal (m). 0 disables. "
                        "Calibrated per motion: M1 z>0.40, M2 x>0.30, "
                        "M3 x>0.22")
    p.add_argument("--task-sustain", type=float, default=1.5,
                   help="rolling-median window (s) for the task signal")
    p.add_argument("--trim-margin", type=float, default=1.0,
                   help="seconds kept before/after the detected task window")
    p.add_argument("--clip", metavar="A:B",
                   help="manual play window in trajectory seconds, e.g. "
                        "6.0:36.5; overrides the automatic task trim")
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

    # gravity feedforward: build (runs the self-check), preflight peak torques
    gravity = {}
    if args.gravity_ff:
        if not os.path.isfile(args.urdf):
            sys.exit(f"--gravity-ff needs --urdf; not found: {args.urdf}")
        for side in ("left", "right"):
            if all(f"{side}_{seg}" in names for seg in ARM_SEGS):
                gravity[side] = GravityFF(args.urdf, side)
        # peak commanded gravity torque over the trajectory (weight=gain=1)
        peak_tau = {}
        for row in Q:
            for side, gff in gravity.items():
                q = [row[names.index(f"{side}_{seg}")] for seg in ARM_SEGS]
                for seg, tv in zip(ARM_SEGS, gff.tau(q)):
                    peak_tau[f"{side}_{seg}"] = max(
                        peak_tau.get(f"{side}_{seg}", 0.0), abs(tv))
        top = sorted(peak_tau.items(), key=lambda kv: -kv[1])[:4]
        print(f"[replay] gravity-ff ON gain={args.gravity_gain:g}; peak |tau|: "
              + "  ".join(f"{k}={v * args.gravity_gain:.1f}Nm" for k, v in top)
              + "  (capped, scaled by overlay weight)")

    # trims (intersected; interior frames are never touched):
    #   1. safe-start: no wrist behind the torso at the window ends
    #   2. task window: cut the human's prep/idle at the clip ends
    #   3. --clip: manual override of the task window
    i0, i1 = 0, n - 1
    if os.path.isfile(args.urdf):     # large-negative --wrist-behind-m disables
        wp = wrist_paths(names, Q, args.urdf)
        i0, i1 = safe_window(wp, args.wrist_behind_m)
        if i0 > 0 or i1 < n - 1:
            print(f"[replay] safe-start trim: skipping {i0} leading + "
                  f"{n - 1 - i1} trailing frame(s) with a wrist behind the "
                  f"torso (artifact)")
        if args.clip:
            a, b = (float(v) for v in args.clip.split(":"))
            j0 = min(range(n), key=lambda i: abs(t[i] - t[0] - a))
            j1 = min(range(n), key=lambda i: abs(t[i] - t[0] - b))
            i0, i1 = max(i0, j0), min(i1, j1)
            print(f"[replay] manual clip {a:.1f}:{b:.1f}s applied")
        elif args.task_thresh > 0:
            tw = task_window(wp, t, args.task_signal, args.task_thresh,
                             args.task_sustain, args.trim_margin)
            if tw is None:
                sys.exit(f"[replay] task trim: signal {args.task_signal} never "
                         f"crosses {args.task_thresh}; wrong threshold/motion?")
            i0, i1 = max(i0, tw[0]), min(i1, tw[1])
        if i0 > 0 or i1 < n - 1:
            print(f"[replay] play window: {t[i0] - t[0]:.1f}..{t[i1] - t[0]:.1f}s "
                  f"of {dur:.1f}s (cut {t[i0] - t[0]:.1f}s prep, "
                  f"{t[-1] - t[i1]:.1f}s tail)")
            play_dur = (t[i1] - t[i0]) / args.speed_scale

    if args.dry_run:
        print("[replay] dry-run OK (no DDS, nothing published)")
        return

    print("=" * 66)
    print("H1-2 ARM TRAJECTORY REPLAY (rt/arm_sdk overlay)")
    print("Robot must be STANDING IN BALANCE (FSM 204), workspace clear,")
    print("second person on the e-stop. Loco is never commanded here.")
    print("=" * 66)

    sdk = ArmSdk(names, args.interface, dry=False,
                 gravity=gravity, gravity_gain=args.gravity_gain)
    sdk.pose = sdk.snapshot()
    rest = list(sdk.pose)
    print(f"[replay] rest pose captured: {[round(v, 2) for v in rest]}")

    try:
        input("ENTER to fade the arm overlay in (Ctrl+C aborts)... ")
        sdk.ramp_weight(1.0, 2.0)
        print("[replay] moving to trajectory start (3 s)...")
        sdk.ramp_pose(Q[i0], 3.0)
        input(f"ENTER to play ({play_dur:.1f} s at speed "
              f"{args.speed_scale:g})... ")
        t_start = time.time()
        i = i0 + 1
        max_step = args.max_vel * ArmSdk.CTRL_DT
        while i <= i1:
            now = (time.time() - t_start) * args.speed_scale + t[i0]
            while i <= i1 and t[i] < now:
                i += 1
            if i > i1:
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
