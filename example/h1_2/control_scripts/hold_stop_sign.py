#!/usr/bin/env python3
"""H1-2 hold a (pretend) stop sign with the right arm when YOLO sees a person.

Reads person events from the same ~/mj_ws/h1-2_sensors/yolo_ws/yolo_bridge.sock
that wave_on_person.py uses (fed by yolo_bridge.py). On a person detection above MIN_CONF, the right
arm raises to a chest-level "presenting" pose, holds for 5 seconds as if
showing a handheld STOP sign, then lowers back to the captured rest pose.
The left arm stays at rest throughout.

Bring-up order:
    1) source /opt/ros/humble/setup.bash         (in a non-conda shell)
       python3 yolo_bridge.py
    2) conda activate rical_unitree              (in this shell)
       python3 hold_stop_sign.py enp128s31f6

Notes
-----
* Assumes the robot is ALREADY in balance stand (FSM 204). This script
  only takes over the arm overlay via rt/arm_sdk and leaves loco alone.
* arm_sdk weight is faded 0->1 once on init and held at 1 for the
  session. Gesture is a joint-space ramp on top of that.
* Cone detections, if any, are ignored — this script only cares about
  the "person" class.
"""

import argparse
import json
import socket
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


SOCKET_PATH    = "/home/mchang344/mj_ws/h1-2_sensors/yolo_ws/yolo_bridge.sock"
TRIGGER_CLASS  = "person"
MIN_CONF       = 0.5

# ── Gesture timings (seconds) ───────────────────────────────────────────────
RAISE_TIME_S   = 2.0
HOLD_TIME_S    = 5.0    # per spec: hold "stop sign" 5 s
LOWER_TIME_S   = 2.0
COOLDOWN_S     = 5.0


# ── H1-2 joint indices ──────────────────────────────────────────────────────
class J:
    WaistYaw       = 12
    LShoulderPitch = 13
    LShoulderRoll  = 14
    LShoulderYaw   = 15
    LElbow         = 16
    LWristRoll     = 17
    LWristPitch    = 18
    LWristYaw      = 19
    RShoulderPitch = 20
    RShoulderRoll  = 21
    RShoulderYaw   = 22
    RElbow         = 23
    RWristRoll     = 24
    RWristPitch    = 25
    RWristYaw      = 26
    Weight         = 27   # arm_sdk blend (0..1)


# arm_sdk control vector: [left arm 7] + [right arm 7] + [waist 1]
ARM_JOINTS = [
    J.LShoulderPitch, J.LShoulderRoll, J.LShoulderYaw, J.LElbow,
    J.LWristRoll, J.LWristPitch, J.LWristYaw,
    J.RShoulderPitch, J.RShoulderRoll, J.RShoulderYaw, J.RElbow,
    J.RWristRoll, J.RWristPitch, J.RWristYaw,
    J.WaistYaw,
]

ARM_KP = [ 80,  80, 60, 60, 30, 30, 30,
           80,  80, 60, 60, 30, 30, 30,
          150]
ARM_KD = [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0,
          2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0,
          2.0]


# ── Pose: right arm holding a stop sign at chest level ──────────────────────
# Only the right-arm slots are overridden — left-arm slots in the target
# vector pull from the captured rest pose, so just the right arm moves.
#
# Calibration (this unit, per pick_cone_palms.py notes):
#   * Shoulder pitch: 0 = arm down,  -1.57 ≈ horizontal forward, more
#     negative = above horizontal.
#   * Shoulder roll (right): negative = abducted outward to the right.
#   * Elbow: 0 ≈ 90° flex, positive = straighter. +1.4 ≈ fully extended.
#
# Target geometry: hand at roughly chest height, arm angled slightly down
# from horizontal so the sign would face an oncoming viewer. Forearm mostly
# extended (slight bend) so the "sign" is held out from the body.
#
# Arm-vector index reference:
#   0..6   : left  arm  [LShP, LShR, LShY, LElb, LWrR, LWrP, LWrY]
#   7..13  : right arm  [RShP, RShR, RShY, RElb, RWrR, RWrP, RWrY]
#   14     : waist yaw
STOP_SIGN_RIGHT_ARM_OVERRIDES = {
    7:  -1.30,   # RShoulderPitch — arm forward, slightly below horizontal
    8:  -0.10,   # RShoulderRoll  — small outward so hand isn't touching body
    9:   0.00,   # RShoulderYaw
    10:  0.80,   # RElbow         — soft bend; sign held forward from body
    11:  0.00,   # RWristRoll
    12:  0.00,   # RWristPitch
    13:  0.00,   # RWristYaw
}


# ── arm_sdk controller (50 Hz pump) ─────────────────────────────────────────
class ArmSdkController:
    CTRL_DT     = 0.02   # 50 Hz
    MAX_VEL     = 0.8    # rad/s — per-step joint delta clamp
    WEIGHT_RATE = 0.5    # 1/s — fade rate for arm_sdk blend weight

    def __init__(self):
        self.msg           = unitree_hg_msg_dds__LowCmd_()
        self.weight        = 0.0
        self.current_pose  = [0.0] * len(ARM_JOINTS)
        self.pose_rest     = None
        self.lowstate      = None
        self._got_state    = False

    def init(self):
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._on_lowstate, 10)
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()

        print("[arm_sdk] waiting for lowstate...")
        while not self._got_state:
            time.sleep(0.1)

        snap = [self.lowstate.motor_state[j].q for j in ARM_JOINTS]
        self.current_pose = list(snap)
        self.pose_rest    = list(snap)
        print(f"[arm_sdk] lowstate received, REST pose captured: "
              f"{[round(q, 2) for q in snap]}")

    def _on_lowstate(self, msg: LowState_):
        self.lowstate   = msg
        self._got_state = True

    def publish_toward(self, target: list):
        max_delta = self.MAX_VEL * self.CTRL_DT
        for i in range(len(ARM_JOINTS)):
            err = target[i] - self.current_pose[i]
            err = max(-max_delta, min(max_delta, err))
            self.current_pose[i] += err

        for i, j in enumerate(ARM_JOINTS):
            mc = self.msg.motor_cmd[j]
            mc.mode = 1
            mc.q    = self.current_pose[i]
            mc.dq   = 0.0
            mc.tau  = 0.0
            mc.kp   = ARM_KP[i]
            mc.kd   = ARM_KD[i]

        self.msg.motor_cmd[J.Weight].q = self.weight
        self.pub.Write(self.msg)

    def ramp_weight(self, target: float, duration: float):
        steps = max(1, int(duration / self.CTRL_DT))
        delta = (target - self.weight) / steps
        for _ in range(steps):
            self.weight = max(0.0, min(1.0, self.weight + delta))
            self.publish_toward(self.current_pose)
            time.sleep(self.CTRL_DT)
        self.weight = target


# ── Person-detection source (UNIX socket reader) ────────────────────────────
class PersonDetector:
    """Reads person events from yolo_bridge over a UNIX socket."""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.person_seen = False
        self._stop       = threading.Event()
        self._lock       = threading.Lock()
        self._thread     = threading.Thread(
            target=self._reader_loop, name="yolo_socket_reader", daemon=True
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(self.socket_path)
            print(f"[detector] connected to {self.socket_path}")
            return s
        except (FileNotFoundError, ConnectionRefusedError):
            return None

    def _reader_loop(self):
        buf = b""
        sock = None
        warned = False
        while not self._stop.is_set():
            if sock is None:
                sock = self._connect()
                if sock is None:
                    if not warned:
                        print(f"[detector] waiting for bridge socket at "
                              f"{self.socket_path} (start yolo_bridge.py "
                              f"outside conda)...")
                        warned = True
                    time.sleep(1.0)
                    continue
                warned = False
                sock.settimeout(0.5)
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    print("[detector] bridge closed, reconnecting...")
                    sock.close()
                    sock = None
                    buf = b""
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._handle_line(line)
            except socket.timeout:
                continue
            except OSError:
                if sock is not None:
                    sock.close()
                sock = None
                buf = b""

        if sock is not None:
            sock.close()

    def _handle_line(self, line: bytes):
        try:
            evt = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if evt.get("class") != TRIGGER_CLASS:
            return
        if float(evt.get("conf", 0)) < MIN_CONF:
            return
        with self._lock:
            self.person_seen = True

    def take_trigger(self) -> bool:
        with self._lock:
            if self.person_seen:
                self.person_seen = False
                return True
        return False

    def clear_pending(self):
        with self._lock:
            self.person_seen = False


# ── State machine: idle → raising → holding → lowering ──────────────────────
class HoldStopSign:
    IDLE, RAISING, HOLDING, LOWERING = range(4)

    def __init__(self):
        self.arm   = ArmSdkController()
        self.node  = None
        self.state = self.IDLE
        self.state_start      = 0.0
        self.last_trigger_end = -1e9

    def init(self):
        self.arm.init()
        print("[arm_sdk] weight 0->1 (1.5 s)")
        self.arm.ramp_weight(1.0, duration=1.5)
        print("[arm_sdk] settle at REST (1.5 s)")
        self._hold_pose_for(self.arm.pose_rest, 1.5)

        self.node = PersonDetector()
        self.node.start()

    def _hold_pose_for(self, target, duration):
        steps = max(1, int(duration / self.arm.CTRL_DT))
        for _ in range(steps):
            self.arm.publish_toward(target)
            time.sleep(self.arm.CTRL_DT)

    def _enter(self, state):
        self.state = state
        self.state_start = time.monotonic()

    def _elapsed(self):
        return time.monotonic() - self.state_start

    def _build_stop_sign_target(self) -> list:
        """Right-arm-only pose: start from captured rest, override right slots."""
        target = list(self.arm.pose_rest)
        for idx, val in STOP_SIGN_RIGHT_ARM_OVERRIDES.items():
            target[idx] = val
        return target

    def run(self):
        print("[main] running. Ctrl+C to exit.")
        try:
            while True:
                now  = time.monotonic()
                rest = self.arm.pose_rest
                target = rest

                if self.state == self.IDLE:
                    if (now - self.last_trigger_end) >= COOLDOWN_S:
                        if self.node.take_trigger():
                            print("[trigger] person detected -> RAISING")
                            self._enter(self.RAISING)
                    else:
                        # Drop any pending flag so we don't immediately
                        # re-fire when cooldown ends.
                        self.node.clear_pending()
                        target = rest

                if self.state == self.RAISING:
                    target = self._build_stop_sign_target()
                    if self._elapsed() >= RAISE_TIME_S:
                        self._enter(self.HOLDING)

                elif self.state == self.HOLDING:
                    target = self._build_stop_sign_target()
                    if self._elapsed() >= HOLD_TIME_S:
                        print(f"[trigger] HOLD {HOLD_TIME_S:.1f}s done "
                              f"-> LOWERING")
                        self._enter(self.LOWERING)

                elif self.state == self.LOWERING:
                    target = rest
                    if self._elapsed() >= LOWER_TIME_S:
                        self.last_trigger_end = now
                        print(f"[trigger] done. cooldown {COOLDOWN_S:.1f}s")
                        self._enter(self.IDLE)

                self.arm.publish_toward(target)
                time.sleep(self.arm.CTRL_DT)
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n[shutdown] returning to REST and fading arm_sdk weight to 0")
        try:
            if self.arm.pose_rest is not None:
                self._hold_pose_for(self.arm.pose_rest, 1.5)
            self.arm.ramp_weight(0.0, duration=1.5)
        except Exception as e:
            print(f"shutdown error: {e}")
        if self.node is not None:
            self.node.stop()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("interface", nargs="?",
                   help="Robot network interface, e.g. enp128s31f6.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 70)
    print("H1-2 HOLD-STOP-SIGN-ON-PERSON")
    print(f" * Source:      {SOCKET_PATH} (events forwarded by yolo_bridge.py)")
    print(f" * Trigger:     class_id == '{TRIGGER_CLASS}', score >= {MIN_CONF:.2f}")
    print(f" * Gesture:     right arm to chest  hold {HOLD_TIME_S:.0f}s  lower")
    print(f" * Cooldown:    {COOLDOWN_S:.1f}s between triggers")
    print(" * Assumes the robot is ALREADY in balance stand (FSM 204).")
    print(" * NOTE: yolo_bridge.py TARGET_CLASSES must include 'person'.")
    print("=" * 70)
    input("Press Enter to engage arm_sdk (Ctrl+C aborts and returns to REST)...")

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    app = HoldStopSign()
    app.init()
    app.run()
