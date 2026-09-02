#!/usr/bin/env python3
"""H1-2 raise BOTH arms for 10 s when YOLO detects a person.

Sequence per trigger: wait 3 s at REST, ramp the arms up slowly (to a pose
60 deg lower than the original raise, so it hangs down rather than reaching
forward), hold 10 s, ramp back to REST, then cooldown.

Reads person-detection events from a UNIX socket (default
~/mj_ws/h1-2_sensors/yolo_ws/yolo_bridge.sock) fed by yolo_bridge.py. The bridge is a separate
ROS 2 process running in system Python 3.10 because rclpy doesn't load
under the conda env's Python 3.11 — split here to keep this script
ROS-free and runnable from the conda env.

Bring-up order:
    1) source /opt/ros/humble/setup.bash         (in a non-conda shell)
       python3 yolo_bridge.py
    2) conda activate rical_unitree              (in this shell)
       python3 wave_on_person.py enp128s31f6

Notes
-----
* H1-2 has NO built-in wave action (the G1ArmActionClient "face wave"/
  "high wave" entries are G1-only). We raise the arms through the
  rt/arm_sdk overlay, same pattern as pick_cone_palms.py.
* A person trigger raises BOTH arms; a traffic-cone trigger still raises
  the right arm only, so the two events stay visually distinguishable.
  Both use the same raise height and the same 3 s delay / 10 s hold timing.
* The script does NOT call LocoClient.Start() — it assumes the operator
  already brought the robot to standing balance (FSM 204). It only takes
  over the arm overlay and leaves loco untouched.
* arm_sdk weight is faded 0->1 once on init and held at 1 for the
  session. Gesture is a joint-space ramp on top of that.
"""

import argparse
import json
import math
import socket
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


SOCKET_PATH = "/home/mchang344/mj_ws/h1-2_sensors/yolo_ws/yolo_bridge.sock"


# ── H1-2 joint indices (same as pick_cone_palms.py) ─────────────────────────
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


# ── Poses (15-element vectors, radians) ─────────────────────────────────────
# Indices: [LShP, LShR, LShY, LElb, LWrR, LWrP, LWrY,
#           RShP, RShR, RShY, RElb, RWrR, RWrP, RWrY,
#           Waist]
#
# POSE_REST is captured from the live robot at init (snapshot of the loco
# controller's current arm joint angles), so the gesture always returns
# the arms to wherever they were when the script started.
#
# Both gestures are expressed as OVERRIDES on top of pose_rest rather than as
# absolute pose vectors, so any slot left un-overridden keeps whatever the loco
# controller had at startup. Applied in WaveOnPerson._build_target.
#
# Arm-vector index reference (15 elements total):
#   0..6   : left  arm  [LShP, LShR, LShY, LElb, LWrR, LWrP, LWrY]
#   7..13  : right arm  [RShP, RShR, RShY, RElb, RWrR, RWrP, RWrY]
#   14     : waist yaw
#
# On THIS unit (per the pick_cone_palms.py calibration notes):
#   * Shoulder pitch: 0 = arm down, -1.57 is roughly horizontal forward, more
#     negative = above horizontal. Same sign on both arms (URDF axis 0 1 0).
#   * Shoulder roll is MIRRORED between the arms: outward is negative on the
#     right (limits -3.4..0.38) and positive on the left (limits -0.38..3.4).
#   * Elbow: 0 is roughly 90 deg of flex, positive = straighter.

# Raise height. The original gesture used -1.4 rad (-80 deg, just under
# horizontal-forward), which reaches the arm well out in front. Shoulder pitch
# is 0 at arms-down and more negative further up, so ADDING a positive angle
# LOWERS the arms and pulls the hand back toward the body. Lowered 60 deg here
# so the raised arm hangs down-and-slightly-forward rather than reaching ahead.
RAISE_PITCH_BASE_RAD = -1.4
RAISE_LOWER_DEG      = 50.0
RAISED_SHOULDER_PITCH = RAISE_PITCH_BASE_RAD + math.radians(RAISE_LOWER_DEG)
# = -0.353 rad = -20.2 deg, well inside the +-[-3.14, +1.57] joint limit.

# Cone trigger: right arm only. Left-arm slots are not overridden, so the left
# arm stays wherever the loco controller had it.
RIGHT_ARM_RAISED_OVERRIDES = {
    7:  RAISED_SHOULDER_PITCH,   # RShoulderPitch, raised forward
    8:  -0.2,   # RShoulderRoll, small outward abduction so hand isn't in face
    9:   0.0,   # RShoulderYaw
    10:  0.5,   # RElbow, slight bend
    11:  0.0,   # RWristRoll
    12:  0.0,   # RWristPitch
    13:  0.0,   # RWristYaw
}

# Person trigger: BOTH arms raise. The left arm mirrors the right values, with
# the roll sign flipped per the mirroring note above. Every angle is inside the
# URDF limits for its joint.
BOTH_ARMS_RAISED_OVERRIDES = {
    0:  RAISED_SHOULDER_PITCH,   # LShoulderPitch
    1:   0.2,   # LShoulderRoll, outward on the left is positive
    2:   0.0,   # LShoulderYaw
    3:   0.5,   # LElbow
    4:   0.0,   # LWristRoll
    5:   0.0,   # LWristPitch
    6:   0.0,   # LWristYaw
    7:  RAISED_SHOULDER_PITCH,   # RShoulderPitch
    8:  -0.2,   # RShoulderRoll
    9:   0.0,   # RShoulderYaw
    10:  0.5,   # RElbow
    11:  0.0,   # RWristRoll
    12:  0.0,   # RWristPitch
    13:  0.0,   # RWristYaw
}


# ── Behavior tuning ─────────────────────────────────────────────────────────
MIN_CONF       = 0.5      # YOLO node already filters at 0.5; redundant guard
COOLDOWN_S     = 30.0      # min seconds between gesture starts
TRIGGER_DELAY_S = 3.0     # wait at REST this long after a trigger, then raise
RAISE_TIMEOUT_S = 8.0     # safety cap on the ramp; normally we leave RAISING
                          # as soon as the arms ARRIVE (see RAISE_TOL_RAD), so
                          # the hold is a full HOLD_TIME_S at the pose. Raised
                          # from 3.0 s so the slower ramp is not clipped.
RAISE_TOL_RAD  = 0.02     # per-joint arrival tolerance
HOLD_TIME_S    = 10.0     # hold raised, measured from arrival
LOWER_TIME_S   = 4.0      # ramp back to rest (slow, matched to MAX_VEL)
# The ramp is rate-limited at ArmSdkController.MAX_VEL (0.2 rad/s, deliberately
# gentle so raising the arms does not shove the CoM forward and make loco step),
# so a ~0.35 rad raise from arms-down takes ~1.8 s. One full gesture is about
# 3.0 + 1.8 + 10.0 + 4.0 = 18.8 s, then COOLDOWN_S before the next trigger is
# accepted. (LOWER_TIME_S keeps its 4.0 s margin though the lower now needs
# ~1.8 s.)


# ── arm_sdk controller (50 Hz pump) ─────────────────────────────────────────
class ArmSdkController:
    CTRL_DT     = 0.02   # 50 Hz
    MAX_VEL     = 0.2    # rad/s — per-step joint delta clamp (slow + gentle so
                         # raising the arms does not push the robot forward)
    WEIGHT_RATE = 0.5    # 1/s — fade rate for arm_sdk blend weight

    def __init__(self):
        self.msg           = unitree_hg_msg_dds__LowCmd_()
        self.weight        = 0.0
        self.current_pose  = [0.0] * len(ARM_JOINTS)
        self.pose_rest     = None   # captured from live robot in init()
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

        # Snapshot the loco controller's current arm pose. This becomes
        # both our starting current_pose AND the REST pose the gesture
        # returns to after each raise.
        snap = [self.lowstate.motor_state[j].q for j in ARM_JOINTS]
        self.current_pose = list(snap)
        self.pose_rest    = list(snap)
        print(f"[arm_sdk] lowstate received, REST pose captured: "
              f"{[round(q, 2) for q in snap]}")

    def _on_lowstate(self, msg: LowState_):
        self.lowstate   = msg
        self._got_state = True

    def publish_toward(self, target: list):
        """Step current_pose toward target (rate-limited) and publish."""
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

    def at_pose(self, target: list, tol: float) -> bool:
        """True once every commanded joint is within tol of target. Uses the
        commanded pose, not the measured one: this gates the hold timer on the
        ramp finishing, not on the arm's tracking error."""
        return all(abs(target[i] - self.current_pose[i]) <= tol
                   for i in range(len(ARM_JOINTS)))

    def ramp_weight(self, target: float, duration: float):
        steps = max(1, int(duration / self.CTRL_DT))
        delta = (target - self.weight) / steps
        for _ in range(steps):
            self.weight = max(0.0, min(1.0, self.weight + delta))
            self.publish_toward(self.current_pose)
            time.sleep(self.CTRL_DT)
        self.weight = target


# ── Detection source ────────────────────────────────────────────────────────
# Classes we react to. The state machine reads cone first, then person, so
# cone wins when both arrive in the same window.
CONE_CLASS   = "traffic cone"
PERSON_CLASS = "person"
TRIGGER_CLASSES = (CONE_CLASS, PERSON_CLASS)


class PersonDetector:
    """Reads person/cone events from yolo_bridge over a UNIX socket.

    Maintains a separate ``*_seen`` flag per class. Reconnects automatically
    if the bridge restarts.
    """

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.person_seen = False
        self.cone_seen   = False
        self._stop       = threading.Event()
        self._lock       = threading.Lock()
        self._thread     = threading.Thread(
            target=self._reader_loop, name="yolo_socket_reader", daemon=True
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _connect(self) -> socket.socket | None:
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
                    print("[detector] bridge closed connection, reconnecting...")
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
        cls = evt.get("class")
        if cls not in TRIGGER_CLASSES:
            return
        if float(evt.get("conf", 0)) < MIN_CONF:
            return
        with self._lock:
            if cls == CONE_CLASS:
                self.cone_seen = True
            elif cls == PERSON_CLASS:
                self.person_seen = True

    def take_trigger(self) -> str | None:
        """Atomically read and clear the highest-priority pending trigger.

        Returns the class name ("traffic cone" / "person") of the trigger
        that fired, or None if nothing's pending. Cone has priority.
        """
        with self._lock:
            if self.cone_seen:
                self.cone_seen = False
                self.person_seen = False   # consume both — cone wins
                return CONE_CLASS
            if self.person_seen:
                self.person_seen = False
                return PERSON_CLASS
        return None

    def clear_pending(self):
        with self._lock:
            self.cone_seen = False
            self.person_seen = False


# ── State machine: idle → waiting → raising → holding → lowering → cooldown ─
class WaveOnPerson:
    IDLE, WAITING, RAISING, HOLDING, LOWERING, COOLDOWN = range(6)

    def __init__(self):
        self.arm   = ArmSdkController()
        self.node  = None
        self.state = self.IDLE
        self.state_start = 0.0
        self.last_trigger_end = -1e9  # so first trigger isn't blocked
        self.active_trigger = None    # class name of the in-flight trigger

    def init(self):
        self.arm.init()
        # Engage arm overlay smoothly; hold weight at 1 for the rest of the run.
        print("[arm_sdk] weight 0->1 (1.5 s)")
        self.arm.ramp_weight(1.0, duration=1.5)
        # Settle to REST pose before listening.
        print("[arm_sdk] go to REST (1.5 s)")
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

    def _build_target(self, overrides: dict) -> list:
        """Start from the captured rest pose and apply a gesture's overrides."""
        target = list(self.arm.pose_rest)
        for idx, val in overrides.items():
            target[idx] = val
        return target

    def _gesture_target(self) -> list:
        """Pose to drive toward during RAISING/HOLDING, picked by trigger."""
        if self.active_trigger == CONE_CLASS:
            return self._build_target(RIGHT_ARM_RAISED_OVERRIDES)
        return self._build_target(BOTH_ARMS_RAISED_OVERRIDES)   # PERSON_CLASS

    def run(self):
        print("[main] running. Ctrl+C to exit.")
        try:
            while True:
                now = time.monotonic()
                rest = self.arm.pose_rest
                target = rest

                if self.state == self.IDLE:
                    if (now - self.last_trigger_end) >= COOLDOWN_S:
                        # Cone has priority over person — take_trigger() returns
                        # the cone class if both are pending.
                        trig = self.node.take_trigger()
                        if trig is not None:
                            self.active_trigger = trig
                            print(f"[trigger] {trig} detected -> WAITING "
                                  f"{TRIGGER_DELAY_S:.1f}s before raising")
                            self._enter(self.WAITING)
                    else:
                        # Still cooling down — drop any pending flags so they
                        # don't immediately re-trigger when cooldown expires.
                        self.node.clear_pending()
                        target = rest

                if self.state == self.WAITING:
                    # Hold REST for the delay. Triggers arriving now are not
                    # consumed here: IDLE is the only state that takes them,
                    # so they just wait and are cleared during cooldown.
                    target = rest
                    if self._elapsed() >= TRIGGER_DELAY_S:
                        print(f"[trigger] {self.active_trigger} -> RAISING")
                        self._enter(self.RAISING)

                if self.state == self.RAISING:
                    target = self._gesture_target()
                    arrived = self.arm.at_pose(target, RAISE_TOL_RAD)
                    if arrived or self._elapsed() >= RAISE_TIMEOUT_S:
                        if not arrived:
                            print(f"[trigger] raise hit the {RAISE_TIMEOUT_S:.1f}s "
                                  f"cap before arriving -> HOLDING anyway")
                        self._enter(self.HOLDING)

                elif self.state == self.HOLDING:
                    target = self._gesture_target()
                    if self._elapsed() >= HOLD_TIME_S:
                        print(f"[trigger] {self.active_trigger} HOLD done -> LOWERING")
                        self._enter(self.LOWERING)

                elif self.state == self.LOWERING:
                    target = rest
                    if self._elapsed() >= LOWER_TIME_S:
                        self.last_trigger_end = now
                        print(f"[trigger] {self.active_trigger} done. "
                              f"cooldown {COOLDOWN_S:.1f}s")
                        self.active_trigger = None
                        self._enter(self.IDLE)

                self.arm.publish_toward(target)
                time.sleep(self.arm.CTRL_DT)
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n[shutdown] returning to REST and fading arm_sdk weight to 0")
        try:
            if self.arm.pose_rest is not None:
                # 4.0 s, not 1.5: the slow MAX_VEL needs the time to fully
                # lower the arms before the weight fades and loco takes over.
                self._hold_pose_for(self.arm.pose_rest, 4.0)
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
    print("H1-2 RAISE-ARMS ON YOLO TRIGGER")
    print(f" * Source:      {SOCKET_PATH} (events forwarded by yolo_bridge.py)")
    print(f" * Triggers:    {CONE_CLASS!r} (priority), {PERSON_CLASS!r}; "
          f"score >= {MIN_CONF:.2f}")
    print(" * Gesture:     person -> BOTH arms, cone -> right arm only")
    print(" * Timing:      wait {:.1f}s after trigger, raise (rate-limited, "
          "{:.1f}s cap), hold {:.1f}s, lower {:.1f}s".format(
              TRIGGER_DELAY_S, RAISE_TIMEOUT_S, HOLD_TIME_S, LOWER_TIME_S))
    print(" * Raise pitch: {:+.3f} rad ({:+.1f} deg), {:.0f} deg lower than "
          "the original".format(RAISED_SHOULDER_PITCH,
                                math.degrees(RAISED_SHOULDER_PITCH),
                                RAISE_LOWER_DEG))
    print(" * Cooldown:    {}s between triggers".format(COOLDOWN_S))
    print(" * Assumes the robot is ALREADY in balance stand (FSM 204).")
    print("=" * 70)
    input("Press Enter to engage arm_sdk (Ctrl+C aborts and returns to REST)...")

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    app = WaveOnPerson()
    app.init()
    app.run()
