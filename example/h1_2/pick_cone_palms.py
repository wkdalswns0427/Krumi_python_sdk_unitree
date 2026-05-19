#!/usr/bin/env python3
"""H1-2 cone pickup using two-palm squeeze (no fingers).

Ported from the verified C++ pattern in:
    Krumi_cpp_sdk_unitree/hit_h1-2/high_level/h1_2_hl_pose3.cpp

Architecture
------------
* Legs / balance: Unitree LocoClient owns FSM 204 (main locomotion). We
  call SetStandHeight(h) directly with numeric meters for SMOOTH knee
  motion. LowStand()/HighStand() are instant FSM transitions and produce
  a sharp drop — avoid for this task.
* Arms + waist: 15 joints (14 arm + 1 waist) overlaid through rt/arm_sdk.
  Slot 27 carries a 0..1 blend weight: 0 = pure loco, 1 = pure arm_sdk.
* Hands: NOT USED. The inspire hand is sidelined; we press the cone
  between the two palms instead.
* Control rate: 50 Hz (20 ms tick).

Hardware-verified calibrations on THIS unit
-------------------------------------------
* Elbow zero offset: q=0 → physical ~90° flexed. POSITIVE values straighten.
    +1.4 ≈ straight   +1.0 ≈ slight bend   +0.8 ≈ soft bend
* Shoulder pitch: 0 = arm down, negative raises forward. -1.57 ≈ horizontal.
* Shoulder roll: +L abducts left outward, -R abducts right outward.
* Forward-lean: arms at -1.57 ShP shift COG → loco steps forward.
  Mitigate with ShP ≈ -1.3, ramp ≥5 s, elbows bent so forearms fold back.

Sequence
--------
1. Capture current arm pose, fade arm_sdk weight 0→1 (3 s)
2. Move to "ready" pose (shoulders neutral + outward roll, elbows extended)
3. Kneel: SetStandHeight from default down to default-0.15 m (5 s)
4. Reach arms forward, palms facing each other, opened wider than cone
5. Squeeze inward to pinch cone between palms
6. Lift: SetStandHeight back up to default (5 s) — cone comes with us
7. Hold lifted (room for future Move() / placement code)
8. Reverse: lower, release squeeze, return to ready, fade weight 1→0

TODO (out of scope for this commit)
-----------------------------------
* RealSense D435i depth → cone center+diameter → adjust SQUEEZE pose
* LocoClient.Move() to transport
* Torque-based contact detection from rt/lowstate motor_state[i].tau_est
"""

import argparse
import json
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
from unitree_sdk2py.h1.loco.h1_loco_api import ROBOT_API_ID_LOCO_GET_STAND_HEIGHT
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


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

# Shoulder gains lowered from the C++ baseline (120) to reduce sustained
# holding current — left arm tripped its over-current latch after a single
# full-sequence run with the higher gains. Bring these back up if motion
# becomes too compliant, but watch arm temperature reports.
ARM_KP = [ 80,  80, 60, 60, 30, 30, 30,
           80,  80, 60, 60, 30, 30, 30,
          150]
ARM_KD = [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0,
          2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0,
          2.0]


# ── Pose library (15-element vectors, radians) ──────────────────────────────
# Indices: [LShP, LShR, LShY, LElb, LWrR, LWrP, LWrY,
#           RShP, RShR, RShY, RElb, RWrR, RWrP, RWrY,
#           Waist]
#
# SHAKEDOWN MODE: magnitudes are intentionally small so a wrong sign
# produces a gentle twitch instead of a swing into the body. Bump SCALE
# up to 1.0 in steps after you've confirmed every direction is correct.

SCALE = 0.3    # 0.3 = ~30% of intended deflection. Raise to 1.0 when verified.

POSE_READY = [
    0.0,  0.10, 0.0, 0.30, 0.0, 0.0, 0.0,
    0.0, -0.10, 0.0, 0.30, 0.0, 0.0, 0.0,
    0.0,
]

# Both arms forward + slightly out. Wrist roll = 0 — at non-zero wrist roll
# the palms rotate upward on this unit, so we leave the wrist alone.
POSE_REACH = [
    -1.30 * SCALE,  0.45 * SCALE, 0.0, 0.85, 0.0, 0.0, 0.0,
    -1.30 * SCALE, -0.45 * SCALE, 0.0, 0.85, 0.0, 0.0, 0.0,
     0.0,
]

# Squeeze: shoulder roll crosses past zero so the arms actively press toward
# the midline. More-negative SQUEEZE_ROLL_MAG = harder clinch.
SQUEEZE_ROLL_MAG = -0.35    # rad. More negative = harder cross-over squeeze.

POSE_SQUEEZE = [
    -1.30 * SCALE,  SQUEEZE_ROLL_MAG * SCALE, 0.0, 0.40, 0.0, 0.0, 0.0,
    -1.30 * SCALE, -SQUEEZE_ROLL_MAG * SCALE, 0.0, 0.40, 0.0, 0.0, 0.0,
     0.0,
]

# POSE_HOLD: arms retracted toward the chest, used while the body is at
# full standing height. H1-2 has no waist-pitch joint, so the only way to
# keep the CoM over the feet while standing tall is to pull the held cone
# closer to the body. Shoulder pitch less negative (arm more vertical),
# elbow bent harder (forearm folds back), shoulder-roll squeeze preserved.
POSE_HOLD = [
    -0.80 * SCALE,  SQUEEZE_ROLL_MAG * SCALE, 0.0, 0.20, 0.0, 0.0, 0.0,
    -0.80 * SCALE, -SQUEEZE_ROLL_MAG * SCALE, 0.0, 0.20, 0.0, 0.0, 0.0,
     0.0,
]


# ── Stand-height targets ────────────────────────────────────────────────────
# Meters (float). Matches the verified C++ behavior in h1_2_hl_pose3.cpp:
# read current stand height with GetStandHeight, ramp it with SetStandHeight
# in metres at ~10 Hz while keeping the arm_sdk publisher pumping at 50 Hz.
KNEEL_DROP             = 0.16   # m — deep squat (past verified C++ 0.15)
DEFAULT_STAND_FALLBACK = 1.00   # m — used only if GetStandHeight fails


# ── arm_sdk controller ──────────────────────────────────────────────────────
class ArmSdkController:
    CTRL_DT     = 0.02   # 50 Hz
    MAX_VEL     = 0.5    # rad/s — per-step joint delta clamp
    WEIGHT_RATE = 0.4    # 1/s — fade rate for arm_sdk blend weight

    def __init__(self):
        self.msg              = unitree_hg_msg_dds__LowCmd_()
        self.weight           = 0.0
        self.current_pose     = list(POSE_READY)
        self.lowstate         = None
        self._got_state       = False

    def init(self):
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._on_lowstate, 10)
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()

        print("Waiting for lowstate...")
        while not self._got_state:
            time.sleep(0.1)

        # Capture the loco controller's current arm pose as our starting pt.
        self.current_pose = [self.lowstate.motor_state[j].q for j in ARM_JOINTS]

    def _on_lowstate(self, msg: LowState_):
        self.lowstate = msg
        self._got_state = True

    def warn_arm_stress(self, tau_limit: float = 25.0, temp_limit: float = 65.0):
        """Print a warning whenever an arm joint's torque or temperature
        exceeds the supplied limits. Run periodically to catch overloads
        before the firmware latches them out.
        """
        if self.lowstate is None:
            return
        for j in ARM_JOINTS:
            ms = self.lowstate.motor_state[j]
            tau = abs(ms.tau_est)
            t   = ms.temperature[0] if hasattr(ms, "temperature") else 0
            if tau > tau_limit or t > temp_limit:
                print(f"  [WARN] joint {j} tau={tau:.1f} Nm  temp={t} C")

    def _publish(self, target: list):
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
            self._publish(self.current_pose)
            time.sleep(self.CTRL_DT)
        self.weight = target

    def goto_pose(self, target: list, duration: float):
        """Drive arm pose toward target over `duration` seconds."""
        steps = int(duration / self.CTRL_DT)
        for _ in range(steps):
            self._publish(target)
            time.sleep(self.CTRL_DT)


# ── Loco controller wrapper ─────────────────────────────────────────────────
class LocoController:
    """Thin wrapper around LocoClient.

    Does NOT touch MotionSwitcher. Whatever mode the robot is in, arm_sdk
    overlays on top of it. Releasing the mode here would kill the balance
    controller and the robot would collapse.

    Stand height: SetStandHeight(meters) is the correct call (matches the
    verified C++ pattern in hit_h1-2/high_level/h1_2_hl_pose3.cpp). The
    critical detail is that the arm_sdk publisher MUST keep publishing at
    50 Hz throughout the height ramp — see ConePickupPalms.hold_with_height_ramp.
    """

    def __init__(self):
        self.client = LocoClient()
        self.client.SetTimeout(5.0)
        self.client.Init()

    def balance_stand(self):
        """Enter FSM 204 (main locomotion / balance stand)."""
        self.client.Start()
        time.sleep(2.0)

    def get_stand_height(self) -> float:
        """Read current stand height in meters via raw RPC.

        Python LocoClient exposes the API ID but no convenience method, so
        we call _Call directly. Returns DEFAULT_STAND_FALLBACK if the reply
        is missing or out of plausible range.
        """
        try:
            code, data = self.client._Call(ROBOT_API_ID_LOCO_GET_STAND_HEIGHT, "")
            if code != 0 or not data:
                return DEFAULT_STAND_FALLBACK
            h = float(json.loads(data).get("data", DEFAULT_STAND_FALLBACK))
            if h < 0.5 or h > 1.5:
                print(f"[loco] GetStandHeight returned {h:.3f} m, using fallback")
                return DEFAULT_STAND_FALLBACK
            return h
        except Exception as e:
            print(f"[loco] GetStandHeight error: {e}, using fallback")
            return DEFAULT_STAND_FALLBACK


# ── Choreography ────────────────────────────────────────────────────────────
class ConePickupPalms:
    HEIGHT_SEND_EVERY_N_TICKS = 5   # 50 Hz arm / 5 = 10 Hz SetStandHeight

    def __init__(self):
        self.arm     = ArmSdkController()
        self.loco    = LocoController()
        self.h_full  = None    # filled in after init from GetStandHeight
        self.h_kneel = None

    def hold_with_height_ramp(self, arm_target: list,
                              h_start: float, h_end: float,
                              duration: float):
        """Holds ``arm_target`` on rt/arm_sdk at 50 Hz while ramping
        SetStandHeight linearly from h_start->h_end at 10 Hz. Matches the
        verified C++ pattern (h1_2_hl_pose3.cpp run_height_ramp lambda).
        """
        dt    = self.arm.CTRL_DT
        steps = max(1, int(duration / dt))
        for i in range(steps):
            if i % self.HEIGHT_SEND_EVERY_N_TICKS == 0:
                t = i / steps
                h = h_start + t * (h_end - h_start)
                self.loco.client.SetStandHeight(h)
            self.arm._publish(arm_target)
            time.sleep(dt)

    def ramp_pose_and_height(self, arm_start: list, arm_end: list,
                             h_start: float, h_end: float,
                             duration: float):
        """Interpolate BOTH arm pose and stand height simultaneously.

        Used to retract the arms while the body rises (and extend them
        while the body lowers), so the CoM doesn't shift forward as the
        lever arm of the held cone grows. arm_sdk pumped at 50 Hz, height
        sent at 10 Hz.
        """
        dt    = self.arm.CTRL_DT
        steps = max(1, int(duration / dt))
        for i in range(steps):
            t = i / steps
            s = t * t * (3 - 2 * t)                    # smoothstep
            arm_target = [arm_start[k] + s * (arm_end[k] - arm_start[k])
                          for k in range(len(arm_start))]
            if i % self.HEIGHT_SEND_EVERY_N_TICKS == 0:
                h = h_start + t * (h_end - h_start)
                self.loco.client.SetStandHeight(h)
            self.arm._publish(arm_target)
            time.sleep(dt)

    def run(self):
        # 1. Ensure standing balance
        print("[loco] balance stand")
        self.loco.balance_stand()

        # 2. Engage arm_sdk smoothly
        print("[arm_sdk] init + weight 0->1 (3 s)")
        self.arm.init()
        self.arm.ramp_weight(1.0, duration=3.0)

        # 3. Ready pose (shoulders neutral + outward, elbows extended)
        print("[arm] ready pose (4 s)")
        self.arm.goto_pose(POSE_READY, duration=4.0)
        time.sleep(0.5)

        # Read current stand height now that the controller is settled.
        self.h_full  = self.loco.get_stand_height()
        self.h_kneel = self.h_full - KNEEL_DROP
        print(f"[loco] stand height: {self.h_full:.3f} -> {self.h_kneel:.3f} m kneel")

        # 4. Kneel — arm_sdk keeps pumping, SetStandHeight ramps at 10 Hz
        print("[loco+arm] kneel (5 s, arm holds READY)")
        self.hold_with_height_ramp(POSE_READY, self.h_full, self.h_kneel, 5.0)

        # 5. Reach: arms forward+out, palms inward, opened wider than cone
        print("[arm] reach (5 s)")
        self.arm.goto_pose(POSE_REACH, duration=5.0)
        time.sleep(0.5)

        # 6. Squeeze
        print("[arm] squeeze (2 s)")
        self.arm.goto_pose(POSE_SQUEEZE, duration=2.0)
        time.sleep(0.5)

        # 7. Lift body AND retract arms together — keeps CoM over the feet
        # as the body rises. Without this the loco controller steps forward.
        print("[loco+arm] lift + retract (5 s, SQUEEZE -> HOLD)")
        self.ramp_pose_and_height(POSE_SQUEEZE, POSE_HOLD,
                                  self.h_kneel, self.h_full, 5.0)
        self.arm.warn_arm_stress()

        # 8. Hold the lifted cone in the retracted HOLD pose for 4 s.
        print("[hold] 4 s (HOLD = retracted squeeze)")
        self.arm.goto_pose(POSE_HOLD, duration=4.0)
        self.arm.warn_arm_stress()

        # 9. Kneel again AND re-extend arms — mirror of step 7.
        print("[loco+arm] kneel + extend (5 s, HOLD -> SQUEEZE)")
        self.ramp_pose_and_height(POSE_HOLD, POSE_SQUEEZE,
                                  self.h_full, self.h_kneel, 5.0)

        print("[arm] release (open palms outward, 2 s)")
        self.arm.goto_pose(POSE_REACH, duration=2.0)

        print("[arm] back to ready (4 s)")
        self.arm.goto_pose(POSE_READY, duration=4.0)

        print("[loco+arm] stand up (5 s, arm holds READY)")
        self.hold_with_height_ramp(POSE_READY, self.h_kneel, self.h_full, 5.0)

        # 9. Fade arm_sdk back to loco
        print("[arm_sdk] weight 1->0 (1.5 s)")
        self.arm.ramp_weight(0.0, duration=1.5)

        print("Done.")

    def emergency_release(self):
        print("\n[abort] fading arm_sdk weight to 0 and standing up")
        try:
            self.arm.ramp_weight(0.0, duration=1.5)
            self.loco.client.SetStandHeight(self.h_full or DEFAULT_STAND_FALLBACK)
        except Exception as e:
            print(f"abort error: {e}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("interface", nargs="?",
                   help="Robot network interface, e.g. enp128s31f6.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 70)
    print("H1-2 CONE PICKUP — TWO-PALM SQUEEZE (no fingers)")
    print(" * Legs/balance: LocoClient FSM 204 + SetStandHeight ramp")
    print(" * Arms+waist:   rt/arm_sdk overlay with smooth weight blend")
    print(" * Hands:        NOT USED — cone is pinched between palms")
    print("=" * 70)
    input("Press Enter to continue (Ctrl+C aborts and re-stands)...")

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    demo = ConePickupPalms()
    try:
        demo.run()
    except KeyboardInterrupt:
        demo.emergency_release()
