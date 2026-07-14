#!/usr/bin/env python3
"""H1-2 low-level pose hold — captures whatever pose the robot is in when
the script starts and holds it.

Workflow:
    1. An assistant manually poses the robot (arms + legs) into the
       desired seated/cockpit position.
    2. The assistant keeps holding it.
    3. Run this script. It releases the high-level motion mode, snapshots
       the current joint angles, and starts driving all 27 motors at
       500 Hz to hold that captured pose.
    4. Once the hold is stable, the assistant lets go.

Notes
-----
* The captured pose is whatever the robot reports in rt/lowstate at the
  moment of init. If the assistant lets go BEFORE the script starts the
  control loop, the limbs will sag and that sagged pose is what gets
  held. Make sure the assistant keeps the pose firm through the
  "[cockpit] waiting for lowstate..." line.
* Legs use soft hold gains that cannot support body weight on their own.
  The cockpit MUST take the load.
* No balance feedback. No external control inputs. This is the simplest
  possible hold script — wire a trajectory generator on top of
  ``self.target_q`` later if you want motion.

SAFETY
------
1. Robot must be securely supported BEFORE you press Enter. Releasing
   the high-level motion mode kills the balance controller, and the
   robot WILL collapse the instant Enter is pressed if the cockpit and
   the assistant aren't both holding it.
2. Keep --kp-scale low (0.3-0.5) on the first run. Bring it up only
   after you've watched the hold stay stable.
3. Keep the remote / E-stop in reach.

Usage:
    python3 drive_cockpit.py enp128s31f6 --kp-scale 0.3
"""

import argparse
import sys
import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


H1_2_NUM_MOTOR = 27


# ── Joint indices (H1-2, official ordering) ─────────────────────────────────
class J:
    LHipYaw       = 0
    LHipPitch     = 1
    LHipRoll      = 2
    LKnee         = 3
    LAnklePitch   = 4
    LAnkleRoll    = 5
    RHipYaw       = 6
    RHipPitch     = 7
    RHipRoll      = 8
    RKnee         = 9
    RAnklePitch   = 10
    RAnkleRoll    = 11
    WaistYaw      = 12
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


class Mode:
    PR = 0   # series control for pitch/roll joints (URDF convention)
    AB = 1   # parallel A/B control


# ── Per-joint hold gains ────────────────────────────────────────────────────
# Soft enough that the robot doesn't fight the cockpit if it shifts under
# the seat belts, stiff enough to keep the seated pose against gravity.
# Hip pitch / knee carry the most static torque (high flexion in sitting),
# so they get the highest leg gains. Ankles deliberately low — they don't
# need to do balance work here.
#
# ALL values are scaled by --kp-scale at runtime. Start scale around 0.3-0.4.
KP_HOLD = [
    # Left leg: yaw, pitch, roll, knee, ankle pitch, ankle roll
     40,  80,  40,  80, 15, 15,
    # Right leg
     40,  80,  40,  80, 15, 15,
    # Waist yaw
     80,
    # Left arm: shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw
     60,  60,  40,  40, 15, 15, 15,
    # Right arm
     60,  60,  40,  40, 15, 15, 15,
]
KD_HOLD = [
    1.5, 2.0, 1.5, 2.0, 0.5, 0.5,
    1.5, 2.0, 1.5, 2.0, 0.5, 0.5,
    1.5,
    1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 0.5,
    1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 0.5,
]
assert len(KP_HOLD) == H1_2_NUM_MOTOR
assert len(KD_HOLD) == H1_2_NUM_MOTOR


# ── Controller ──────────────────────────────────────────────────────────────
class DriveCockpit:
    CTRL_DT = 0.002   # 500 Hz, matching the H1-2 low-level example

    def __init__(self, kp_scale: float):
        self.kp_scale     = max(0.0, min(1.0, kp_scale))

        self.low_cmd      = unitree_hg_msg_dds__LowCmd_()
        self.low_state    = None
        self.mode_machine = 0
        self.got_state    = False
        self.crc          = CRC()

        self.target_q = None   # 27-vec captured from lowstate at init

    # ── DDS plumbing ────────────────────────────────────────────────────────
    def init(self):
        # Releasing the high-level service is what makes low-level rt/lowcmd
        # writes effective. On a STANDING robot this collapses it instantly;
        # only do this when the robot is supported by the cockpit.
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, result = msc.CheckMode()
        while result.get("name"):
            print(f"[cockpit] releasing motion mode: {result['name']}")
            msc.ReleaseMode()
            time.sleep(1.0)
            _, result = msc.CheckMode()
        print("[cockpit] motion mode released")

        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._on_lowstate, 10)

        print("[cockpit] waiting for lowstate... (keep holding the pose)")
        while not self.got_state:
            time.sleep(0.05)

        self.target_q = [self.low_state.motor_state[i].q for i in range(H1_2_NUM_MOTOR)]
        print("[cockpit] captured pose:")
        print(f"  legs  [0..11] = {[round(q, 2) for q in self.target_q[0:12]]}")
        print(f"  waist [12]    = {round(self.target_q[12], 2)}")
        print(f"  L arm [13..19]= {[round(q, 2) for q in self.target_q[13:20]]}")
        print(f"  R arm [20..26]= {[round(q, 2) for q in self.target_q[20:27]]}")

    def _on_lowstate(self, msg: LowState_):
        self.low_state = msg
        if not self.got_state:
            self.mode_machine = msg.mode_machine
            self.got_state = True

    def start(self):
        self.thread = RecurrentThread(
            interval=self.CTRL_DT, target=self._step, name="cockpit_ctrl"
        )
        self.thread.Start()

    # ── Control loop ────────────────────────────────────────────────────────
    def _step(self):
        self.low_cmd.mode_pr      = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine

        for i in range(H1_2_NUM_MOTOR):
            mc = self.low_cmd.motor_cmd[i]
            mc.mode = 1   # enable
            mc.q    = self.target_q[i]
            mc.dq   = 0.0
            mc.tau  = 0.0
            mc.kp   = KP_HOLD[i] * self.kp_scale
            mc.kd   = KD_HOLD[i]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("interface", nargs="?",
                   help="Robot network interface, e.g. enp128s31f6.")
    p.add_argument("--kp-scale", type=float, default=0.3,
                   help="Scale on the KP_HOLD table (0..1). Start at 0.3 and "
                        "raise only once the hold is visibly stable.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("H1-2 LOW-LEVEL POSE HOLD (capture-and-hold)")
    print("=" * 70)
    print("Have an assistant hold the robot in the desired pose NOW and keep")
    print("holding it. Pressing Enter releases the high-level motion mode")
    print("and the robot will collapse without that support.")
    print(f"  Kp scale : {args.kp_scale:.2f}  (applied to KP_HOLD table)")
    print("=" * 70)
    try:
        input("Press Enter to release motion mode and start holding... "
              "(Ctrl+C aborts)")
    except KeyboardInterrupt:
        print("\nAborted before init.")
        sys.exit(0)

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    app = DriveCockpit(args.kp_scale)
    app.init()
    app.start()

    print("[cockpit] holding captured pose. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[cockpit] stopping. Re-engage motion mode (e.g. via the "
              "remote) before standing the robot up.")


if __name__ == "__main__":
    main()
