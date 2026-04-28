#!/usr/bin/env python3
"""Flex H1-2 elbows to a target angle and open both Inspire RH56DFTP hands.

REQUIRED before running: start the RS-485 DDS bridge on the robot:
    ssh unitree@192.168.123.1
    cd inspire_hand_ws
    python inspire_hand_sdk/example/Headless_driver_485_double.py
    # (or run Headless_driver_485.py separately for LR='r' and LR='l')

Inspire RH56DFTP DDS interface (Unitree official docs):
    Publish  -> rt/inspire_hand/ctrl/r   inspire::inspire_hand_ctrl  (right)
    Publish  -> rt/inspire_hand/ctrl/l   inspire::inspire_hand_ctrl  (left)
    Subscribe<- rt/inspire_hand/state/r  inspire::inspire_hand_state (right)
    Subscribe<- rt/inspire_hand/state/l  inspire::inspire_hand_state (left)

Joint order per hand (6 joints):
    0=pinky  1=ring  2=middle  3=index  4=thumb-bend  5=thumb-rotation

Position units (mode=2):  0 = fully open    ~1000 = fully closed
Angle units    (mode=1):  0 = straight open  (in 0.1-degree steps)

mode bitmask:  1=angle  2=position  4=force  8=speed  (combine by adding)
"""

import argparse
import math
import time
from dataclasses import dataclass, field

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


# ---------------------------------------------------------------------------
# Inspire RH56DFTP IDL types (inline — matches inspire_hand_sdk/hand_idl)
# ---------------------------------------------------------------------------

@dataclass
@annotate.final
@annotate.autoid("sequential")
class inspire_hand_ctrl(idl.IdlStruct, typename="inspire.inspire_hand_ctrl"):
    pos_set:   types.sequence[types.int16, 6] = field(default_factory=lambda: [0]*6)
    angle_set: types.sequence[types.int16, 6] = field(default_factory=lambda: [0]*6)
    force_set: types.sequence[types.int16, 6] = field(default_factory=lambda: [0]*6)
    speed_set: types.sequence[types.int16, 6] = field(default_factory=lambda: [0]*6)
    mode:      types.int8 = 0


@dataclass
@annotate.final
@annotate.autoid("sequential")
class inspire_hand_state(idl.IdlStruct, typename="inspire.inspire_hand_state"):
    pos_act:     types.sequence[types.int16, 6] = field(default_factory=lambda: [0]*6)
    angle_act:   types.sequence[types.int16, 6] = field(default_factory=lambda: [0]*6)
    force_act:   types.sequence[types.int16, 6] = field(default_factory=lambda: [0]*6)
    current:     types.sequence[types.int16, 6] = field(default_factory=lambda: [0]*6)
    err:         types.sequence[types.uint8, 6] = field(default_factory=lambda: [0]*6)
    status:      types.sequence[types.uint8, 6] = field(default_factory=lambda: [0]*6)
    temperature: types.sequence[types.uint8, 6] = field(default_factory=lambda: [0]*6)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H1_2_NUM_BODY_MOTOR = 27

HAND_OPEN  = 0     # position units
HAND_CLOSE = 1000  # position units

MODE_POSITION = 2  # mode bitmask: use pos_set


class H1_2_JointIndex:
    LeftElbow  = 16
    RightElbow = 23


class Mode:
    PR = 0


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class HandAndElbowTest:
    def __init__(self, elbow_angle: float, ramp_duration: float, hand_pos: int):
        self.control_dt    = 0.002   # 500 Hz body control
        self.hand_dt       = 0.05    # 20 Hz — RS-485 bridge runs ~20 Hz
        self.elbow_angle   = elbow_angle
        self.ramp_duration = ramp_duration

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.right_hand_cmd = inspire_hand_ctrl()
        self.right_hand_cmd.pos_set = [hand_pos] * 6
        self.right_hand_cmd.mode    = MODE_POSITION

        self.left_hand_cmd = inspire_hand_ctrl()
        self.left_hand_cmd.pos_set = [hand_pos] * 6
        self.left_hand_cmd.mode    = MODE_POSITION

        self.low_state          = None
        self.mode_machine       = 0
        self.start_q            = None
        self.start_time         = None
        self.received_low_state = False
        self.crc                = CRC()

    def init(self):
        self._release_motion_mode()

        # Body (H1-2 uses unitree_hg LowCmd_/LowState_)
        self.lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_pub.Init()

        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init(self._low_state_handler, 10)

        # Inspire RH56DFTP hands — separate topic per hand, custom IDL
        self.right_hand_pub = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_hand_ctrl)
        self.left_hand_pub  = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_hand_ctrl)
        self.right_hand_pub.Init()
        self.left_hand_pub.Init()

        self.right_state_sub = ChannelSubscriber("rt/inspire_hand/state/r", inspire_hand_state)
        self.left_state_sub  = ChannelSubscriber("rt/inspire_hand/state/l", inspire_hand_state)
        self.right_state_sub.Init(lambda msg: self._hand_state_handler("R", msg), 10)
        self.left_state_sub.Init( lambda msg: self._hand_state_handler("L", msg), 10)

    def start(self):
        print("Waiting for body low state...")
        while not self.received_low_state:
            time.sleep(0.1)

        self.start_q    = [self.low_state.motor_state[i].q for i in range(H1_2_NUM_BODY_MOTOR)]
        self.start_time = time.time()

        self.lowcmd_thread = RecurrentThread(
            interval=self.control_dt, target=self._write_low_cmd, name="elbow_control"
        )
        self.hand_thread = RecurrentThread(
            interval=self.hand_dt, target=self._write_hand_cmd, name="hand_control"
        )
        self.lowcmd_thread.Start()
        self.hand_thread.Start()

    def _release_motion_mode(self):
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, result = msc.CheckMode()
        while result["name"]:
            print(f"Releasing motion mode: {result['name']}")
            msc.ReleaseMode()
            time.sleep(1.0)
            _, result = msc.CheckMode()

    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg
        if not self.received_low_state:
            self.mode_machine       = msg.mode_machine
            self.received_low_state = True

    def _hand_state_handler(self, side: str, msg: inspire_hand_state):
        print(f"[{side} hand] pos={list(msg.pos_act)}  err={list(msg.err)}")

    def _write_low_cmd(self):
        elapsed = time.time() - self.start_time
        ratio   = min(max(elapsed / self.ramp_duration, 0.0), 1.0)

        self.low_cmd.mode_pr      = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine

        for i in range(H1_2_NUM_BODY_MOTOR):
            motor      = self.low_cmd.motor_cmd[i]
            motor.mode = 1
            motor.q    = self.start_q[i]
            motor.dq   = 0.0
            motor.tau  = 0.0
            motor.kp   = 100.0 if i < 13 else 50.0
            motor.kd   = 1.0

        for elbow in (H1_2_JointIndex.LeftElbow, H1_2_JointIndex.RightElbow):
            start = self.start_q[elbow]
            self.low_cmd.motor_cmd[elbow].q = start + ratio * (self.elbow_angle - start)

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_pub.Write(self.low_cmd)

    def _write_hand_cmd(self):
        self.right_hand_pub.Write(self.right_hand_cmd)
        self.left_hand_pub.Write(self.left_hand_cmd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Flex H1-2 elbows and command both Inspire RH56DFTP hands."
    )
    parser.add_argument(
        "interface", nargs="?",
        help="Robot network interface, e.g. enp2s0.",
    )
    parser.add_argument(
        "--elbow-angle", type=float, default=math.pi / 2.0,
        help="Target elbow angle in radians (default: pi/2).",
    )
    parser.add_argument(
        "--ramp-duration", type=float, default=3.0,
        help="Seconds to ramp to the target elbow angle (default: 3).",
    )
    parser.add_argument(
        "--hand-pos", type=int, default=HAND_OPEN,
        help=f"Hand position: {HAND_OPEN}=open, {HAND_CLOSE}=closed (default: {HAND_OPEN}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("WARNING: Ensure arms and hands have clear space.")
    print("REQUIRED: inspire_hand_ws driver must be running on your PC (not the robot).")
    print("  cd ~/mj_ws/z_unitree_official/inspire_hand_ws")
    print("  python inspire_hand_sdk/example/Headless_driver_485_double.py")
    input("Press Enter to continue...")

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    test = HandAndElbowTest(args.elbow_angle, args.ramp_duration, args.hand_pos)
    test.init()
    test.start()

    print("Running. Press Ctrl+C to stop.")
    while True:
        time.sleep(1.0)
