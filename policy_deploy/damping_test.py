#!/usr/bin/env python3
"""Damping-only bring-up test for H1-2 — run this BEFORE any policy deploy.

Sets every motor to (kp=0, kd=5, q=current_q) at 500 Hz.  Verifies:
  1. NIC / DDS connectivity
  2. CRC + mode_machine handshake
  3. MotionSwitcher release succeeded (no embedded controller fighting us)
  4. All 27 motor indices respond — robot sags slowly with damping, no twitches

If a single joint twitches violently, your CRC is wrong, mode_machine is
unset, or motion mode wasn't released — debug here, not in policy deploy.

Run with the robot HANGING.  ESTOP within arm's reach.

Usage:
    python3 damping_test.py [NIC]

Example:
    python3 damping_test.py eth0
"""

from __future__ import annotations

import sys
import time

import numpy as np
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from h12_joint_map import H1_2_NUM_MOTOR


CONTROL_DT = 0.002   # 500 Hz (matches Unitree's low_level example)
KP_DAMPING = 0.0     # zero stiffness — no target tracking
KD_DAMPING = 5.0     # damping only
MODE_PR    = 0       # series-control for pitch/roll


class DampingTest:
    def __init__(self) -> None:
        self.low_cmd: LowCmd_         = unitree_hg_msg_dds__LowCmd_()
        self.low_state: LowState_     = None
        self._mode_machine: int       = 0
        self._mode_machine_seen: bool = False
        self._crc                     = CRC()

    def init(self) -> None:
        # Release any default high-level controller (ai_sport / locomotion).
        # If we skip this, the embedded controller publishes its own LowCmd
        # at the same rate and your kp=0 stream gets overwritten.
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        while result.get("name"):
            print(f"[damping] Releasing motion mode: {result['name']}")
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1.0)
        print("[damping] Motion mode released.")

        self._cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._cmd_pub.Init()

        self._state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._state_sub.Init(self._on_lowstate, 10)

        print("[damping] Waiting for first lowstate...")
        while not self._mode_machine_seen:
            time.sleep(0.1)
        print(f"[damping] First lowstate received "
              f"(mode_machine={self._mode_machine}).")

    def _on_lowstate(self, msg: LowState_) -> None:
        self.low_state = msg
        if not self._mode_machine_seen:
            self._mode_machine = msg.mode_machine
            self._mode_machine_seen = True

    def start(self) -> None:
        self._thread = RecurrentThread(
            interval=CONTROL_DT,
            target=self._write_cmd,
            name="damping_ctrl",
        )
        self._thread.Start()

    def _write_cmd(self) -> None:
        if self.low_state is None:
            return
        self.low_cmd.mode_pr      = MODE_PR
        self.low_cmd.mode_machine = self._mode_machine
        for i in range(H1_2_NUM_MOTOR):
            m = self.low_cmd.motor_cmd[i]
            m.mode = 1                                    # Enable
            m.q   = float(self.low_state.motor_state[i].q)  # hold current
            m.dq  = 0.0
            m.kp  = KP_DAMPING
            m.kd  = KD_DAMPING
            m.tau = 0.0
        self.low_cmd.crc = self._crc.Crc(self.low_cmd)
        self._cmd_pub.Write(self.low_cmd)


def main() -> None:
    print("=" * 72)
    print(" H1-2 DAMPING-ONLY TEST")
    print(" kp=0, kd=5 on all 27 motors at 500 Hz.")
    print(" Expected behavior: limp robot, slow sag under gravity, no twitches.")
    print("=" * 72)
    print(" SAFETY:")
    print("   - Robot HANGING on overhead support")
    print("   - ESTOP within arm's reach")
    print("   - Feet NOT bearing weight")
    print("=" * 72)
    input("Press Enter to continue (Ctrl+C to abort)...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    test = DampingTest()
    test.init()
    test.start()

    print("[damping] Streaming kp=0/kd=5 cmd. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
            if test.low_state is not None:
                imu_rpy = list(test.low_state.imu_state.rpy)
                print(f"[damping] imu rpy = [{imu_rpy[0]:+.2f}, "
                      f"{imu_rpy[1]:+.2f}, {imu_rpy[2]:+.2f}]")
    except KeyboardInterrupt:
        print("\n[damping] Stopped.")


if __name__ == "__main__":
    main()
