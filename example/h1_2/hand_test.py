#!/usr/bin/env python3
"""Open/close H1-2 Inspire hands via DDS while leaving body balance untouched.

The hand bridge service on the robot owns rt/inspire/cmd independently of the
high-level motion service, so this script publishes hand commands only — it
does NOT release the motion mode and does NOT touch rt/lowcmd. The robot
keeps standing/balancing under whatever high-level service is active.

Interface:
    Publish  -> rt/inspire/cmd     unitree_go::msg::dds_::MotorCmds_   (12 motors)
    Subscribe<- rt/inspire/state   unitree_go::msg::dds_::MotorStates_ (12 motors)

Motor layout (single combined message, 12 motors):
    Right hand [0..5] : pinky, ring, middle, index, thumb-bend, thumb-rotation
    Left  hand [6..11]: pinky, ring, middle, index, thumb-bend, thumb-rotation

q is normalized: 1.0 = open, 0.0 = closed.
"""

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


NUM_HAND_MOTORS = 12
HAND_OPEN  = 1.0
HAND_CLOSE = 0.0


def make_hand_cmd(q: float) -> MotorCmds_:
    msg = MotorCmds_()
    msg.cmds = []
    for _ in range(NUM_HAND_MOTORS):
        m = unitree_go_msg_dds__MotorCmd_()
        m.q  = q
        m.dq = 0.0
        m.tau = 0.0
        m.kp = 0.0
        m.kd = 0.0
        msg.cmds.append(m)
    return msg


class HandOnlyController:
    def __init__(self, hand_q: float, publish_hz: float = 20.0):
        self.hand_cmd = make_hand_cmd(hand_q)
        self.period   = 1.0 / publish_hz

    def init(self):
        self.cmd_pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self.cmd_pub.Init()

        self.state_sub = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self.state_sub.Init(self._on_state, 10)

    def _on_state(self, msg: MotorStates_):
        if not msg.states:
            return
        rq = [f"{msg.states[i].q:.2f}" for i in range(6)]
        lq = [f"{msg.states[i].q:.2f}" for i in range(6, 12)]
        print(f"[hand] R={rq}  L={lq}")

    def set_hand_q(self, q: float):
        for cmd in self.hand_cmd.cmds:
            cmd.q = q

    def run(self):
        while True:
            self.cmd_pub.Write(self.hand_cmd)
            time.sleep(self.period)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Command both Inspire hands via DDS. Leaves body balance alone."
    )
    parser.add_argument("interface", nargs="?",
                        help="Robot network interface, e.g. enp128s31f6.")
    parser.add_argument("--hand-q", type=float, default=HAND_OPEN,
                        help=f"Hand: {HAND_OPEN}=open, {HAND_CLOSE}=closed (default: {HAND_OPEN}).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("This script controls hands only — the robot's high-level balance")
    print("controller stays active. Keep the robot supported regardless.")
    input("Press Enter to continue...")

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    controller = HandOnlyController(args.hand_q)
    controller.init()

    print(f"Publishing hand_q={args.hand_q}. Press Ctrl+C to stop.")
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nStopped.")
