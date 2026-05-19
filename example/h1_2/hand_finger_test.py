#!/usr/bin/env python3
"""Isolate one finger at a time to diagnose stuck motors.

Usage:
    python hand_finger_test.py enp128s31f6 --finger L2 --q 0.0
    python hand_finger_test.py enp128s31f6 --finger L2 --q 1.0

Finger names: R0..R5 (right pinky..thumb-rot), L0..L5 (left).
Other fingers are held open (1.0) so they can't mechanically block the one
under test.
"""
import argparse
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


NAMES = ["pinky", "ring", "middle", "index", "thumb-bend", "thumb-rot"]


def finger_index(label: str) -> int:
    side, idx = label[0].upper(), int(label[1])
    if side not in "RL" or not (0 <= idx <= 5):
        raise ValueError(f"bad finger label {label}; expected R0..R5 or L0..L5")
    return idx + (0 if side == "R" else 6)


def make_cmd(target_index: int, target_q: float) -> MotorCmds_:
    """All fingers open (1.0) except target finger which gets target_q."""
    msg = MotorCmds_()
    msg.cmds = []
    for i in range(12):
        m = unitree_go_msg_dds__MotorCmd_()
        m.q = target_q if i == target_index else 1.0
        msg.cmds.append(m)
    return msg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("interface", nargs="?")
    p.add_argument("--finger", required=True,
                   help="R0..R5 or L0..L5 (R=right L=left, 0=pinky..5=thumb-rot)")
    p.add_argument("--q", type=float, required=True,
                   help="Target q for the isolated finger (0.0=close, 1.0=open).")
    args = p.parse_args()

    idx = finger_index(args.finger)
    side, fnum = args.finger[0].upper(), int(args.finger[1])
    print(f"Testing {side} {NAMES[fnum]} (global idx={idx}) target q={args.q}")
    print("All other fingers held at 1.0 (open) to avoid interference.")

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
    pub.Init()

    def on_state(msg: MotorStates_):
        if not msg.states:
            return
        q = msg.states[idx].q
        marker = "  <-- TARGET" if abs(q - args.q) < 0.1 else ""
        print(f"  {side}{fnum} ({NAMES[fnum]}) q={q:.3f}{marker}")

    sub = ChannelSubscriber("rt/inspire/state", MotorStates_)
    sub.Init(on_state, 10)

    cmd = make_cmd(idx, args.q)
    print("Publishing at 20 Hz. Ctrl+C to stop.")
    try:
        while True:
            pub.Write(cmd)
            time.sleep(0.05)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
