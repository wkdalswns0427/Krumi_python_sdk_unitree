#!/usr/bin/env python3
"""Minimal probe: subscribe to the Inspire hand state topic.

If [hand] lines print -> bridge is up, hand_test.py should work.
If nothing prints     -> bridge not running on robot, or DDS not crossing.
"""
import sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_


def on_msg(msg):
    q = [f"{s.q:.2f}" for s in msg.states[:12]]
    print(f"[hand] q={q}")


if __name__ == "__main__":
    iface = sys.argv[1] if len(sys.argv) > 1 else None
    if iface:
        ChannelFactoryInitialize(0, iface)
    else:
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber("rt/inspire/state", MotorStates_)
    sub.Init(on_msg, 10)

    print(f"Listening on rt/inspire/state via {iface or 'default'} for 15s...")
    for _ in range(15):
        time.sleep(1)
    print("Done.")
