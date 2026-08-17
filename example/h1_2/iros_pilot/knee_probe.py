#!/usr/bin/env python3
"""knee_probe.py - print the H1-2's live knee angles from lowstate.

Purpose: verify the CROUCHED stance for the S2 native run. The ergo figure
models the crouch as a 70 deg knee bend (--native-knee-deg); this reads the
real bend so the figure can be re-rendered with the measured value.

Run while the robot stands (normal, then LowStand / SetStandHeight):
    python3 knee_probe.py --interface enp128s31f6
"""

import argparse
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import (ChannelSubscriber,
                                         ChannelFactoryInitialize)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

LEFT_KNEE, RIGHT_KNEE = 3, 9
LEFT_HIP_PITCH, RIGHT_HIP_PITCH = 1, 7


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interface", required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()
    ChannelFactoryInitialize(0, args.interface)
    state = {}
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda m: state.update(m=m), 10)
    t0 = time.time()
    print("reading knees (Ctrl+C to stop)...")
    while time.time() - t0 < args.seconds:
        time.sleep(0.5)
        m = state.get("m")
        if m is None:
            continue
        lk = np.degrees(m.motor_state[LEFT_KNEE].q)
        rk = np.degrees(m.motor_state[RIGHT_KNEE].q)
        lh = np.degrees(m.motor_state[LEFT_HIP_PITCH].q)
        rh = np.degrees(m.motor_state[RIGHT_HIP_PITCH].q)
        print(f"  knee L {lk:+6.1f}  R {rk:+6.1f} deg   "
              f"hip-pitch L {lh:+6.1f}  R {rh:+6.1f} deg")
    if state.get("m") is None:
        sys.exit("no lowstate received - check --interface")
    print("note: REBA bend = knee joint angle off straight; pass the mean "
          "value to s2_pickstack_ergo.py --native-knee-deg")


if __name__ == "__main__":
    main()
