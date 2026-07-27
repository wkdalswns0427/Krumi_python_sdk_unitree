#!/usr/bin/env python3
"""waist_probe.py - can ANY loco mode give arm_sdk the waist on H1-2?

Findings so far (2026-07-18): in balance stand (FSM 204) the waist ignores
nonzero rt/arm_sdk commands on the real robot. Unitree's own H1-2 teleop
(xr_teleoperate) commands joints 13-26 only, waist excluded, and the
official h1_2_arm_sdk example lists kWaistYaw but never moves it. So the
overlay path to the waist is unproven anywhere. ONE lever is untested: the
loco service exposes SET_BALANCE_MODE (api 8102) which the Python client
does not wrap; on G1 the balance mode changes what the controller reserves.

This tool, robot STANDING in balance, e-stop manned:
  1. reads FSM id / FSM mode / balance mode (api 8001/8002/8003), prints raw
  2. probes the waist via arm_sdk in the CURRENT mode (baseline)
  3. for each other balance mode (0/1), ENTER-gated: sets it, re-probes,
     and RESTORES the original mode afterward
  4. prints a verdict table

A probe = fade the overlay in, ramp waist to +8 deg over 1.5 s at the
official kp 200, read lowstate, ramp back, fade out. Arms hold their
snapshot throughout; every mode switch is announced first (the robot may
shift its stance when the balance mode changes).

Usage:
    python3 waist_probe.py --interface enp128s31f6
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "scripts"))
from replay_arm import ArmSdk, ARM_SEGS  # noqa: E402

NAMES = ([f"left_{s}" for s in ARM_SEGS] + [f"right_{s}" for s in ARM_SEGS]
         + ["waist_yaw"])
WAIST = NAMES.index("waist_yaw")
PROBE_DEG = 8.0
PROBE_S = 1.5
PASS_DEG = 4.0

# loco api ids (h1_loco_api.py; SET_BALANCE_MODE is not wrapped by the client)
GET_FSM_ID, GET_FSM_MODE, GET_BALANCE_MODE = 8001, 8002, 8003
SET_BALANCE_MODE = 8102


def loco_get(loco, api_id):
    code, data = loco._Call(api_id, "{}")
    val = None
    if code == 0 and data:
        try:
            val = json.loads(data).get("data")
        except Exception:
            val = data
    return code, val


def loco_set_balance(loco, mode):
    return loco._Call(SET_BALANCE_MODE, json.dumps({"data": int(mode)}))[0]


def probe(sdk):
    """Fade in, command +8 deg waist, measure, ramp back, fade out."""
    sdk.pose = sdk.snapshot()
    sdk.ramp_weight(1.0, 1.0)

    def ramp(to_rad, seconds):
        steps = max(1, int(seconds / ArmSdk.CTRL_DT))
        start = sdk.pose[WAIST]
        for i in range(1, steps + 1):
            pose = list(sdk.pose)
            pose[WAIST] = start + (i / steps) * (to_rad - start)
            sdk.publish(pose)
            time.sleep(ArmSdk.CTRL_DT)

    ramp(math.radians(PROBE_DEG), PROBE_S)
    time.sleep(0.3)
    got = math.degrees(abs(sdk.joint_q("waist_yaw")))
    ramp(0.0, PROBE_S)
    sdk.ramp_weight(0.0, 1.0)
    return got


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface", required=True)
    args = p.parse_args()

    from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
    sdk = ArmSdk(NAMES, args.interface, dry=False)   # inits DDS + lowstate
    sdk.gain_override["waist_yaw"] = (200.0, 2.0)     # official example gain
    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()

    print("=" * 66)
    print("WAIST AUTHORITY PROBE - robot STANDING, e-stop manned")
    print("=" * 66)
    for name, api in (("FSM id", GET_FSM_ID), ("FSM mode", GET_FSM_MODE),
                      ("balance mode", GET_BALANCE_MODE)):
        code, val = loco_get(loco, api)
        print(f"  {name:14}: {val}   (rpc code {code})")
    code, orig_mode = loco_get(loco, GET_BALANCE_MODE)
    results = {}

    input("\nENTER to probe the waist in the CURRENT mode... ")
    got = probe(sdk)
    results[f"current (bal={orig_mode})"] = got
    print(f"  -> measured {got:.1f} deg "
          f"({'RESPONDS' if got >= PASS_DEG else 'locked'})")

    candidates = [m for m in (0, 1) if m != orig_mode]
    for m in candidates:
        ans = input(f"\nTry balance mode {m}? Robot may shift stance. "
                    f"[y/N] ").strip().lower()
        if ans != "y":
            continue
        code = loco_set_balance(loco, m)
        print(f"  SetBalanceMode({m}) rpc code {code}")
        if code != 0:
            results[f"bal={m}"] = None
            continue
        time.sleep(1.5)
        input(f"ENTER to probe the waist in balance mode {m}... ")
        got = probe(sdk)
        results[f"bal={m}"] = got
        print(f"  -> measured {got:.1f} deg "
              f"({'RESPONDS' if got >= PASS_DEG else 'locked'})")

    if orig_mode is not None:
        print(f"\nrestoring balance mode {orig_mode}...")
        loco_set_balance(loco, orig_mode)

    print("\n" + "=" * 66)
    print("VERDICT")
    for k, v in results.items():
        s = "rpc refused" if v is None else \
            f"{v:.1f} deg -> {'WAIST AVAILABLE' if v >= PASS_DEG else 'locked'}"
        print(f"  {k:22}: {s}")
    print("=" * 66)
    print("If any mode says WAIST AVAILABLE: run pickup_and_turn with that")
    print("balance mode + --turn-mode waist. If all locked: the H1-2 loco")
    print("firmware reserves the waist; the waist-turn demo needs low-level")
    print("control on the gantry (loco released), or the stepping turn.")


if __name__ == "__main__":
    main()
