#!/usr/bin/env python3
"""pickup_and_turn.py - robot-native box relocation: lift at waist, turn, set down.
Usage:
    /usr/bin/python3 pickup_and_turn.py --dry-run
    python3 pickup_and_turn.py --interface enp128s31f6                # auto
    python3 pickup_and_turn.py --interface enp128s31f6 --turn-mode step \
        --turn-deg 180
"""

import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_SCRIPTS = os.path.join(HERE, "..", "experiments", "scripts")
sys.path.insert(0, EXP_SCRIPTS)
from replay_arm import (ArmSdk, GravityFF, ARM_SEGS, SDK_INDEX,  # noqa: E402
                        FROZEN_GRAVITY_GAIN)

NAMES = ([f"left_{s}" for s in ARM_SEGS] + [f"right_{s}" for s in ARM_SEGS]
         + ["waist_yaw"])
WAIST = NAMES.index("waist_yaw")
WAIST_KP_KD = (200.0, 2.0)          # official h1_2_arm_sdk example gains

# Poses extracted from the executed M3_1 trajectory (see module docstring).
# Order per arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow; wrists 0.
CARRY_L = [-0.234, 0.082, 0.071, 0.902]
CARRY_R = [-0.131, -0.174, -0.153, 0.941]
REACH_L = [0.254, 0.013, -0.064, 0.886]     # M3 lowest-wrist + 0.25 pitch
REACH_R = [0.361, -0.172, -0.035, 0.952]

PROBE_DEG = 8.0                     # waist probe amplitude
PROBE_S = 1.5                       # probe ramp time
PROBE_PASS_DEG = 4.0                # measured waist must exceed this


def full_pose(l4, r4, squeeze=0.0, waist=0.0):
    """15-joint pose from per-arm 4-DOF (+ inward squeeze on shoulder roll)."""
    l = list(l4)
    r = list(r4)
    l[1] -= squeeze                          # left roll inward
    r[1] += squeeze                          # right roll inward (mirrored)
    return l + [0.0, 0.0, 0.0] + r + [0.0, 0.0, 0.0] + [waist]


def waist_limit_deg(urdf):
    for j in ET.parse(urdf).getroot().findall("joint"):
        if j.get("name") == "torso_joint":
            lim = j.find("limit")
            return (math.degrees(float(lim.get("lower"))),
                    math.degrees(float(lim.get("upper"))))
    sys.exit("torso_joint not found in URDF")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface", help="robot network interface, e.g. enp128s31f6")
    p.add_argument("--turn-mode", choices=["auto", "waist", "step"],
                   default="auto",
                   help="waist joint, whole-body stepping, or probe-then-pick")
    p.add_argument("--turn-deg", type=float, default=130.0,
                   help="signed turn target. Waist mode clamps to the URDF "
                        "torso limit -5 deg (+/-130); step mode allows up to "
                        "+/-180")
    p.add_argument("--turn-rate", type=float, default=15.0,
                   help="waist-mode speed, deg/s (default 15, quasi-static)")
    p.add_argument("--step-rate", type=float, default=0.3,
                   help="step-mode yaw rate, rad/s (default 0.3 ~ 17 deg/s)")
    p.add_argument("--squeeze", type=float, default=0.10,
                   help="inward shoulder-roll squeeze while holding, rad")
    p.add_argument("--gravity-gain", type=float, default=FROZEN_GRAVITY_GAIN)
    p.add_argument("--no-gravity-ff", action="store_true",
                   help="disable the gravity feedforward (on by default; the "
                        "box is a payload, keep it on)")
    p.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    lo, hi = waist_limit_deg(args.urdf)
    waist_target = max(lo + 5.0, min(hi - 5.0, args.turn_deg))
    step_target = max(-180.0, min(180.0, args.turn_deg))
    if args.turn_mode != "step" and abs(waist_target - args.turn_deg) > 1.0:
        print(f"[pick+turn] waist mode: {args.turn_deg:+.0f} deg requested but "
              f"the URDF torso limit is [{lo:.0f}, {hi:.0f}]; waist target "
              f"clamps to {waist_target:+.0f} deg (step mode can do the full "
              f"turn)")

    reach = full_pose(REACH_L, REACH_R)
    grasp = full_pose(REACH_L, REACH_R, squeeze=args.squeeze)
    carry = full_pose(CARRY_L, CARRY_R, squeeze=args.squeeze)

    wt = abs(waist_target) / args.turn_rate
    st = math.radians(abs(step_target)) / args.step_rate
    print(f"[pick+turn] mode={args.turn_mode}; waist turn {waist_target:+.0f} "
          f"deg in {wt:.0f}s, step turn {step_target:+.0f} deg in {st:.0f}s")
    print(f"[pick+turn] M3 mimicry replay reference: 61-73 s at full speed")

    if args.dry_run:
        print("[pick+turn] dry-run OK (no DDS, nothing published)")
        return

    print("=" * 66)
    print("PICKUP AND TURN")
    print("Robot STANDING IN BALANCE (FSM 204). Clear the FULL sweep arc")
    print("(box radius ~0.8 m). In step mode the robot STEPS IN PLACE:")
    print("its whole footprint must be clear. E-stop manned.")
    print("=" * 66)

    gravity = {}
    if not args.no_gravity_ff:
        gravity = {s: GravityFF(args.urdf, s) for s in ("left", "right")}
    sdk = ArmSdk(NAMES, args.interface, dry=False,
                 gravity=gravity, gravity_gain=args.gravity_gain)
    sdk.gain_override["waist_yaw"] = WAIST_KP_KD
    sdk.pose = sdk.snapshot()
    rest = list(sdk.pose)
    stamps = [("start", time.time())]

    def mark(name):
        stamps.append((name, time.time()))

    def waist_ramp(to_rad, seconds):
        steps = max(1, int(seconds / ArmSdk.CTRL_DT))
        start = sdk.pose[WAIST]
        for i in range(1, steps + 1):
            pose = list(sdk.pose)
            pose[WAIST] = start + (i / steps) * (to_rad - start)
            sdk.publish(pose)
            time.sleep(ArmSdk.CTRL_DT)

    def probe_waist():
        """Does the overlay actually own the waist? Command 8 deg, check
        lowstate follows; always ramp back to 0."""
        print(f"[pick+turn] probing waist authority ({PROBE_DEG:.0f} deg, "
              f"{PROBE_S:.1f} s)...")
        waist_ramp(math.radians(PROBE_DEG), PROBE_S)
        time.sleep(0.3)
        got = math.degrees(abs(sdk.joint_q("waist_yaw")))
        waist_ramp(0.0, PROBE_S)
        ok = got >= PROBE_PASS_DEG
        print(f"[pick+turn] waist probe: measured {got:.1f} deg -> "
              f"{'WAIST RESPONDS' if ok else 'waist locked by loco; using STEP mode'}")
        return ok

    loco = None

    def step_turn(deg):
        """Rotate the whole robot in place (open-loop timed), arms holding."""
        nonlocal loco
        if loco is None:
            from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
            loco = LocoClient()
            loco.SetTimeout(10.0)
            loco.Init()
        vyaw = args.step_rate if deg > 0 else -args.step_rate
        dur = math.radians(abs(deg)) / args.step_rate
        print(f"[pick+turn] stepping {deg:+.0f} deg at {args.step_rate:.2f} "
              f"rad/s (~{dur:.0f} s), arms holding...")
        loco.Move(0.0, 0.0, vyaw, continous_move=True)
        t0 = time.time()
        while time.time() - t0 < dur:
            sdk.publish(sdk.pose)            # keep the hold + gravity ff alive
            time.sleep(ArmSdk.CTRL_DT)
        loco.StopMove()
        time.sleep(0.5)

    try:
        input("ENTER to fade the overlay in (Ctrl+C aborts)... ")
        sdk.ramp_weight(1.0, 2.0)
        mark("overlay in")

        mode = args.turn_mode
        if mode == "auto":
            mode = "waist" if probe_waist() else "step"
        turn_deg = waist_target if mode == "waist" else step_target
        print(f"[pick+turn] TURN MODE: {mode} ({turn_deg:+.0f} deg)")

        print("[pick+turn] reaching for the box (3 s)...")
        sdk.ramp_pose(reach, 3.0)
        mark("reach")
        input("Place the box in the hands, ENTER to grasp + lift... ")
        sdk.ramp_pose(grasp, 1.0)
        sdk.ramp_pose(carry, 2.5)
        mark("lift")
        input(f"ENTER to turn {turn_deg:+.0f} deg ({mode})... ")
        if mode == "waist":
            waist_ramp(math.radians(turn_deg), wt)
        else:
            step_turn(turn_deg)
        mark("turn")
        time.sleep(1.0)
        input("ENTER to set the box down... ")
        # lower with the squeeze held; waist stays wherever the turn put it
        lower = full_pose(REACH_L, REACH_R, squeeze=args.squeeze,
                          waist=sdk.pose[WAIST])
        sdk.ramp_pose(lower, 2.5)
        mark("lower")
        input("Take the box, ENTER to release and come home... ")
        sdk.ramp_pose(full_pose(REACH_L, REACH_R, waist=sdk.pose[WAIST]), 1.0)
        if mode == "waist":
            waist_ramp(0.0, wt)
        else:
            step_turn(-turn_deg)
        mark("turn home")
        sdk.ramp_pose(rest, 3.0)
        sdk.ramp_weight(0.0, 2.0)
        mark("done")
        print("[pick+turn] overlay released; loco has the arms again")
        print("-" * 50)
        print(f"phase timings (mode {mode}, for the mimicry comparison):")
        for (n0, t0), (n1, t1) in zip(stamps, stamps[1:]):
            print(f"    {n1:<14}{t1 - t0:6.1f} s")
        task = stamps[-2][1] - stamps[1][1]
        print(f"    TASK TOTAL (reach..turn home): {task:.1f} s")
    except KeyboardInterrupt:
        print("\n[pick+turn] INTERRUPT: stopping loco, fading overlay out")
        try:
            if loco is not None:
                loco.StopMove()
            sdk.ramp_weight(0.0, 2.5)
        finally:
            print("[pick+turn] overlay released")
        sys.exit(1)


if __name__ == "__main__":
    main()
