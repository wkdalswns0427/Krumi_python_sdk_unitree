#!/usr/bin/env python3
"""zed_realtime_node.py - live ZED skeleton -> H1-2 arm_sdk @ ~50 Hz.

Track C real-time teleop node. Feeds live ZED body-tracking keypoints
through the IROS-frozen retarget (direction-matching Gauss-Newton IK on the
H1-2 URDF, --yaw-neutral 0.02 --yaw-reg 0.05 global) and streams to
rt/arm_sdk with the same weight-fade / gravity-feedforward pattern as
replay_arm.py. Causal only: One-Euro on joint targets (no zero-phase
filters), per-joint velocity clamp, dead-man ENTER at arm and start.

Latency (t0..t3) is logged per frame; --print-stats prints median / p95
at exit.

    t0 : ZED grab returned
    t1 : skeleton -> canonical joints -> arm directions ready
    t2 : per-side IK solved, full pose assembled and filtered
    t3 : command published on rt/arm_sdk

Reuses:
  - ../../../experiments/scripts/retarget_arm.py  (IK, torso frame, OneEuro)
  - ../../../experiments/scripts/replay_arm.py    (ArmSdk, GravityFF)
  - ./zed_to_joints.py                            (canonical joint mapping)

Usage:
    # pre-flight, no DDS, no robot; can run without a person if you also
    # pass --allow-no-body (holds the home pose):
    /usr/bin/python3 zed_realtime_node.py --dry-run

    # on the robot (conda env with unitree_sdk2py, robot STANDING in balance):
    python3 zed_realtime_node.py --interface enp128s31f6

    # self-test (no hardware, no ZED):
    /usr/bin/python3 zed_realtime_node.py --self-test

SAFETY: workspace clear, arms only, second person on the e-stop. The node
snapshots the current arm pose, fades the arm_sdk weight in, ramps to a
neutral home pose, then arms streaming only after the operator presses
ENTER twice. Ctrl+C at any point freezes the last commanded pose and fades
the weight out cleanly.
"""

import argparse
import csv
import os
import signal
import sys
import time

import numpy as np

# ── Locate the IROS-frozen retarget + replay modules ─────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "experiments", "scripts"))
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _HERE)

from zed_to_joints import (  # noqa: E402
    canonical_index_map, pick_primary_body)
from retarget_arm import (  # noqa: E402
    ArmChainFK, solve_arm, torso_frame, arm_directions, OneEuro,
    load_limits, STRAIGHT_ARM_DEG)
from replay_arm import (  # noqa: E402
    ArmSdk, GravityFF, ARM_SEGS, FROZEN_GRAVITY_GAIN)

SIDES = ("left", "right")
OUT_JOINTS = [f"{s}_{seg}" for s in SIDES for seg in ARM_SEGS] + ["waist_yaw"]

# Neutral "home" pose = all zeros. Arms hang at the sides (H1-2 zero pose).
HOME_POSE = [0.0] * len(OUT_JOINTS)


_stop = False


def _sigint(signum, frame):
    global _stop
    _stop = True
    print("\n[realtime] Ctrl+C -> freezing pose, fading weight out")


def body_to_joints_dict(body, index_map):
    """{canonical_name: np.array([x,y,z])} for one body. NaN keypoints dropped."""
    out = {}
    if body is None:
        return out
    kp = body.keypoint
    for zed_idx, canonical in index_map.items():
        if zed_idx >= len(kp):
            continue
        x, y, z = float(kp[zed_idx][0]), float(kp[zed_idx][1]), float(kp[zed_idx][2])
        if x != x or y != y or z != z:
            continue
        out[canonical] = np.array([x, y, z])
    return out


def clamp_velocity(prev_pose, target_pose, max_vel, dt):
    """Per-joint rate limit: |q_new - q_prev| <= max_vel * dt."""
    if prev_pose is None:
        return list(target_pose)
    out = []
    max_step = max_vel * dt
    for p, t in zip(prev_pose, target_pose):
        d = max(-max_step, min(max_step, t - p))
        out.append(p + d)
    return out


def assemble_pose(q_per_side):
    """15-joint pose ordered per OUT_JOINTS. Wrists + waist held at zero."""
    pose = []
    for side in SIDES:
        q4 = q_per_side[side]
        pose.extend([float(q4[0]), float(q4[1]), float(q4[2]), float(q4[3]),
                     0.0, 0.0, 0.0])
    pose.append(0.0)
    return pose


def latency_stats(rows):
    """Median / p95 (ms) for each t0->t1, t1->t2, t2->t3, t0->t3."""
    if not rows:
        return {}
    a = np.array(rows)  # (n, 5): frame, t0, t1, t2, t3
    stages = {
        "grab->skeleton": (a[:, 2] - a[:, 1]) * 1000,
        "skeleton->ik":   (a[:, 3] - a[:, 2]) * 1000,
        "ik->publish":    (a[:, 4] - a[:, 3]) * 1000,
        "end-to-end":     (a[:, 4] - a[:, 1]) * 1000,
    }
    stats = {}
    for name, arr in stages.items():
        stats[name] = (float(np.median(arr)), float(np.percentile(arr, 95)),
                       float(np.max(arr)))
    if len(a) >= 2:
        dt = np.diff(a[:, 1])
        stats["achieved_hz"] = (float(1.0 / np.median(dt)),
                                float(1.0 / np.percentile(dt, 95)),
                                float(1.0 / np.max(dt)))
    return stats


def print_stats(stats, frames, held_frames):
    if not stats:
        print("[realtime] no frames logged")
        return
    print("=" * 66)
    print(f"[realtime] {frames} frames processed, {held_frames} with no body "
          f"(held last pose)")
    for name, (med, p95, mx) in stats.items():
        unit = "Hz" if name == "achieved_hz" else "ms"
        print(f"  {name:20s}  median={med:7.2f}{unit}  p95={p95:7.2f}{unit}  "
              f"max={mx:7.2f}{unit}")


def write_latency_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t0_grab", "t1_pose", "t2_ik", "t3_publish"])
        w.writerows(rows)
    print(f"[realtime] latency log -> {path}")


def wait_enter(prompt):
    """Blocking prompt; returns True on ENTER, False on Ctrl+C."""
    try:
        input(prompt)
        return True
    except (KeyboardInterrupt, EOFError):
        return False


def stream(args, arm_sdk, zed, sl, fks, limits, index_map, gravity):
    """The 50-Hz control loop. Returns (rows, frames, held_frames)."""
    bodies = sl.Bodies()
    runtime = sl.BodyTrackingRuntimeParameters()
    runtime.detection_confidence_threshold = args.min_conf

    euro = OneEuro(min_cutoff=args.euro_min_cutoff, beta=args.euro_beta)
    q_prev = {s: np.zeros(4) for s in SIDES}
    pose_prev = HOME_POSE[:]

    rows, held_frames = [], 0
    frame_idx = 0
    last_t = time.monotonic()

    print("[realtime] streaming (Ctrl+C to stop)")
    while not _stop:
        grab_err = zed.grab()
        if grab_err != sl.ERROR_CODE.SUCCESS:
            if grab_err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            print(f"[realtime] grab error: {grab_err}", file=sys.stderr)
            continue
        t0 = time.monotonic()
        zed.retrieve_bodies(bodies, runtime)
        body = pick_primary_body(bodies)
        j = body_to_joints_dict(body, index_map)
        R = torso_frame(j) if j else None

        arms_ok = R is not None
        if arms_ok:
            targets_per_side = {}
            for side in SIDES:
                ad = arm_directions(j, R, side)
                if ad is None:
                    arms_ok = False
                    break
                u_hat, f_hat, flex_deg = ad
                targets_per_side[side] = (u_hat, f_hat, flex_deg)

        t1 = time.monotonic()

        if arms_ok:
            solved = {}
            for side in SIDES:
                u_hat, f_hat, flex_deg = targets_per_side[side]
                lim4 = limits[side]
                q4, _res, _cl = solve_arm(
                    fks[side], u_hat, f_hat, q_prev[side], lim4,
                    suppress_yaw=(flex_deg < STRAIGHT_ARM_DEG),
                    yaw_reg=args.yaw_reg, yaw_neutral=args.yaw_neutral)
                solved[side] = q4
                q_prev[side] = q4
            target_pose = assemble_pose(solved)
            target_pose = euro(target_pose, t=t1).tolist()
        else:
            held_frames += 1
            target_pose = pose_prev  # hold last commanded pose

        dt = max(t1 - last_t, 1e-3)
        last_t = t1
        pose_cmd = clamp_velocity(pose_prev, target_pose, args.max_vel, dt)
        t2 = time.monotonic()

        arm_sdk.publish(pose_cmd)
        pose_prev = pose_cmd
        t3 = time.monotonic()

        rows.append((frame_idx, t0, t1, t2, t3))
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    return rows, frame_idx, held_frames


def wait_for_person(zed, sl, index_map, args):
    """Return once we've seen a body in N consecutive frames (or Ctrl+C)."""
    bodies = sl.Bodies()
    runtime = sl.BodyTrackingRuntimeParameters()
    runtime.detection_confidence_threshold = args.min_conf
    print(f"[realtime] waiting for a person in the ZED FoV "
          f"(need {args.warmup_frames} consecutive detections)...")
    hit = 0
    while not _stop and hit < args.warmup_frames:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.02)
            continue
        zed.retrieve_bodies(bodies, runtime)
        body = pick_primary_body(bodies)
        j = body_to_joints_dict(body, index_map)
        hit = hit + 1 if torso_frame(j) is not None else 0
    if not _stop:
        print(f"[realtime] person detected ({hit} consecutive frames)")


def run(args):
    global _stop
    signal.signal(signal.SIGINT, _sigint)

    import pyzed.sl as sl

    # ── URDF / IK setup ─────────────────────────────────────────────────────
    if not os.path.isfile(args.urdf):
        sys.exit(f"[realtime] URDF not found: {args.urdf}")
    fks = {s: ArmChainFK(args.urdf, s) for s in SIDES}
    all_limits = load_limits(args.urdf)
    limits = {s: [all_limits[f"{s}_shoulder_pitch_joint"],
                  all_limits[f"{s}_shoulder_roll_joint"],
                  all_limits[f"{s}_shoulder_yaw_joint"],
                  all_limits[f"{s}_elbow_joint"]] for s in SIDES}

    # ── Gravity feedforward (default ON, frozen gain) ───────────────────────
    gravity = {}
    if args.gravity_ff:
        for s in SIDES:
            gravity[s] = GravityFF(args.urdf, s)

    # ── Open ZED + body tracking (FAST for latency) ─────────────────────────
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"[realtime] ZED open failed: {err}")

    try:
        if args.record:
            rec = sl.RecordingParameters(args.record, sl.SVO_COMPRESSION_MODE.H264)
            rerr = zed.enable_recording(rec)
            if rerr != sl.ERROR_CODE.SUCCESS:
                sys.exit(f"[realtime] enable_recording failed: {rerr}")
            print(f"[realtime] recording SVO to: {args.record}")

        pos = sl.PositionalTrackingParameters()
        pos.set_as_static = args.static
        zed.enable_positional_tracking(pos)

        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = False
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        body_params.body_format = getattr(sl.BODY_FORMAT, f"BODY_{args.body_format}")
        zed.enable_body_tracking(body_params)

        idx_map = canonical_index_map(body_params.body_format)

        if not args.allow_no_body:
            wait_for_person(zed, sl, idx_map, args)
        if _stop:
            return 0

        # ── Arm sdk: snapshot, fade in, go to home ──────────────────────────
        arm_sdk = ArmSdk(OUT_JOINTS, args.interface, args.dry_run,
                         gravity=gravity, gravity_gain=args.gravity_gain)
        snapshot = arm_sdk.snapshot()
        arm_sdk.pose = snapshot
        print(f"[realtime] arm snapshot: "
              f"L_ShP={snapshot[0]:+.2f} R_ShP={snapshot[7]:+.2f}")

        if not wait_enter("[realtime] ENTER to arm arm_sdk (Ctrl+C to abort)... "):
            print("[realtime] aborted before arming")
            return 0

        print("[realtime] fading weight 0 -> 1 (2 s)")
        arm_sdk.ramp_weight(1.0, 2.0)
        print("[realtime] ramping to home pose (3 s)")
        arm_sdk.ramp_pose(HOME_POSE, 3.0)

        if not wait_enter("[realtime] ENTER to start streaming (Ctrl+C safe)... "):
            print("[realtime] aborted before streaming")
        else:
            rows, frames, held = stream(args, arm_sdk, zed, sl, fks, limits,
                                        idx_map, gravity)
            if args.latency_log:
                write_latency_csv(rows, args.latency_log)
            if args.print_stats:
                print_stats(latency_stats(rows), frames, held)

        # ── Safe shutdown ───────────────────────────────────────────────────
        print("[realtime] ramping back to snapshot (3 s)")
        arm_sdk.ramp_pose(snapshot, 3.0)
        print("[realtime] fading weight 1 -> 0 (2 s)")
        arm_sdk.ramp_weight(0.0, 2.0)
        return 0
    finally:
        if args.record:
            zed.disable_recording()
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()


def self_test():
    """No ZED / DDS / URDF — exercise helpers only."""
    # assemble_pose
    pose = assemble_pose({"left": [0.1, 0.2, 0.3, 0.4],
                          "right": [-0.1, -0.2, -0.3, -0.4]})
    assert len(pose) == 15, len(pose)
    assert pose[:4] == [0.1, 0.2, 0.3, 0.4], pose[:4]
    assert pose[4:7] == [0.0, 0.0, 0.0], pose[4:7]
    assert pose[7:11] == [-0.1, -0.2, -0.3, -0.4], pose[7:11]
    assert pose[14] == 0.0

    # clamp_velocity: no prev -> passthrough
    assert clamp_velocity(None, [0.5], max_vel=4.0, dt=0.02) == [0.5]
    # clamp_velocity: 10 rad/s delta over 20 ms should clamp to 4*0.02=0.08 rad
    out = clamp_velocity([0.0], [1.0], max_vel=4.0, dt=0.02)
    assert abs(out[0] - 0.08) < 1e-9, out

    # body_to_joints_dict: NaN drop
    class FB: pass
    fb = FB()
    fb.keypoint = np.array([[0.1, 0.2, 0.3],
                            [float("nan"), 0.0, 0.0]])
    fb.keypoint_confidence = np.array([80.0, 50.0])
    d = body_to_joints_dict(fb, {0: "left_shoulder", 1: "left_elbow"})
    assert list(d.keys()) == ["left_shoulder"], d

    # latency_stats
    rows = [(0, 0.000, 0.010, 0.020, 0.030),
            (1, 0.020, 0.028, 0.036, 0.041),
            (2, 0.040, 0.049, 0.055, 0.062)]
    s = latency_stats(rows)
    assert "end-to-end" in s and "achieved_hz" in s
    assert s["end-to-end"][0] > 0

    print("[self-test] OK — assemble_pose, clamp_velocity, body dict, latency stats")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"))
    ap.add_argument("--interface", default="",
                    help="robot network interface (e.g. enp128s31f6)")
    ap.add_argument("--body-format", choices=["18", "34", "38"], default="38")
    ap.add_argument("--resolution",
                    choices=["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"],
                    default="HD720", help="lower = faster grab (default HD720)")
    ap.add_argument("--min-conf", type=int, default=40)
    ap.add_argument("--static", action="store_true",
                    help="camera stationary; better positional-tracking perf")
    ap.add_argument("--record", default="",
                    help="also record the live stream to this SVO2 path")

    ap.add_argument("--yaw-neutral", type=float, default=0.02,
                    help="frozen 0.02 (branch-select toward natural yaw)")
    ap.add_argument("--yaw-reg", type=float, default=0.05,
                    help="frozen 0.05 (continuity)")
    ap.add_argument("--euro-min-cutoff", type=float, default=1.0)
    ap.add_argument("--euro-beta", type=float, default=0.5)
    ap.add_argument("--max-vel", type=float, default=4.0,
                    help="per-joint rate limit (rad/s)")

    ap.add_argument("--gravity-ff", action=argparse.BooleanOptionalAction,
                    default=True, help="gravity feedforward (default ON)")
    ap.add_argument("--gravity-gain", type=float, default=FROZEN_GRAVITY_GAIN)

    ap.add_argument("--warmup-frames", type=int, default=10,
                    help="require this many consecutive body detections before arming")
    ap.add_argument("--allow-no-body", action="store_true",
                    help="skip the person-detection warmup (dev only)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = no limit)")

    ap.add_argument("--latency-log", default="",
                    help="write t0..t3 CSV to this path")
    ap.add_argument("--print-stats", action="store_true", default=True,
                    help="print latency median/p95 at exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="no DDS: ZED + IK + logging only, no arm_sdk publish")
    ap.add_argument("--self-test", action="store_true")

    args = ap.parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
