#!/usr/bin/env python3
"""zed_to_joints.py - ZED SDK body tracking to canonical joints CSV.

Sibling to ../../../experiments/scripts/bag_to_joints.py — emits the same
`frame,joint,x,y,z,conf` long CSV, so a ZED capture drops into the existing
retarget / replay pipeline (retarget_arm.py, angles.py, rula_reba_score.py,
replay_arm.py) with no downstream changes.

Two sources:
  <path>.svo2   process an SVO2 recording (positional arg)
  (omitted)     open the live ZED camera

Body format defaults to BODY_38 (richest joint set: adds spine, neck, feet
beyond the 12 canonical). BODY_18 / BODY_34 also supported — all three cover
the 12 canonical joints (L/R shoulder, elbow, wrist, hip, knee, ankle) that
the retargeter and RULA scorer read.

Coordinate system: RIGHT_HANDED_Z_UP_X_FWD (metres), matching the robot
convention. Downstream retargeter builds its own torso frame from
shoulders/hips so any consistent frame works, but flag deviations if you
ever cross-compare against an iPhone capture.

Usage:
    # offline: process an SVO2 file into a joints CSV
    /usr/bin/python3 zed_to_joints.py capture.svo2 --out joints_zed.csv

    # live: record SVO2 while streaming, so the same capture is re-processable
    /usr/bin/python3 zed_to_joints.py --record capture.svo2 --out joints_zed.csv

    # self-test: no ZED SDK / camera needed
    /usr/bin/python3 zed_to_joints.py --self-test

Requires: pyzed.sl (install via /usr/local/zed/get_python_api.py). Run with
system /usr/bin/python3, matching the rest of the experiment tooling.
"""

import argparse
import csv
import os
import sys


# Canonical joint names — must match bag_to_joints.py MP_INDEX_TO_JOINT so
# downstream tools inner-join by joint name across sensors transparently.
CANONICAL_JOINTS = (
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
)

# ZED enum-member name → canonical name. Enum names are identical across
# BODY_18 / BODY_34 / BODY_38, so we look up by NAME at runtime rather than
# hard-coding integer indices (which shift between formats).
ZED_NAME_TO_CANONICAL = {
    "LEFT_SHOULDER":  "left_shoulder",
    "RIGHT_SHOULDER": "right_shoulder",
    "LEFT_ELBOW":     "left_elbow",
    "RIGHT_ELBOW":    "right_elbow",
    "LEFT_WRIST":     "left_wrist",
    "RIGHT_WRIST":    "right_wrist",
    "LEFT_HIP":       "left_hip",
    "RIGHT_HIP":      "right_hip",
    "LEFT_KNEE":      "left_knee",
    "RIGHT_KNEE":     "right_knee",
    "LEFT_ANKLE":     "left_ankle",
    "RIGHT_ANKLE":    "right_ankle",
}


def canonical_index_map(body_format):
    """{zed_int_index: canonical_name} for the chosen BODY_FORMAT."""
    import pyzed.sl as sl
    parts_enum = {
        sl.BODY_FORMAT.BODY_18: sl.BODY_18_PARTS,
        sl.BODY_FORMAT.BODY_34: sl.BODY_34_PARTS,
        sl.BODY_FORMAT.BODY_38: sl.BODY_38_PARTS,
    }[body_format]
    mapping = {}
    for member in parts_enum:
        canonical = ZED_NAME_TO_CANONICAL.get(member.name)
        if canonical is not None:
            mapping[int(member.value)] = canonical
    missing = set(CANONICAL_JOINTS) - set(mapping.values())
    if missing:
        raise RuntimeError(
            f"BODY_FORMAT missing canonical joints: {sorted(missing)}")
    return mapping


def pick_primary_body(bodies):
    """Highest-confidence body from a Bodies result. Single-subject captures
    are the norm here, so this is deliberately simple."""
    best = None
    best_conf = -1.0
    for body in bodies.body_list:
        conf = float(getattr(body, "confidence", 0.0) or 0.0)
        if conf > best_conf:
            best_conf = conf
            best = body
    return best


def body_to_rows(body, frame_idx, index_map):
    """(frame, joint, x, y, z, conf) tuples for one body's canonical joints.
    Drops keypoints that ZED reports as NaN (undetected)."""
    rows = []
    if body is None:
        return rows
    kp = body.keypoint
    kp_conf = body.keypoint_confidence
    for zed_idx, canonical in index_map.items():
        if zed_idx >= len(kp):
            continue
        x, y, z = float(kp[zed_idx][0]), float(kp[zed_idx][1]), float(kp[zed_idx][2])
        if x != x or y != y or z != z:
            continue
        conf = float(kp_conf[zed_idx]) / 100.0 if zed_idx < len(kp_conf) else 0.0
        rows.append((frame_idx, canonical, x, y, z, conf))
    return rows


def run(args):
    import pyzed.sl as sl

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

    if args.source_path:
        if not args.source_path.lower().endswith((".svo", ".svo2")):
            print(f"[zed] source must be .svo or .svo2 (got {args.source_path})",
                  file=sys.stderr)
            return 2
        init.set_from_svo_file(args.source_path)
        print(f"[zed] SVO input: {args.source_path}")
    else:
        print("[zed] live camera")

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[zed] open failed: {err}", file=sys.stderr)
        return 2

    try:
        if args.record:
            rec = sl.RecordingParameters(args.record, sl.SVO_COMPRESSION_MODE.H264)
            rerr = zed.enable_recording(rec)
            if rerr != sl.ERROR_CODE.SUCCESS:
                print(f"[zed] enable_recording failed: {rerr}", file=sys.stderr)
                return 2
            print(f"[zed] recording SVO to: {args.record}")

        pos = sl.PositionalTrackingParameters()
        pos.set_as_static = args.static
        zed.enable_positional_tracking(pos)

        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = False
        body_params.detection_model = getattr(
            sl.BODY_TRACKING_MODEL, f"HUMAN_BODY_{args.model.upper()}")
        body_params.body_format = getattr(sl.BODY_FORMAT, f"BODY_{args.body_format}")
        zed.enable_body_tracking(body_params)

        runtime = sl.BodyTrackingRuntimeParameters()
        runtime.detection_confidence_threshold = args.min_conf

        idx_map = canonical_index_map(body_params.body_format)

        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        bodies = sl.Bodies()
        grabbed = processed = frames_with_body = total_rows = 0
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "joint", "x", "y", "z", "conf"])
            try:
                while True:
                    grab_err = zed.grab()
                    if grab_err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                        break
                    if grab_err != sl.ERROR_CODE.SUCCESS:
                        print(f"[zed] grab error: {grab_err}", file=sys.stderr)
                        break
                    grabbed += 1
                    if (grabbed - 1) % args.stride != 0:
                        continue
                    zed.retrieve_bodies(bodies, runtime)
                    body = pick_primary_body(bodies)
                    rows = body_to_rows(body, processed, idx_map)
                    if rows:
                        writer.writerows(rows)
                        frames_with_body += 1
                        total_rows += len(rows)
                    processed += 1
                    if args.max_frames and processed >= args.max_frames:
                        break
            except KeyboardInterrupt:
                print("\n[zed] interrupted", file=sys.stderr)

        print(f"[zed] grabbed={grabbed} processed={processed} "
              f"frames_with_body={frames_with_body} rows={total_rows} "
              f"body_format=BODY_{args.body_format} out={args.out}")
        return 0 if total_rows > 0 else 1
    finally:
        if args.record:
            zed.disable_recording()
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()


def self_test():
    """Exercise mapping + CSV writer without pyzed / camera."""
    import tempfile
    import numpy as np

    class FakeBody:
        pass

    fb = FakeBody()
    fb.keypoint = np.array([
        [0.10,  0.05, 1.20],   # 0
        [0.15, -0.10, 1.10],   # 1
        [0.20, -0.25, 1.05],   # 2
        [float("nan"), 0.0, 0.0],  # 3: undetected → must be dropped
    ])
    fb.keypoint_confidence = np.array([80.0, 60.0, 40.0, 50.0])
    fb.confidence = 75.0

    idx_map = {
        0: "left_shoulder", 1: "left_elbow",
        2: "left_wrist",    3: "left_hip",
    }
    rows = body_to_rows(fb, frame_idx=7, index_map=idx_map)
    assert len(rows) == 3, f"NaN drop failed, got {len(rows)} rows"
    assert rows[0] == (7, "left_shoulder", 0.10, 0.05, 1.20, 0.8), rows[0]
    assert abs(rows[1][5] - 0.6) < 1e-9

    class FakeBodies:
        def __init__(self, bl):
            self.body_list = bl

    low = FakeBody(); low.confidence = 30.0
    high = fb
    picked = pick_primary_body(FakeBodies([low, high]))
    assert picked is high, "pick_primary_body must pick highest confidence"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        w = csv.writer(tf)
        w.writerow(["frame", "joint", "x", "y", "z", "conf"])
        w.writerows(rows)
        path = tf.name
    with open(path) as f:
        header = f.readline().strip().split(",")
        first = f.readline().strip().split(",")
    os.unlink(path)
    assert header == ["frame", "joint", "x", "y", "z", "conf"], header
    assert first[:2] == ["7", "left_shoulder"], first

    print("[self-test] OK — mapping, NaN drop, conf rescale, primary-body pick, CSV round-trip")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source_path", nargs="?", default="",
                    help="SVO2 file path; omit for live camera")
    ap.add_argument("--out", help="output CSV path (required unless --self-test)")
    ap.add_argument("--body-format", choices=["18", "34", "38"], default="38")
    ap.add_argument("--model", choices=["fast", "medium", "accurate"],
                    default="accurate", help="ZED body-tracking model")
    ap.add_argument("--resolution",
                    choices=["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"],
                    default="HD1080")
    ap.add_argument("--min-conf", type=int, default=40,
                    help="ZED detection confidence threshold (0-100)")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth grabbed frame")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N processed frames (0 = no limit)")
    ap.add_argument("--static", action="store_true",
                    help="camera is stationary (tripod) — better tracking perf")
    ap.add_argument("--record", default="",
                    help="live mode: also record SVO2 to this path")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.out:
        ap.error("--out is required (unless --self-test)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
