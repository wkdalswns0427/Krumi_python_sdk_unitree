#!/usr/bin/env python3
"""bag_to_joints.py - RGB(-D) capture to per-frame 3D joints for Block 1.

Produces the per-frame joint-position files that block1_compare.py consumes.
Two modes, one per pipeline, run on the SAME capture so frame indices line up:

  --mode mono : the MONOCULAR estimate. MediaPipe Pose 'world' landmarks
                (the model's own metric 3D, root-relative, metres). No depth.

  --mode rgbd : the METRIC REFERENCE. MediaPipe 2D image landmarks are
                back-projected through the depth image using the camera
                intrinsics (pin-hole deprojection) into metric 3D.

Two capture sources, selected with --source:

  --source rosbag   (default) a ROS2 bag with color + aligned depth +
                    camera_info topics (the robot-mounted D435i).

  --source rgbd-dir an RGBD capture directory, e.g. an iPhone Pro / iPad Pro
                    LiDAR recording exported by Record3D or Stray Scanner:
                      color:      <dir>/rgb.mp4  or  <dir>/rgb/*.png|jpg
                      depth:      <dir>/depth/*.png (16-bit mm) or *.npy (float m)
                      intrinsics: <dir>/intrinsics.json ({fx,fy,cx,cy} or 3x3 K)
                                  or camera_matrix.csv
                    Paths auto-detect but override with --rgb / --depth /
                    --intrinsics. Depth is resized to the color resolution
                    (nearest) and assumed aligned to color.

Output is the canonical long CSV (frame,joint,x,y,z,conf) documented in
README.md. The exported landmark set is selected with --joints:
  arm  : the 6 retargeted joints (L/R shoulder, elbow, wrist) plus L/R hip
         (hips ride along so block1_compare can derive the pelvis root)
  full : arm set + L/R knee, L/R ankle (12 total) for the RULA/REBA scorer.
         Default. Same MediaPipe pass, zero extra cost; in rgbd mode every
         landmark goes through the identical confidence-gated depth sampling
         path. Face landmarks (nose, ears) are deliberately not exported;
         the scorer uses a neutral-neck assumption.

IMPORTANT: the two modes are in DIFFERENT coordinate frames (mono is
MediaPipe's person frame, rgbd is the camera frame), so compare them with
    block1_compare.py mono.csv rgbd.csv --procrustes
(standard "relative 3D error" / PA-MPJPE family). See README.md.

Dependencies (offline, no ROS install needed): pip3 install --user rosbags
mediapipe. cv2 + numpy already present. Keep numpy below 2.

Usage:
    # ROS2 bag (D435i):
    python3 bag_to_joints.py BAGDIR --mode rgbd --out joints/rgbd_M1_S1_R1.csv
    # iPhone LiDAR capture directory:
    python3 bag_to_joints.py CAPDIR --source rgbd-dir --mode rgbd \
        --out joints/rgbd_M1_S1_R1.csv
    python3 bag_to_joints.py --self-test          # no capture / mediapipe needed
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


# ── Canonical joints (indices match block1_compare.py / MediaPipe Pose) ──────
# Face landmarks (0 nose, 7/8 ears) are deliberately EXCLUDED from the export
# (user decision 2026-07-09). Consequence: neck_flexion cannot be derived, so
# the RULA/REBA scorer applies a neutral-neck assumption (see angles.py).
MP_INDEX_TO_JOINT = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow",    14: "right_elbow",
    15: "left_wrist",    16: "right_wrist",
    23: "left_hip",      24: "right_hip",
    25: "left_knee",     26: "right_knee",
    27: "left_ankle",    28: "right_ankle",
}

# Joint-set selection (--joints). "arm" is the pre-extension behavior: the 6
# retargeted joints PLUS the hips, which have always ridden along so that
# block1_compare can derive the pelvis root. "full" adds the legs for the
# RULA/REBA scorer. Retargeting keeps consuming only the 6 arm joints.
ARM_JOINT_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
FULL_JOINT_INDICES = sorted(MP_INDEX_TO_JOINT)
JOINT_SETS = {"arm": ARM_JOINT_INDICES, "full": FULL_JOINT_INDICES}


# ── Core geometry / IO (pure numpy; unit-testable without ROS/MediaPipe) ─────
def deproject(u, v, z, fx, fy, cx, cy):
    """Pixel (u,v) + metric depth z -> 3D point in the camera optical frame.

    Standard pin-hole model, same convention as rs2_deproject_pixel_to_point:
        X = (u - cx) * z / fx,  Y = (v - cy) * z / fy,  Z = z
    """
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)


def sample_depth(depth, conf, u, v, radius, scale, conf_min, spread_max_mm,
                 pick="median", search_radius=0):
    """Metric depth at native-resolution pixel (u, v) from a (2*radius+1) patch.

    Keeps non-zero pixels, and (when conf is given) only those with confidence
    >= conf_min. Then:
      pick='median': median of survivors, but REJECT if the in-patch spread
                     exceeds spread_max_mm (a thin limb over a far background).
      pick='nearest': the closest surviving surface (the limb when it straddles
                      limb/background), which keeps extended-limb joints instead
                      of rejecting them. Spread is not a reject here.
    If the patch has NO accepted pixel and search_radius > radius, search the
    wider (2*search_radius+1) window and take the accepted pixel nearest to the
    landmark (pixel distance). Returns (z_metres, reason, recovered): z is None
    with a reason string when rejected; recovered=True when the wider search
    supplied the value.
    """
    h, w = depth.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None, "oob", False
    u0, u1 = max(0, ui - radius), min(w, ui + radius + 1)
    v0, v1 = max(0, vi - radius), min(h, vi + radius + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float64)
    keep = patch > 0
    if conf is not None:
        keep &= (conf[v0:v1, u0:u1] >= conf_min)
    surv = patch[keep]
    if surv.size == 0:
        if search_radius > radius:
            s0u, s1u = max(0, ui - search_radius), min(w, ui + search_radius + 1)
            s0v, s1v = max(0, vi - search_radius), min(h, vi + search_radius + 1)
            wide = depth[s0v:s1v, s0u:s1u].astype(np.float64)
            wkeep = wide > 0
            if conf is not None:
                wkeep &= (conf[s0v:s1v, s0u:s1u] >= conf_min)
            ys, xs = np.nonzero(wkeep)
            if ys.size:
                d2 = (ys + s0v - vi) ** 2 + (xs + s0u - ui) ** 2
                k = int(np.argmin(d2))
                return float(wide[ys[k], xs[k]]) * scale, None, True
        return None, "no_valid_depth", False
    if pick == "nearest":
        return float(surv.min()) * scale, None, False
    if spread_max_mm and (surv.max() - surv.min()) * scale * 1000.0 > spread_max_mm:
        return None, "spread", False
    return float(np.median(surv)) * scale, None, False


def rows_from_world(world_lms, frame_idx, indices):
    """MediaPipe world landmarks -> canonical CSV rows (mono mode).

    world_lms: dict {mp_index: (x, y, z, visibility)} in metres.
    indices: which MediaPipe landmark indices to emit (JOINT_SETS entry).
    """
    rows = []
    for idx in indices:
        lm = world_lms.get(idx)
        if lm is None:
            continue
        x, y, z, vis = lm
        rows.append((frame_idx, MP_INDEX_TO_JOINT[idx], x, y, z, vis))
    return rows


JUMP_RELOCK_AFTER = 5   # consecutive jump rejections before the gate re-locks

# Sampling policies (--depth-policy). Numbers are min accepted confidence.
# 'limb' joints are thin/fast structures where LiDAR conf-2 is sparse; root
# joints (shoulders, hips) stay strict so the skeleton frame stays clean.
LIMB_JOINTS = {"left_elbow", "right_elbow", "left_wrist", "right_wrist",
               "left_knee", "right_knee", "left_ankle", "right_ankle",
               "nose", "left_ear", "right_ear"}
DEPTH_POLICIES = ("strict", "relaxed-limb", "relaxed-all", "search")


def build_policy(policy, override_conf_min=None):
    """-> (conf_min_by_joint, search_radius). override applies to all joints."""
    names = list(MP_INDEX_TO_JOINT.values())
    if policy == "strict":
        conf_by = {n: 2 for n in names}
        search = 0
    elif policy == "relaxed-limb":
        conf_by = {n: (1 if n in LIMB_JOINTS else 2) for n in names}
        search = 0
    elif policy == "relaxed-all":
        conf_by = {n: 1 for n in names}
        search = 0
    elif policy == "search":
        conf_by = {n: (1 if n in LIMB_JOINTS else 2) for n in names}
        search = 2
    else:
        raise ValueError(f"unknown depth policy '{policy}'")
    if override_conf_min is not None:
        conf_by = {n: override_conf_min for n in names}
    return conf_by, search


def rows_from_depth(image_lms, depth, conf, K, frame_idx, color_wh, cfg, track,
                    indices):
    """MediaPipe image landmarks + native depth -> canonical rgbd rows.

    image_lms: {mp_index: (u_norm, v_norm, visibility)}, normalised [0,1].
    color_wh: colour (W, H) the landmarks are normalised to; K is the colour
    intrinsics. Depth may be lower resolution (e.g. iPhone LiDAR 256x192): it is
    sampled at the native depth pixel while x/y use the colour pixel.

    cfg keys: radius, scale, conf_min_by (int or {joint: int}), spread_max_mm,
    pick, search_radius, jump_speed (m/s, 0 disables), jump_fix_mm (overrides
    speed when set), fps. The jump threshold is dt-aware: max plausible joint
    speed times the actual time between the compared frames, so detection gaps
    loosen the gate naturally.

    track: {joint: [z, frame_idx, streak]} persistent across frames.
    Returns (rows, stats) where stats counts rejections plus 'recovered'.
    """
    fx, fy, cx, cy = K
    cw, ch = color_wh
    dh, dw = depth.shape[:2]
    conf_by = cfg["conf_min_by"]
    uniform = not isinstance(conf_by, dict)
    rows, stats = [], {}
    for idx in indices:
        lm = image_lms.get(idx)
        if lm is None:
            continue
        u_norm, v_norm, vis = lm
        joint = MP_INDEX_TO_JOINT[idx]
        cmin = conf_by if uniform else conf_by[joint]
        z, reason, recovered = sample_depth(
            depth, conf, u_norm * dw, v_norm * dh, cfg["radius"], cfg["scale"],
            cmin, cfg["spread_max_mm"], cfg["pick"], cfg["search_radius"])
        if z is None:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        if recovered:
            stats["recovered"] = stats.get("recovered", 0) + 1
        prev = track.get(joint)
        if prev is not None:
            if cfg["jump_fix_mm"]:
                thr_mm = cfg["jump_fix_mm"]
            elif cfg["jump_speed"]:
                dt = (frame_idx - prev[1]) / cfg["fps"]
                thr_mm = cfg["jump_speed"] * 1000.0 * max(dt, 1e-6)
            else:
                thr_mm = None
            if thr_mm is not None and abs(z - prev[0]) * 1000.0 > thr_mm:
                streak = prev[2] + 1
                if streak < JUMP_RELOCK_AFTER:
                    # Sporadic jump: reject the sample, keep the lock.
                    prev[2] = streak
                    stats["jump"] = stats.get("jump", 0) + 1
                    continue
                # Many consecutive far readings: the original lock was wrong
                # (or the joint truly moved). Re-lock instead of rejecting
                # forever.
                stats["jump_relock"] = stats.get("jump_relock", 0) + 1
        track[joint] = [z, frame_idx, 0]
        p = deproject(u_norm * cw, v_norm * ch, z, fx, fy, cx, cy)
        rows.append((frame_idx, joint, p[0], p[1], p[2], vis))
    return rows, stats


def write_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "joint", "x", "y", "z", "conf"])
        for fi, joint, x, y, z, c in rows:
            w.writerow([fi, joint, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{c:.4f}"])


# ── Image decoding ───────────────────────────────────────────────────────────
def decode_color(msg):
    """sensor_msgs/Image (rgb8/bgr8) -> HxWx3 uint8 RGB."""
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    if enc == "bgr8":
        arr = arr[:, :, ::-1]
    elif enc != "rgb8":
        sys.exit(f"unsupported color encoding '{msg.encoding}' (want rgb8/bgr8)")
    return np.ascontiguousarray(arr)


def decode_depth(msg):
    """sensor_msgs/Image (16UC1/mono16) -> HxW uint16 (raw depth units)."""
    enc = msg.encoding.lower()
    if enc not in ("16uc1", "mono16"):
        sys.exit(f"unsupported depth encoding '{msg.encoding}' (want 16UC1)")
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)


# ── MediaPipe pose detector (lazy import) ────────────────────────────────────
class MediaPipeDetector:
    def __init__(self, model_complexity=1, min_det=0.5, indices=None):
        import mediapipe as mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=model_complexity,
            enable_segmentation=False, min_detection_confidence=min_det)
        self.indices = FULL_JOINT_INDICES if indices is None else indices

    def detect(self, rgb):
        """Returns (image_lms, world_lms) dicts keyed by MP index, or (None, None)."""
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return None, None
        image_lms, world_lms = {}, {}
        for idx in self.indices:
            il = res.pose_landmarks.landmark[idx]
            image_lms[idx] = (il.x, il.y, float(il.visibility))
            if res.pose_world_landmarks:
                wl = res.pose_world_landmarks.landmark[idx]
                world_lms[idx] = (wl.x, wl.y, wl.z, float(wl.visibility))
        return image_lms, world_lms

    def close(self):
        self._pose.close()


# ── Rosbag reading (lazy import of rosbags) ──────────────────────────────────
def read_camera_info(reader, info_topic):
    from rosbags.highlevel import AnyReader  # noqa: F401 (type only)
    conns = [c for c in reader.connections if c.topic == info_topic]
    if not conns:
        sys.exit(f"camera_info topic '{info_topic}' not in bag")
    for conn, _, raw in reader.messages(connections=conns):
        m = reader.deserialize(raw, conn.msgtype)
        k = m.k  # ROS2 CameraInfo 3x3 row-major
        return (float(k[0]), float(k[4]), float(k[2]), float(k[5]))
    sys.exit(f"no camera_info messages on '{info_topic}'")


def iter_frames(reader, color_topic, depth_topic, need_depth):
    """Yield (frame_idx, rgb, depth_or_None), pairing each color frame with the
    most recent depth frame (aligned_depth_to_color is ~simultaneous)."""
    topics = [color_topic] + ([depth_topic] if need_depth else [])
    conns = [c for c in reader.connections if c.topic in topics]
    if not any(c.topic == color_topic for c in conns):
        sys.exit(f"color topic '{color_topic}' not in bag")
    if need_depth and not any(c.topic == depth_topic for c in conns):
        sys.exit(f"depth topic '{depth_topic}' not in bag")

    latest_depth = None
    idx = 0
    for conn, _, raw in reader.messages(connections=conns):
        m = reader.deserialize(raw, conn.msgtype)
        if conn.topic == depth_topic:
            latest_depth = decode_depth(m)
        elif conn.topic == color_topic:
            if need_depth and latest_depth is None:
                continue                   # wait until we have a depth frame
            yield idx, decode_color(m), latest_depth, None   # no per-pixel conf
            idx += 1


class FakeDetector:
    """Returns fixed landmarks for every frame (self-test only, no MediaPipe)."""

    def __init__(self, image_lms, world_lms):
        self._il, self._wl = image_lms, world_lms

    def detect(self, rgb):
        return self._il, self._wl

    def close(self):
        pass


def depth_cfg_from_args(args, scale, fps):
    """Build the rows_from_depth sampling config from CLI args."""
    conf_by, search = build_policy(args.depth_policy, args.depth_conf_min)
    return dict(radius=args.depth_window, scale=scale, conf_min_by=conf_by,
                spread_max_mm=args.depth_spread_max, pick=args.depth_pick,
                search_radius=search, jump_speed=args.max_joint_speed,
                jump_fix_mm=args.depth_jump_max, fps=fps)


def process_frames(args, detector, K, frame_iter, need_depth, scale, fps):
    """Shared per-frame loop over a source yielding (idx, rgb, depth, conf)."""
    all_rows, n_frames, n_nopose, img_wh = [], 0, 0, None
    track, rejects = {}, {}
    cfg = depth_cfg_from_args(args, scale, fps) if need_depth else None
    for idx, rgb, depth, conf in frame_iter:
        if args.stride > 1 and idx % args.stride:
            continue
        if args.max_frames and n_frames >= args.max_frames:
            break                       # limit frames PROCESSED, pose or not
        n_frames += 1
        if img_wh is None:
            img_wh = (rgb.shape[1], rgb.shape[0])
        image_lms, world_lms = detector.detect(rgb)
        if image_lms is None:
            n_nopose += 1
            continue
        if args.mode == "mono":
            all_rows += rows_from_world(world_lms, idx, args.joint_indices)
        else:
            rws, rj = rows_from_depth(image_lms, depth, conf, K, idx, img_wh,
                                      cfg, track, args.joint_indices)
            all_rows += rws
            for k, v in rj.items():
                rejects[k] = rejects.get(k, 0) + v
    return all_rows, n_frames, n_nopose, rejects


def run_rosbag(args, detector, need_depth):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from pathlib import Path

    scale = args.depth_scale if args.depth_scale is not None else 0.001
    # ROS2 bags reference standard message types by name without embedding their
    # definitions, so rosbags >= 0.10 needs an explicit typestore to decode them.
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([Path(args.input)], default_typestore=typestore) as reader:
        n_color = sum(c.msgcount for c in reader.connections
                      if c.topic == args.color_topic)
        dur = max((reader.end_time - reader.start_time) / 1e9, 1e-6)
        fps = args.fps or (n_color / dur if n_color else 30.0)
        K = read_camera_info(reader, args.info_topic) if need_depth else None
        frame_iter = iter_frames(reader, args.color_topic, args.depth_topic,
                                 need_depth)
        rows, nf, nnp, rej = process_frames(args, detector, K, frame_iter,
                                            need_depth, scale, fps)
    return rows, nf, nnp, K, scale, rej, dict(video=n_color, fps=fps)


# ── iPhone / RGBD-directory source ───────────────────────────────────────────
def _natural_key(path):
    import re
    m = re.findall(r"\d+", os.path.basename(path))
    return (int(m[-1]) if m else -1, path)


def _auto(root, names, want_dir):
    for n in names:
        cand = os.path.join(root, n)
        if (os.path.isdir(cand) if want_dir else os.path.isfile(cand)):
            return cand
    return None


def load_intrinsics(path):
    """fx,fy,cx,cy from JSON (fx/fy/cx/cy or a 3x3 K) or a 3x3 CSV (row-major)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f:
            d = json.load(f)
        if all(k in d for k in ("fx", "fy", "cx", "cy")):
            return (float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]))
        for key in ("K", "intrinsic_matrix", "camera_matrix", "intrinsics"):
            if key in d:
                k = np.array(d[key], dtype=float).flatten()
                return (k[0], k[4], k[2], k[5])
        sys.exit(f"{path}: JSON needs fx/fy/cx/cy or a 3x3 K")
    k = (np.loadtxt(path, delimiter=",") if ext == ".csv"
         else np.loadtxt(path)).flatten()
    return (k[0], k[4], k[2], k[5])


def _probe_depth_dtype(path):
    import cv2
    if path.lower().endswith(".npy"):
        return np.load(path).dtype
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).dtype


def _color_frames(rgb_path):
    import cv2
    import glob
    if os.path.isfile(rgb_path):                       # video file
        cap = cv2.VideoCapture(rgb_path)
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield np.ascontiguousarray(bgr[:, :, ::-1])   # BGR -> RGB
        cap.release()
    else:                                              # directory of images
        files = [f for f in sorted(glob.glob(os.path.join(rgb_path, "*")),
                                   key=_natural_key)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in files:
            bgr = cv2.imread(f, cv2.IMREAD_COLOR)
            yield np.ascontiguousarray(bgr[:, :, ::-1])


def _depth_files(depth_path):
    import glob
    return [f for f in sorted(glob.glob(os.path.join(depth_path, "*")),
                              key=_natural_key)
            if f.lower().endswith((".png", ".npy", ".exr", ".tif", ".tiff"))]


def _load_depth(path):
    import cv2
    d = np.load(path) if path.lower().endswith(".npy") \
        else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[:, :, 0]
    return d                                    # native resolution, no resize


def _load_conf(path, depth_shape):
    import cv2
    c = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if c is None:
        return None
    if c.ndim == 3:
        c = c[:, :, 0]
    if c.shape[:2] != depth_shape[:2]:          # keep conf aligned to depth res
        c = cv2.resize(c, (depth_shape[1], depth_shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    return c


def _resolve_rgbd(args, need_depth):
    root = args.input
    rgb = args.rgb or _auto(root, ["rgb.mp4", "color.mp4", "rgb.mov"], False) \
        or _auto(root, ["rgb", "color", "images"], True)
    if rgb is None:
        sys.exit(f"rgbd-dir: no color source under {root} (use --rgb)")
    depth = intr = confd = None
    if need_depth:
        depth = args.depth or _auto(root, ["depth", "depths", "depth16"], True)
        intr = args.intrinsics or _auto(
            root, ["intrinsics.json", "camera_matrix.csv", "camera_intrinsics.json",
                   "metadata.json"], False)
        confd = args.confidence or _auto(root, ["confidence", "conf"], True)
        if depth is None:
            sys.exit(f"rgbd-dir: no depth dir under {root} (use --depth)")
        if intr is None:
            sys.exit(f"rgbd-dir: no intrinsics under {root} (use --intrinsics)")
    return rgb, depth, intr, confd


def frames_from_rgbd_dir(rgb_path, depth_path, conf_path, need_depth):
    """Yield (idx, rgb, depth, conf): color[i] paired with depth[i]/conf[i]."""
    depth_list = _depth_files(depth_path) if need_depth else []
    conf_list = _depth_files(conf_path) if (need_depth and conf_path) else []
    if need_depth and not depth_list:
        sys.exit(f"rgbd-dir: no depth frames under {depth_path}")
    idx = 0
    for rgb in _color_frames(rgb_path):
        depth = conf = None
        if need_depth:
            if idx >= len(depth_list):
                break
            depth = _load_depth(depth_list[idx])
            if depth is None:
                continue
            if conf_list and idx < len(conf_list):
                conf = _load_conf(conf_list[idx], depth.shape)
        yield idx, rgb, depth, conf
        idx += 1


def _probe_video(rgb_path, fps_override):
    """(total_color_frames, fps) for a video file or an image directory."""
    import cv2
    if os.path.isfile(rgb_path):
        cap = cv2.VideoCapture(rgb_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    else:
        import glob
        n = len([f for f in glob.glob(os.path.join(rgb_path, "*"))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        fps = 30.0
    return n, (fps_override or fps)


def run_rgbd_dir(args, detector, need_depth):
    rgb_path, depth_path, intr_path, conf_path = _resolve_rgbd(args, need_depth)
    K = load_intrinsics(intr_path) if need_depth else None
    n_video, fps = _probe_video(rgb_path, args.fps)
    scale = args.depth_scale
    if scale is None:
        scale = 0.001
        if need_depth and np.issubdtype(
                _probe_depth_dtype(_depth_files(depth_path)[0]), np.floating):
            scale = 1.0                                # float depth is in metres
    if need_depth:
        print(f"[bag_to_joints] rgbd-dir  rgb={os.path.basename(rgb_path)}  "
              f"depth={os.path.basename(depth_path)}  "
              f"confidence={os.path.basename(conf_path) if conf_path else 'none'}  "
              f"intrinsics={os.path.basename(intr_path)}  depth_scale={scale}")
    else:
        print(f"[bag_to_joints] rgbd-dir(mono)  rgb={os.path.basename(rgb_path)}")
    frame_iter = frames_from_rgbd_dir(rgb_path, depth_path, conf_path, need_depth)
    rows, nf, nnp, rej = process_frames(args, detector, K, frame_iter, need_depth,
                                        scale, fps)
    totals = dict(video=n_video, fps=fps)
    if need_depth:
        totals["depth"] = len(_depth_files(depth_path))
    return rows, nf, nnp, K, scale, rej, totals


def run(args, detector=None):
    need_depth = (args.mode == "rgbd")
    if not hasattr(args, "joint_indices"):
        args.joint_indices = JOINT_SETS[getattr(args, "joints", "full")]
    own = detector is None
    if own:
        detector = MediaPipeDetector(args.model_complexity,
                                     indices=args.joint_indices)
    try:
        if args.source == "rosbag":
            rows, nf, nnp, K, _, rej, tot = run_rosbag(args, detector, need_depth)
        else:
            rows, nf, nnp, K, _, rej, tot = run_rgbd_dir(args, detector, need_depth)
    finally:
        if own:
            detector.close()

    write_rows(args.out, rows)
    detected = nf - nnp
    fps = tot.get("fps", 30.0)
    dt = args.stride / fps if fps else float("nan")
    depth_note = f"  depth_frames={tot['depth']}" if "depth" in tot else ""
    print(f"[bag_to_joints] video_frames={tot.get('video', '?')}{depth_note}  "
          f"processed={nf} (stride={args.stride}, fps={fps:.1f}, "
          f"dt={dt * 1000:.1f} ms)")
    print(f"[bag_to_joints] source={args.source} mode={args.mode}  frames={nf}  "
          f"pose_detected={detected}  no_pose={nnp}")
    if need_depth and rej:
        kept = len(rows)
        total_rej = sum(rej.values())
        pct = 100.0 * total_rej / (total_rej + kept) if (total_rej + kept) else 0.0
        print(f"[bag_to_joints] depth-rejected joint samples: {total_rej} "
              f"({pct:.1f}% of would-be joints)  {dict(sorted(rej.items()))}")
    counts = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    order = [MP_INDEX_TO_JOINT[i] for i in args.joint_indices]
    print("[bag_to_joints] rows per joint: "
          + "  ".join(f"{n}={counts.get(n, 0)}" for n in order))
    print(f"[bag_to_joints] wrote {len(rows)} joint rows -> {args.out}")
    if need_depth and K is not None:
        print(f"[bag_to_joints] intrinsics fx,fy,cx,cy = "
              f"{tuple(round(v, 1) for v in K)}")


# ── Depth confidence diagnostic (--diagnose-depth) ───────────────────────────
def _diagnose_loop(args, detector, frame_iter, scale, fps, n_video):
    """Per-joint LiDAR confidence composition at landmark pixels, NO gates."""
    order = [MP_INDEX_TO_JOINT[i] for i in args.joint_indices]
    comp = {n: {} for n in order}          # center-pixel composition counts
    patch = {n: [0, 0, 0] for n in order}  # [n, has_conf2, has_conf1plus]
    prev, jumps = {}, {n: [] for n in order}
    raw, n_frames, n_nopose = [], 0, 0
    r = args.depth_window

    for idx, rgb, depth, conf in frame_iter:
        if args.stride > 1 and idx % args.stride:
            continue
        if args.max_frames and n_frames >= args.max_frames:
            break
        n_frames += 1
        image_lms, _ = detector.detect(rgb)
        if image_lms is None:
            n_nopose += 1
            continue
        dh, dw = depth.shape[:2]
        for mp_idx in args.joint_indices:
            lm = image_lms.get(mp_idx)
            if lm is None:
                continue
            joint = MP_INDEX_TO_JOINT[mp_idx]
            ui, vi = int(round(lm[0] * dw)), int(round(lm[1] * dh))
            if not (0 <= ui < dw and 0 <= vi < dh):
                comp[joint]["oob"] = comp[joint].get("oob", 0) + 1
                raw.append((idx, joint, "oob", "", "", "", "", ""))
                continue
            d0 = int(depth[vi, ui])
            c0 = int(conf[vi, ui]) if conf is not None else None
            center = "nodepth" if d0 == 0 else \
                ("na" if c0 is None else str(c0))
            comp[joint][center] = comp[joint].get(center, 0) + 1
            u0, u1 = max(0, ui - r), min(dw, ui + r + 1)
            v0, v1 = max(0, vi - r), min(dh, vi + r + 1)
            dp = depth[v0:v1, u0:u1]
            nz = int((dp == 0).sum())
            if conf is not None:
                cp = conf[v0:v1, u0:u1]
                n2 = int(((cp == 2) & (dp > 0)).sum())
                n1 = int(((cp == 1) & (dp > 0)).sum())
                n0 = int(((cp == 0) & (dp > 0)).sum())
            else:
                n2 = n1 = n0 = -1              # source has no confidence maps
            p = patch[joint]
            p[0] += 1
            p[1] += 1 if n2 > 0 else 0
            p[2] += 1 if (n2 > 0 or n1 > 0) else 0
            raw.append((idx, joint, center,
                        f"{d0 * scale * 1000:.0f}" if d0 else "",
                        n2, n1, n0, nz))
            if d0:
                z_mm = d0 * scale * 1000.0
                if joint in prev:
                    jumps[joint].append((abs(z_mm - prev[joint][0]),
                                         idx - prev[joint][1]))
                prev[joint] = (z_mm, idx)

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "joint", "center_conf", "center_depth_mm",
                    "patch_conf2", "patch_conf1", "patch_conf0", "patch_zero"])
        w.writerows(raw)

    dt = args.stride / fps
    print(f"[diagnose] video_frames={n_video}  processed={n_frames}  "
          f"pose_detected={n_frames - n_nopose}  no_pose={n_nopose}")
    print(f"[diagnose] stride={args.stride}  fps={fps:.1f}  "
          f"dt={dt * 1000:.1f} ms  patch={2 * r + 1}x{2 * r + 1}")
    print(f"[diagnose] raw CSV -> {out}")
    print()
    hdr = (f"{'joint':<16}{'n':>6}{'c2%':>7}{'c1%':>7}{'c0%':>7}{'nodep%':>8}"
           f"{'oob%':>7}{'patch2%':>9}{'patch1+%':>9}")
    print(hdr)
    for n in order:
        c = comp[n]
        tot = sum(c.values())
        if tot == 0:
            print(f"{n:<16}{0:>6}")
            continue

        def pc(key):
            return 100.0 * c.get(key, 0) / tot

        p = patch[n]
        p2 = 100.0 * p[1] / p[0] if p[0] else 0.0
        p1 = 100.0 * p[2] / p[0] if p[0] else 0.0
        print(f"{n:<16}{tot:>6}{pc('2'):>7.1f}{pc('1'):>7.1f}{pc('0'):>7.1f}"
              f"{pc('nodepth'):>8.1f}{pc('oob'):>7.1f}{p2:>9.1f}{p1:>9.1f}")
    print()
    speed = args.max_joint_speed or 4.0
    print(f"{'joint':<16}{'pairs':>7}{'med_dz':>8}{'p95_dz':>8}"
          f"{'trip300':>9}{'tripV':>7}   (dz mm; tripV = {speed:g} m/s x dt)")
    for n in order:
        js = jumps[n]
        if not js:
            print(f"{n:<16}{0:>7}")
            continue
        dz = np.array([a for a, _ in js])
        thr = np.array([speed * 1000.0 * (d / fps) for _, d in js])
        t300 = 100.0 * float((dz > 300.0).mean())
        tv = 100.0 * float((dz > thr).mean())
        print(f"{n:<16}{len(js):>7}{np.median(dz):>8.0f}"
              f"{np.percentile(dz, 95):>8.0f}{t300:>8.1f}%{tv:>6.1f}%")


def run_diagnose(args):
    if not hasattr(args, "joint_indices"):
        args.joint_indices = JOINT_SETS[getattr(args, "joints", "full")]
    detector = MediaPipeDetector(args.model_complexity,
                                 indices=args.joint_indices)
    try:
        if args.source == "rosbag":
            from rosbags.highlevel import AnyReader
            from rosbags.typesys import Stores, get_typestore
            from pathlib import Path
            scale = args.depth_scale if args.depth_scale is not None else 0.001
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            with AnyReader([Path(args.input)],
                           default_typestore=typestore) as reader:
                n_color = sum(c.msgcount for c in reader.connections
                              if c.topic == args.color_topic)
                dur = max((reader.end_time - reader.start_time) / 1e9, 1e-6)
                fps = args.fps or (n_color / dur if n_color else 30.0)
                frame_iter = iter_frames(reader, args.color_topic,
                                         args.depth_topic, True)
                _diagnose_loop(args, detector, frame_iter, scale, fps, n_color)
        else:
            rgb_path, depth_path, intr_path, conf_path = _resolve_rgbd(args, True)
            n_video, fps = _probe_video(rgb_path, args.fps)
            scale = args.depth_scale
            if scale is None:
                scale = 0.001
                if np.issubdtype(
                        _probe_depth_dtype(_depth_files(depth_path)[0]),
                        np.floating):
                    scale = 1.0
            print(f"[diagnose] rgbd-dir  rgb={os.path.basename(rgb_path)}  "
                  f"depth={os.path.basename(depth_path)}  "
                  f"confidence={os.path.basename(conf_path) if conf_path else 'NONE'}")
            frame_iter = frames_from_rgbd_dir(rgb_path, depth_path, conf_path,
                                              True)
            _diagnose_loop(args, detector, frame_iter, scale, fps, n_video)
    finally:
        detector.close()


# ── Self-test (numpy only; no bag, no MediaPipe) ─────────────────────────────
def run_self_test():
    print("[self-test] deprojection round-trip + CSV, no bag/mediapipe")
    fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
    K = (fx, fy, cx, cy)
    W, H, scale = 640, 480, 0.001
    depth = np.zeros((H, W), dtype=np.uint16)

    # Known 3D points -> project to pixels -> plant depth -> expect recovery.
    truth = {
        11: (0.20, -0.30, 2.50), 12: (-0.20, -0.30, 2.50),
        13: (0.30, 0.00, 2.55),  14: (-0.30, 0.00, 2.55),
        15: (0.35, 0.25, 2.60),  16: (-0.35, 0.25, 2.60),
        23: (0.12, 0.45, 2.50),  24: (-0.12, 0.45, 2.50),
    }
    image_lms, world_lms = {}, {}
    for idx, (X, Y, Z) in truth.items():
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        depth[int(round(v)), int(round(u))] = int(round(Z / scale))
        image_lms[idx] = (u / W, v / H, 0.95)
        world_lms[idx] = (X, Y, Z, 0.95)     # pretend world == camera for test

    tcfg = dict(radius=1, scale=scale, conf_min_by=2, spread_max_mm=0.0,
                pick="median", search_radius=0, jump_speed=0.0,
                jump_fix_mm=None, fps=30.0)
    rgbd_rows, _ = rows_from_depth(image_lms, depth, None, K, 0, (W, H),
                                   tcfg, {}, sorted(truth))
    max_err = 0.0
    by_joint = {r[1]: np.array(r[2:5]) for r in rgbd_rows}
    for idx, (X, Y, Z) in truth.items():
        got = by_joint[MP_INDEX_TO_JOINT[idx]]
        max_err = max(max_err, float(np.linalg.norm(got - np.array([X, Y, Z]))))
    print(f"[self-test] rgbd rows: {len(rgbd_rows)}  "
          f"max deprojection error: {max_err * 1000:.4f} mm")
    assert len(rgbd_rows) == len(truth), "rgbd dropped joints"
    assert max_err < 5e-3, f"deprojection round-trip too large: {max_err}"

    mono_rows = rows_from_world(world_lms, 0, sorted(truth))
    assert len(mono_rows) == len(truth), "mono dropped joints"

    # hole handling: a joint whose depth patch is empty must be dropped
    empty, _ = rows_from_depth({99: (0.5, 0.5, 0.9)}, np.zeros((H, W), np.uint16),
                               None, K, 0, (W, H), tcfg, {}, FULL_JOINT_INDICES)
    # index 99 isn't a canonical joint, so also confirm a real idx with no depth drops
    empty2, _ = rows_from_depth({15: (0.5, 0.5, 0.9)}, np.zeros((H, W), np.uint16),
                                None, K, 0, (W, H), tcfg, {}, FULL_JOINT_INDICES)
    assert empty == [] and empty2 == [], "hole handling failed"

    # search policy: no accepted pixel in 3x3, one plantable at ring radius 2
    sdepth = np.zeros((H, W), np.uint16)
    sdepth[240, 322] = 2500                    # landmark center is (320, 240)
    scfg = dict(tcfg)
    no_search, st0 = rows_from_depth({15: (0.5, 0.5, 0.9)}, sdepth, None, K, 0,
                                     (W, H), scfg, {}, FULL_JOINT_INDICES)
    scfg["search_radius"] = 2
    with_search, st1 = rows_from_depth({15: (0.5, 0.5, 0.9)}, sdepth, None, K, 0,
                                       (W, H), scfg, {}, FULL_JOINT_INDICES)
    assert no_search == [] and st0.get("no_valid_depth") == 1, "search off failed"
    assert len(with_search) == 1 and st1.get("recovered") == 1, "search recovery failed"
    print("[self-test] search-policy recovery OK")

    # dt-aware jump gate: 4 m/s over 1 frame at 30 fps = 133 mm threshold
    jcfg = dict(tcfg, jump_speed=4.0)
    jd = np.zeros((H, W), np.uint16)
    jd[240, 320] = 2500
    trk = {}
    rows_from_depth({15: (0.5, 0.5, 0.9)}, jd, None, K, 0, (W, H), jcfg, trk,
                    FULL_JOINT_INDICES)
    jd[240, 320] = 2700                        # +200 mm in one frame: > 133 mm
    r1, s1 = rows_from_depth({15: (0.5, 0.5, 0.9)}, jd, None, K, 1, (W, H),
                             jcfg, trk, FULL_JOINT_INDICES)
    assert r1 == [] and s1.get("jump") == 1, "dt-aware jump gate failed"
    jd[240, 320] = 2700
    r2, _ = rows_from_depth({15: (0.5, 0.5, 0.9)}, jd, None, K, 31, (W, H),
                            jcfg, trk, FULL_JOINT_INDICES)
    assert len(r2) == 1, "dt-aware gate should pass after a long gap (30 frames)"
    print("[self-test] dt-aware jump gate OK")

    import tempfile
    out = os.path.join(tempfile.mkdtemp(prefix="bag2j_"), "selftest_rgbd.csv")
    write_rows(out, rgbd_rows)
    print(f"[self-test] wrote {out}")

    _self_test_rgbd_dir()
    print("[self-test] PASS")


def _self_test_rgbd_dir():
    """End-to-end rgbd-dir path with a synthetic capture and a FakeDetector."""
    import cv2
    import tempfile
    print("[self-test] rgbd-dir end to end (synthetic capture, no mediapipe)")
    fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
    W, H = 640, 480
    truth = {11: (0.20, -0.30, 2.50), 12: (-0.20, -0.30, 2.50),
             13: (0.30, 0.00, 2.55), 14: (-0.30, 0.00, 2.55),
             15: (0.35, 0.25, 2.60), 16: (-0.35, 0.25, 2.60),
             23: (0.12, 0.45, 2.50), 24: (-0.12, 0.45, 2.50)}
    depth = np.zeros((H, W), dtype=np.uint16)
    image_lms = {}
    for idx, (X, Y, Z) in truth.items():
        u, v = fx * X / Z + cx, fy * Y / Z + cy
        depth[int(round(v)), int(round(u))] = int(round(Z / 0.001))
        image_lms[idx] = (u / W, v / H, 0.95)

    d = tempfile.mkdtemp(prefix="bag2j_dir_")
    os.makedirs(os.path.join(d, "rgb"))
    os.makedirs(os.path.join(d, "depth"))
    cv2.imwrite(os.path.join(d, "rgb", "000000.png"), np.zeros((H, W, 3), np.uint8))
    cv2.imwrite(os.path.join(d, "depth", "000000.png"), depth)   # 16-bit PNG
    with open(os.path.join(d, "intrinsics.json"), "w") as f:
        json.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy}, f)

    ns = argparse.Namespace(
        input=d, source="rgbd-dir", mode="rgbd", out=os.path.join(d, "out.csv"),
        rgb=None, depth=None, confidence=None, intrinsics=None, depth_scale=None,
        depth_window=1, depth_policy="strict", depth_conf_min=2,
        depth_spread_max=0.0, max_joint_speed=0.0, depth_jump_max=None,
        depth_pick="median", fps=None, diagnose_depth=False, stride=1,
        max_frames=0, color_topic=None, depth_topic=None, info_topic=None)
    run(ns, detector=FakeDetector(image_lms, {}))

    got = {}
    with open(ns.out) as f:
        r = csv.DictReader(f)
        for row in r:
            got[row["joint"]] = np.array([float(row["x"]), float(row["y"]),
                                          float(row["z"])])
    assert len(got) == len(truth), f"rgbd-dir wrote {len(got)} joints, want {len(truth)}"
    err = max(float(np.linalg.norm(got[MP_INDEX_TO_JOINT[i]] - np.array(p)))
              for i, p in truth.items())
    print(f"[self-test] rgbd-dir max deprojection error: {err * 1000:.4f} mm")
    assert err < 5e-3, f"rgbd-dir round-trip too large: {err}"


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?",
                   help="ROS2 bag dir (rosbag) or capture dir (rgbd-dir)")
    p.add_argument("--source", choices=["rosbag", "rgbd-dir"], default="rosbag",
                   help="capture source (default: rosbag)")
    p.add_argument("--mode", choices=["mono", "rgbd"], help="which pipeline")
    p.add_argument("--joints", choices=["arm", "full"], default="full",
                   help="arm = pre-extension set (6 retargeted joints + hips "
                        "for pelvis alignment); full = adds head and legs for "
                        "RULA/REBA scoring (default)")
    p.add_argument("--out", help="output canonical joints CSV "
                                 "(default: <input>/b2g/<mode>.csv)")
    # rosbag source
    p.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--depth-topic",
                   default="/camera/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    # rgbd-dir source (auto-detected under input/ if omitted)
    p.add_argument("--rgb", help="rgbd-dir: color mp4 or image dir")
    p.add_argument("--depth", help="rgbd-dir: depth image dir")
    p.add_argument("--confidence", help="rgbd-dir: confidence image dir "
                                        "(Stray Scanner 0/1/2); auto-detected")
    p.add_argument("--intrinsics", help="rgbd-dir: intrinsics json or 3x3 csv")
    p.add_argument("--depth-scale", type=float, default=None,
                   help="raw depth unit -> metres. Default auto: 0.001 for "
                        "uint16 mm, 1.0 for float metres")
    # depth sampling hardening (native-resolution patch)
    p.add_argument("--depth-window", type=int, default=1,
                   help="depth patch half-size in NATIVE depth px (1 = 3x3)")
    p.add_argument("--depth-policy", choices=list(DEPTH_POLICIES),
                   default="relaxed-limb",
                   help="confidence policy: relaxed-limb = conf 1 on limbs, 2 on "
                        "shoulders/hips (DEFAULT, frozen 2026-07-10 from the "
                        "depth_policy_matrix assessment on M1_R_1 + test1); "
                        "strict = conf 2 everywhere; relaxed-all = conf 1 "
                        "everywhere; search = relaxed-limb + 5x5 nearest-pixel "
                        "fallback (rejected by the matrix: background bleed)")
    p.add_argument("--depth-pick", choices=["median", "nearest"], default="median",
                   help="median = median of survivors + spread reject; "
                        "nearest = closest surface (keeps extended limbs)")
    p.add_argument("--depth-conf-min", type=int, default=None,
                   help="override the policy's min confidence for ALL joints "
                        "(Stray Scanner 0/1/2). Default: policy decides")
    p.add_argument("--depth-spread-max", type=float, default=150.0,
                   help="reject a joint if its in-patch depth spread exceeds this "
                        "(mm); catches limb-over-background. 0 disables")
    p.add_argument("--max-joint-speed", type=float, default=4.0,
                   help="jump gate: max plausible joint speed (m/s); threshold = "
                        "speed x actual dt between compared frames. 0 disables")
    p.add_argument("--depth-jump-max", type=float, default=None,
                   help="jump gate: FIXED mm threshold overriding the dt-aware "
                        "speed threshold (legacy behavior). Default: dt-aware")
    p.add_argument("--fps", type=float, default=None,
                   help="override source frame rate used for dt (default: probed "
                        "from the video / bag)")
    p.add_argument("--diagnose-depth", action="store_true",
                   help="rgbd diagnostic: record per-joint LiDAR confidence and "
                        "depth WITHOUT gates, print composition + jump stats, "
                        "write raw CSV. No joints CSV is produced")
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument("--self-test", action="store_true",
                   help="verify deprojection + CSV with synthetic data")
    return p.parse_args()


def _looks_like_rgbd_dir(path):
    """Capture dir (rgb.mp4 / rgb/ / depth/), NOT a rosbag2 dir (metadata.yaml)."""
    return (os.path.isdir(path)
            and not os.path.isfile(os.path.join(path, "metadata.yaml"))
            and (os.path.isfile(os.path.join(path, "rgb.mp4"))
                 or os.path.isdir(os.path.join(path, "rgb"))
                 or os.path.isdir(os.path.join(path, "depth"))))


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.input and args.source == "rosbag" and _looks_like_rgbd_dir(args.input):
        print("[bag_to_joints] input looks like an RGBD capture dir; "
              "using --source rgbd-dir")
        args.source = "rgbd-dir"
    if args.diagnose_depth:
        if not args.input:
            sys.exit("error: --diagnose-depth needs an INPUT capture")
        if not os.path.exists(args.input):
            sys.exit(f"error: input not found: {args.input}")
        args.mode = "rgbd"                 # diagnostic is depth-side by nature
        if not args.out:
            args.out = os.path.join(args.input.rstrip("/"), "b2g",
                                    "depth_diag.csv")
        run_diagnose(args)
        return
    if not (args.input and args.mode):
        sys.exit("error: need INPUT and --mode (or --self-test)")
    if not os.path.exists(args.input):
        sys.exit(f"error: input not found: {args.input}")
    if not args.out:                       # default: b2g folder inside the data dir
        args.out = os.path.join(args.input.rstrip("/"), "b2g", f"{args.mode}.csv")
    run(args)


if __name__ == "__main__":
    main()
