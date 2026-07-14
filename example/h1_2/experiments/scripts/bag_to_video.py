#!/usr/bin/env python3
"""bag_to_video.py - render a camera stream from a rosbag to an mp4.

A ROS2 bag stores raw frames, not a video file. Those frames ARE the footage,
so this renders the color (or aligned-depth) stream to a viewable mp4 for
review, framing checks, or paper figures. The lossless frames stay in the bag;
this is only a convenience render.

Prefer this over recording a separate video live: no extra CPU during capture,
lossless source, and you already have the frames.

Usage:
    python3 bag_to_video.py BAGDIR --out M1_S1_R1.mp4
    python3 bag_to_video.py BAGDIR --stream depth --out depth.mp4 --depth-max 4.0
    python3 bag_to_video.py BAGDIR --out quick.mp4 --max-frames 90   # ~3 s at 30 fps

Needs: cv2 + rosbags (already used by bag_to_joints.py). No ROS install needed.
"""

import argparse
import os
import sys

import numpy as np


def decode_color_bgr(msg):
    """sensor_msgs/Image (rgb8/bgr8) -> HxWx3 BGR uint8 (cv2 order)."""
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    if enc == "rgb8":
        arr = arr[:, :, ::-1]                 # cv2 writes BGR
    elif enc != "bgr8":
        sys.exit(f"unsupported color encoding '{msg.encoding}' (want rgb8/bgr8)")
    return np.ascontiguousarray(arr)


def colorize_depth(msg, depth_max_m, scale):
    """16UC1 depth -> BGR colormap, clipped at depth_max_m metres."""
    import cv2
    d = np.frombuffer(msg.data, dtype=np.uint16).reshape(
        msg.height, msg.width).astype(np.float32) * scale
    norm = np.clip(d / depth_max_m, 0.0, 1.0)
    d8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    colored[d == 0] = 0                        # black where there is no depth
    return colored


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", help="ROS2 bag directory")
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--stream", choices=["color", "depth"], default="color")
    p.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--depth-topic",
                   default="/camera/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--fps", type=float, default=30.0, help="output fps (default 30)")
    p.add_argument("--depth-scale", type=float, default=0.001,
                   help="raw depth unit -> metres (RealSense 16UC1 = 0.001)")
    p.add_argument("--depth-max", type=float, default=4.0,
                   help="depth colormap saturates at this many metres")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = p.parse_args()

    if not os.path.exists(args.bag):
        sys.exit(f"bag not found: {args.bag}")

    import cv2
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from pathlib import Path

    topic = args.color_topic if args.stream == "color" else args.depth_topic
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    writer, n = None, 0
    with AnyReader([Path(args.bag)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            sys.exit(f"topic '{topic}' not in bag")
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            frame = (decode_color_bgr(m) if args.stream == "color"
                     else colorize_depth(m, args.depth_max, args.depth_scale))
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
                if not writer.isOpened():
                    sys.exit("cv2.VideoWriter failed to open (codec/path issue)")
            writer.write(frame)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    print(f"[bag_to_video] wrote {n} {args.stream} frames @ {args.fps:g} fps "
          f"-> {args.out}")
    if n == 0:
        sys.exit(f"no frames on topic '{topic}'")


if __name__ == "__main__":
    main()
