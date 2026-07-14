#!/usr/bin/env python3
"""visualize_joints.py - view the per-frame 3D joints from bag_to_joints.py.

Reads the canonical long CSV (frame,joint,x,y,z,conf) that bag_to_joints.py
writes and renders the upper-body skeleton (shoulders, elbows, wrists, hips)
as an interactive 3D plot, a saved image, or an animation. Can overlay two
CSVs (for example mono vs rgbd) to eyeball the difference.

Display axes: horizontal = x, depth into screen = z, up = -y (the CSV y is
down in both camera and MediaPipe world frames, so up is negated).

Usage:
    # one frame (defaults to the most complete frame), interactive window:
    python3 visualize_joints.py smoke2_rgbd.csv
    # a specific frame, saved to png:
    python3 visualize_joints.py smoke2_rgbd.csv --frame 425 --out frame425.png
    # animate every frame to a video:
    python3 visualize_joints.py smoke2_rgbd.csv --out smoke2.mp4 --fps 20
    # overlay mono vs rgbd (both centred on the pelvis):
    python3 visualize_joints.py mono.csv --overlay rgbd.csv --frame 425 --out cmp.png

Runs under system python3.10 with matplotlib. No ROS, no MediaPipe.
"""

import argparse
import csv
import os
import sys

import numpy as np


JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip"]

# (group name, color, [(joint_a, joint_b), ...])
SKELETON = [
    ("left arm",  "#2196F3", [("left_shoulder", "left_elbow"),
                              ("left_elbow", "left_wrist")]),
    ("right arm", "#F44336", [("right_shoulder", "right_elbow"),
                              ("right_elbow", "right_wrist")]),
    ("shoulders", "#4CAF50", [("left_shoulder", "right_shoulder")]),
    ("torso",     "#9C27B0", [("left_shoulder", "left_hip"),
                              ("right_shoulder", "right_hip"),
                              ("left_hip", "right_hip")]),
]


# ── loading ──────────────────────────────────────────────────────────────────
def load_csv(path):
    """{frame_index: {joint: (np.array([x,y,z]), conf)}}."""
    frames = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        need = {"frame", "joint", "x", "y", "z"}
        if not need.issubset(reader.fieldnames or []):
            sys.exit(f"{path}: expected columns {sorted(need)}; "
                     f"got {reader.fieldnames}")
        has_conf = "conf" in reader.fieldnames
        for row in reader:
            fi = int(float(row["frame"]))
            p = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            conf = float(row["conf"]) if has_conf and row["conf"] != "" else 1.0
            frames.setdefault(fi, {})[row["joint"].strip()] = (p, conf)
    if not frames:
        sys.exit(f"{path}: no rows")
    return frames


def visible(frame_joints, conf_min):
    return {j: p for j, (p, c) in frame_joints.items() if c >= conf_min}


def best_frame(frames, conf_min):
    """Frame index with the most joints above the confidence threshold."""
    return max(frames, key=lambda fi: (len(visible(frames[fi], conf_min)), -fi))


def pelvis(pts):
    if "left_hip" in pts and "right_hip" in pts:
        return (pts["left_hip"] + pts["right_hip"]) / 2.0
    return np.mean(list(pts.values()), axis=0) if pts else np.zeros(3)


def centered(pts, mode):
    if mode == "pelvis" and pts:
        off = pelvis(pts)
        return {j: p - off for j, p in pts.items()}
    return pts


# ── drawing ──────────────────────────────────────────────────────────────────
def _disp(p):
    """CSV (x, y, z) -> plot (x, z, -y): x right, z depth, up = -y."""
    return p[0], p[2], -p[1]


def draw_pose(ax, pts, uniform_color=None, label=True):
    """pts: {joint: xyz}. Bones colored by group, or all uniform_color."""
    for group, color, edges in SKELETON:
        c = uniform_color or color
        for a, b in edges:
            if a in pts and b in pts:
                xa, ya, za = _disp(pts[a])
                xb, yb, zb = _disp(pts[b])
                ax.plot([xa, xb], [ya, yb], [za, zb], color=c, linewidth=2.5,
                        alpha=0.9, zorder=2)
    for j, p in pts.items():
        x, y, z = _disp(p)
        ax.scatter(x, y, z, c=uniform_color or "white",
                   edgecolors="black", s=40, zorder=3, depthshade=False)
        if label and uniform_color is None:
            ax.text(x, y, z + 0.03, j.replace("_", " "), fontsize=6.5,
                    ha="center", color="#dddddd")


def set_equal_axes(ax, all_pts):
    P = np.array([_disp(p) for p in all_pts])
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2
    r = max((hi - lo).max() / 2, 0.1)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def style(ax, title, elev, azim):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (right)")
    ax.set_ylabel("z (depth)")
    ax.set_zlabel("-y (up)")
    ax.view_init(elev=elev, azim=azim)


# ── main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="canonical joints CSV from bag_to_joints.py")
    p.add_argument("--overlay", help="second CSV to overlay (drawn in orange)")
    p.add_argument("--frame", type=int, default=None,
                   help="frame index (default: the most complete frame)")
    p.add_argument("--out", help="save to PNG (static) or MP4/GIF (animation)")
    p.add_argument("--conf-min", type=float, default=0.3,
                   help="hide joints below this confidence (default: 0.3)")
    p.add_argument("--center", choices=["none", "pelvis"], default=None,
                   help="translate to pelvis root (default: pelvis if --overlay)")
    p.add_argument("--fps", type=float, default=20.0, help="animation fps")
    p.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    p.add_argument("--elev", type=float, default=10.0)
    p.add_argument("--azim", type=float, default=-75.0)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.csv):
        sys.exit(f"not found: {args.csv}")
    frames = load_csv(args.csv)
    overlay = load_csv(args.overlay) if args.overlay else None
    center = args.center or ("pelvis" if overlay else "none")

    is_video = bool(args.out) and args.out.lower().endswith((".mp4", ".gif", ".mov"))

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.patches as mpatches

    def frame_pts(store, fi):
        return centered(visible(store.get(fi, {}), args.conf_min), center)

    def legend(ax):
        handles = [mpatches.Patch(color=c, label=g) for g, c, _ in SKELETON]
        if overlay:
            handles.append(mpatches.Patch(color="#FF9800", label="overlay"))
        ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.85)

    if not is_video:
        fi = args.frame if args.frame is not None else best_frame(frames, args.conf_min)
        pts = frame_pts(frames, fi)
        if not pts:
            sys.exit(f"frame {fi}: no joints above conf {args.conf_min}")
        allp = list(pts.values())
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_pose(ax, pts)
        if overlay:
            opts = frame_pts(overlay, fi)
            draw_pose(ax, opts, uniform_color="#FF9800", label=False)
            allp += list(opts.values())
        set_equal_axes(ax, allp)
        style(ax, f"{os.path.basename(args.csv)}  frame {fi}  "
                  f"({len(pts)} joints, conf>={args.conf_min})", args.elev, args.azim)
        legend(ax)
        if args.out:
            plt.savefig(args.out, dpi=140, bbox_inches="tight")
            print(f"wrote {args.out}  (frame {fi})")
        else:
            plt.show()
        return

    # animation
    import matplotlib.animation as animation
    idxs = sorted(frames)[::args.stride]
    allp = [p for fi in idxs for p in frame_pts(frames, fi).values()]
    if overlay:
        allp += [p for fi in idxs for p in frame_pts(overlay, fi).values()]
    if not allp:
        sys.exit("nothing to animate above the confidence threshold")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(k):
        ax.clear()
        fi = idxs[k]
        pts = frame_pts(frames, fi)
        draw_pose(ax, pts)
        if overlay:
            draw_pose(ax, frame_pts(overlay, fi), uniform_color="#FF9800", label=False)
        set_equal_axes(ax, allp)
        style(ax, f"{os.path.basename(args.csv)}  frame {fi}  "
                  f"({k + 1}/{len(idxs)})", args.elev, args.azim)
        if k == 0:
            legend(ax)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(idxs),
                                   interval=1000.0 / args.fps, blit=False)
    if args.out.lower().endswith(".gif"):
        anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=110)
    else:
        anim.save(args.out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400),
                  dpi=110)
    print(f"wrote {args.out}  ({len(idxs)} frames @ {args.fps:g} fps)")


if __name__ == "__main__":
    main()
