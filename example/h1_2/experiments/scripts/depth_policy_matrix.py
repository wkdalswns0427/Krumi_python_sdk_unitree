#!/usr/bin/env python3
"""depth_policy_matrix.py - coverage vs error for every depth policy.

Output: one assessment matrix (stdout + CSV): rows per joint (coverage),
rejection breakdown, and per-joint mean/median/p95/max error plus overall.
The point is coverage gained vs error paid, per policy. No winner is picked
here; that is a human decision.

Usage:
    /usr/bin/python3 depth_policy_matrix.py CAPDIR [--mono MONO.csv]
        [--stride 2] [--out-prefix CAPDIR/b2g/matrix]

The mono CSV must exist at the SAME stride (frame indices are joined).
Defaults to CAPDIR/b2g/iph_mono.csv or full_mono.csv if present.
"""

import argparse
import csv
import os
import sys

import numpy as np

import example.h1_2.experiments.scripts.bag_to_joints as b2j
import example.h1_2.experiments.scripts.block1_compare as bc


def find_mono(capdir, explicit):
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit(f"mono CSV not found: {explicit}")
        return explicit
    for name in ("iph_mono.csv", "full_mono.csv", "mono.csv"):
        cand = os.path.join(capdir, "b2g", name)
        if os.path.isfile(cand):
            return cand
    sys.exit(f"no mono CSV under {capdir}/b2g; generate one first with "
             f"bag_to_joints --mode mono (same --stride), or pass --mono")


def run_policies(args):
    """One MediaPipe pass; returns {policy: (rows, stats)} + run info."""
    shim = argparse.Namespace(input=args.capture, rgb=None, depth=None,
                              confidence=None, intrinsics=None)
    rgb_path, depth_path, intr_path, conf_path = b2j._resolve_rgbd(shim, True)
    K = b2j.load_intrinsics(intr_path)
    n_video, fps = b2j._probe_video(rgb_path, args.fps)
    scale = 0.001
    if np.issubdtype(
            b2j._probe_depth_dtype(b2j._depth_files(depth_path)[0]),
            np.floating):
        scale = 1.0

    indices = b2j.JOINT_SETS["full"]
    policies = {}
    for name in b2j.DEPTH_POLICIES:
        conf_by, search = b2j.build_policy(name)
        cfg = dict(radius=args.depth_window, scale=scale, conf_min_by=conf_by,
                   spread_max_mm=args.depth_spread_max, pick=args.depth_pick,
                   search_radius=search, jump_speed=args.max_joint_speed,
                   jump_fix_mm=None, fps=fps)
        policies[name] = dict(cfg=cfg, track={}, rows=[], stats={})

    detector = b2j.MediaPipeDetector(args.model_complexity, indices=indices)
    n_frames = n_nopose = 0
    img_wh = None
    try:
        for idx, rgb, depth, conf in b2j.frames_from_rgbd_dir(
                rgb_path, depth_path, conf_path, True):
            if args.stride > 1 and idx % args.stride:
                continue
            if args.max_frames and n_frames >= args.max_frames:
                break
            n_frames += 1
            if img_wh is None:
                img_wh = (rgb.shape[1], rgb.shape[0])
            image_lms, _ = detector.detect(rgb)
            if image_lms is None:
                n_nopose += 1
                continue
            for pol in policies.values():
                rows, stats = b2j.rows_from_depth(
                    image_lms, depth, conf, K, idx, img_wh, pol["cfg"],
                    pol["track"], indices)
                pol["rows"] += rows
                for k, v in stats.items():
                    pol["stats"][k] = pol["stats"].get(k, 0) + v
    finally:
        detector.close()

    info = dict(video=n_video, processed=n_frames, detected=n_frames - n_nopose,
                fps=fps, dt_ms=1000.0 * args.stride / fps)
    return policies, info


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture", help="RGBD capture directory (rgbd-dir layout)")
    p.add_argument("--mono", help="mono joints CSV at the same stride")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--depth-window", type=int, default=1)
    p.add_argument("--depth-spread-max", type=float, default=150.0)
    p.add_argument("--depth-pick", choices=["median", "nearest"], default="median")
    p.add_argument("--max-joint-speed", type=float, default=4.0)
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--out-prefix",
                   help="default: <capture>/b2g/matrix")
    args = p.parse_args()

    if not os.path.isdir(args.capture):
        sys.exit(f"capture dir not found: {args.capture}")
    cap_name = os.path.basename(os.path.abspath(args.capture))
    mono_path = find_mono(args.capture, args.mono)
    prefix = args.out_prefix or os.path.join(args.capture, "b2g", "matrix")
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    print(f"[matrix] capture={cap_name}  mono={mono_path}  stride={args.stride}")
    policies, info = run_policies(args)
    print(f"[matrix] video_frames={info['video']}  processed={info['processed']}"
          f"  pose_detected={info['detected']}  fps={info['fps']:.1f}"
          f"  dt={info['dt_ms']:.1f} ms")

    joint_order = [b2j.MP_INDEX_TO_JOINT[i] for i in b2j.JOINT_SETS["full"]]
    reject_keys = ["no_valid_depth", "spread", "jump", "jump_relock", "oob",
                   "recovered"]

    # per-policy joints CSVs + coverage
    coverage = {}
    for name, pol in policies.items():
        out_csv = f"{prefix}_{name}_rgbd.csv"
        b2j.write_rows(out_csv, pol["rows"])
        cnt = {}
        for row in pol["rows"]:
            cnt[row[1]] = cnt.get(row[1], 0) + 1
        coverage[name] = cnt

    print("\n== coverage: rows per joint ==")
    print(f"{'joint':<16}" + "".join(f"{n:>14}" for n in policies))
    for j in joint_order:
        print(f"{j:<16}" + "".join(f"{coverage[n].get(j, 0):>14}"
                                   for n in policies))
    print("\n== depth sampling stats ==")
    print(f"{'reason':<16}" + "".join(f"{n:>14}" for n in policies))
    for k in reject_keys:
        print(f"{k:<16}" + "".join(f"{policies[n]['stats'].get(k, 0):>14}"
                                   for n in policies))

    # block1_compare per policy (arm scope, procrustes)
    mono = bc.load_frames(mono_path, "m")
    matrix_rows = []
    print("\n== block1_compare (arm scope, --procrustes) ==")
    print(f"{'policy':<14}{'joint':<16}{'pairs':>7}{'mean':>8}{'median':>8}"
          f"{'p95':>8}{'max':>9}  (mm)")
    for name in policies:
        rgbd = bc.load_frames(f"{prefix}_{name}_rgbd.csv", "m")
        errors, meta = bc.compare(mono, rgbd, "pelvis", 0.5, "min", "rot")
        rows, overall = bc.summarize(errors)
        for r in rows:
            label = r["joint"]
            print(f"{name:<14}{label:<16}{r['n']:>7}{r['mean']:>8.2f}"
                  f"{r['median']:>8.2f}{r['p95']:>8.2f}{r['max']:>9.2f}")
            matrix_rows.append(dict(
                capture=cap_name, policy=name, joint=label,
                coverage_rows=coverage[name].get(label, ""),
                pairs=r["n"], mean_mm=f"{r['mean']:.3f}",
                median_mm=f"{r['median']:.3f}", p95_mm=f"{r['p95']:.3f}",
                max_mm=f"{r['max']:.3f}"))
        print("-" * 78)

    # matrix CSV: errors + coverage per joint, plus one stats row per reason
    out_matrix = f"{prefix}_assessment.csv"
    with open(out_matrix, "w", newline="") as f:
        fields = ["capture", "policy", "joint", "coverage_rows", "pairs",
                  "mean_mm", "median_mm", "p95_mm", "max_mm"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(matrix_rows)
        for name in policies:
            for k in reject_keys:
                w.writerow(dict(capture=cap_name, policy=name,
                                joint=f"stat_{k}",
                                coverage_rows=policies[name]["stats"].get(k, 0),
                                pairs="", mean_mm="", median_mm="", p95_mm="",
                                max_mm=""))
    print(f"[matrix] wrote {out_matrix} and per-policy joints CSVs "
          f"({prefix}_<policy>_rgbd.csv)")


if __name__ == "__main__":
    main()
