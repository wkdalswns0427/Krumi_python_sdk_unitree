#!/usr/bin/env python3
"""skeleton_quality.py - wrist-path smoothness / dropout metrics for a joints CSV.

Gate 2 sanity check (PLAN_v2): before committing a capture session to ZED SDK
BODY_38, confirm its skeleton is not noticeably noisier or more gap-prone than
the iPhone MediaPipe skeleton the IROS paper used, at a comparable reach. Both
sources write the same `frame,joint,x,y,z,conf` schema, so this reads either.

Metrics per joint (default: the two wrists, the fastest-moving end effector and
the one the retargeting tracks):

  dropout%   frames where the joint is absent, non-finite, or conf < --conf-min.
             Directly the "M2/M3 have null points" concern.
  reach_p95  95th-percentile distance from the same-side shoulder (m). Compare
             smoothness at similar reach, not across different reaches.
  jitter_mm  median magnitude of the discrete second difference of position
             (mm), over consecutive present frames. High-frequency wobble in
             millimetres; sampling-rate-fair only when the two clips share fps.
  accel_rms  RMS acceleration (mm/s^2), and
  jerk_rms   RMS jerk (mm/s^3), both using --fps, so they are comparable across
             different frame rates (iPhone vs ZED HD720@60). Lower = smoother.
  speed_p50  median wrist speed (mm/s), a reality check on the clip.

Usage:
    skeleton_quality.py capture.csv --fps 30
    skeleton_quality.py zed.csv --compare iphone.csv --fps 60   # ZED fps
"""

import argparse
import csv
import sys

import numpy as np


def load(path):
    """{joint: (frames ndarray, xyz ndarray Nx3, conf ndarray N)} from the
    frame,joint,x,y,z,conf long CSV, sorted by frame."""
    per = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            j = r["joint"]
            per.setdefault(j, []).append(
                (int(r["frame"]), float(r["x"]), float(r["y"]),
                 float(r["z"]), float(r["conf"])))
    out = {}
    for j, rows in per.items():
        rows.sort()
        a = np.array(rows, float)
        out[j] = (a[:, 0].astype(int), a[:, 1:4], a[:, 4])
    return out


def _present_mask(frames, xyz, conf, conf_min):
    return (conf >= conf_min) & np.isfinite(xyz).all(axis=1)


def joint_metrics(data, joint, fps, conf_min):
    """Metrics dict for one joint, or None if the joint is absent entirely."""
    if joint not in data:
        return None
    frames, xyz, conf = data[joint]
    ok = _present_mask(frames, xyz, conf, conf_min)
    n = len(frames)
    m = dict(n=n, dropout_pct=100.0 * (1.0 - ok.mean()) if n else 100.0)

    # reach vs same-side shoulder, on frames where both are present
    side = "left" if joint.startswith("left") else "right"
    sh = data.get(f"{side}_shoulder")
    if sh is not None:
        sf, sxyz, sc = sh
        common = np.intersect1d(frames[ok], sf[_present_mask(sf, sxyz, sc, conf_min)])
        if len(common):
            wi = np.searchsorted(frames, common)
            si = np.searchsorted(sf, common)
            d = np.linalg.norm(xyz[wi] - sxyz[si], axis=1)
            m["reach_p95_m"] = float(np.percentile(d, 95))

    # smoothness: finite differences over runs of adjacent present frames, so a
    # dropout never manufactures a huge fake velocity spike. Frame numbers are
    # the source-video index and may step by >1 (iPhone captures step by 2), so
    # "adjacent" is one modal step apart, not literally +1; a dropped low-conf
    # frame then reads as a >step gap and breaks the run. dt = 1/fps regardless,
    # since --fps is the sample rate.
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    step = max(1, step)
    dt = 1.0 / fps
    vel, acc, jerk = [], [], []
    run = []
    prev_f = None
    idx = np.where(ok)[0]
    for i in idx:
        fnum = frames[i]
        if prev_f is not None and fnum != prev_f + step:
            _accum(run, xyz, dt, vel, acc, jerk)
            run = []
        run.append(i)
        prev_f = fnum
    _accum(run, xyz, dt, vel, acc, jerk)

    if vel:
        v = np.concatenate(vel)
        m["speed_p50_mm_s"] = float(np.median(v) * 1000.0)
    if acc:
        a = np.concatenate(acc)
        m["jitter_mm"] = float(np.median(np.abs(a)) * (dt * dt) * 1000.0)
        m["accel_rms_mm_s2"] = float(np.sqrt(np.mean(a * a)) * 1000.0)
    if jerk:
        jk = np.concatenate(jerk)
        m["jerk_rms_mm_s3"] = float(np.sqrt(np.mean(jk * jk)) * 1000.0)
    return m


def _accum(run, xyz, dt, vel, acc, jerk):
    """Append |velocity|, second-difference (accel*dt^2 basis) and third-diff
    magnitudes for one run of >=2 / >=3 / >=4 consecutive present indices."""
    if len(run) < 2:
        return
    p = xyz[run]
    d1 = np.diff(p, axis=0) / dt
    vel.append(np.linalg.norm(d1, axis=1))
    if len(run) >= 3:
        a = (p[2:] - 2 * p[1:-1] + p[:-2]) / (dt * dt)
        acc.append(np.linalg.norm(a, axis=1))
    if len(run) >= 4:
        jk = (p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3]) / (dt ** 3)
        jerk.append(np.linalg.norm(jk, axis=1))


ORDER = ["n", "dropout_pct", "reach_p95_m", "speed_p50_mm_s",
         "jitter_mm", "accel_rms_mm_s2", "jerk_rms_mm_s3"]
FMT = {"n": "{:.0f}", "dropout_pct": "{:.1f}", "reach_p95_m": "{:.3f}",
       "speed_p50_mm_s": "{:.0f}", "jitter_mm": "{:.2f}",
       "accel_rms_mm_s2": "{:.0f}", "jerk_rms_mm_s3": "{:.0f}"}


def print_one(label, m):
    print(f"  {label}")
    for k in ORDER:
        if k in m:
            print(f"    {k:16} {FMT[k].format(m[k])}")


def print_compare(joint, ma, mb, la, lb):
    print(f"  {joint}")
    print(f"    {'metric':16} {la:>12} {lb:>12} {'ratio a/b':>10}")
    for k in ORDER:
        if k in ma or k in mb:
            va, vb = ma.get(k), mb.get(k)
            sa = FMT[k].format(va) if va is not None else "-"
            sb = FMT[k].format(vb) if vb is not None else "-"
            rat = f"{va / vb:.2f}" if (va and vb) else "-"
            print(f"    {k:16} {sa:>12} {sb:>12} {rat:>10}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="joints CSV (frame,joint,x,y,z,conf)")
    ap.add_argument("--compare", help="second CSV to tabulate beside the first")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="frame rate for the physical accel/jerk (iPhone ~30, "
                         "ZED HD720@60). Set it per clip when comparing.")
    ap.add_argument("--compare-fps", type=float, default=None,
                    help="fps for the --compare clip if it differs")
    ap.add_argument("--conf-min", type=float, default=0.3,
                    help="a joint below this confidence counts as a dropout")
    ap.add_argument("--joints", default="left_wrist,right_wrist",
                    help="comma-separated joints to report")
    args = ap.parse_args()

    joints = [j.strip() for j in args.joints.split(",") if j.strip()]
    da = load(args.csv)
    print(f"[quality] {args.csv}  (fps={args.fps:g}, conf-min={args.conf_min})")
    if not args.compare:
        for j in joints:
            m = joint_metrics(da, j, args.fps, args.conf_min)
            if m is None:
                print(f"  {j}: ABSENT from this capture")
            else:
                print_one(j, m)
        return

    db = load(args.compare)
    bfps = args.compare_fps if args.compare_fps else args.fps
    print(f"[quality] vs {args.compare}  (fps={bfps:g})")
    print(f"  a = {args.csv}")
    print(f"  b = {args.compare}")
    for j in joints:
        ma = joint_metrics(da, j, args.fps, args.conf_min)
        mb = joint_metrics(db, j, bfps, args.conf_min)
        if ma is None and mb is None:
            print(f"  {j}: ABSENT from both")
            continue
        print_compare(j, ma or {}, mb or {}, "a", "b")


if __name__ == "__main__":
    main()
