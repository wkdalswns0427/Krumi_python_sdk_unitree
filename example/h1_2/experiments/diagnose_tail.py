#!/usr/bin/env python3
"""diagnose_tail.py - explain the worst mono-vs-rgbd joint errors.

Ranks the largest per-(frame,joint) errors from a Block 1 comparison and, for
each, shows the raw rgbd depth versus the body-plane depth and the native
depth patch around the joint (min/median/max + Stray Scanner confidence).
If the sampled z jumped to the background while the nearest in-patch surface
sits near the body plane, that is depth edge bleed on an extended limb.

Usage:
    python3 diagnose_tail.py CAPDIR/b2g/iph_mono.csv CAPDIR/b2g/iph_rgbd.csv \
        [--capture-dir CAPDIR] [--top 10]
"""

import argparse
import csv
import os
import sys

import numpy as np

import block1_compare as bc


def load_intrinsics(capdir):
    k = np.loadtxt(os.path.join(capdir, "camera_matrix.csv"),
                   delimiter=",").flatten()
    return k[0], k[4], k[2], k[5]          # fx, fy, cx, cy


def worst_errors(mono, rgbd, top, conf_excl):
    """Replicate block1_compare --procrustes and rank per-joint errors (mm)."""
    recs = []
    for fi in sorted(set(mono) & set(rgbd)):
        fm, fr = mono[fi], rgbd[fi]
        confs = [fm.conf.get(j, 1.0) for j in bc.EVAL_JOINTS if j in fm.pos]
        if confs and min(confs) < conf_excl:          # same mono exclusion
            continue
        joints = [j for j in bc.EVAL_JOINTS if j in fm.pos and j in fr.pos]
        if len(joints) < 3:
            continue
        M = np.array([fm.pos[j] for j in joints])
        G = np.array([fr.pos[j] for j in joints])
        Ma = bc._procrustes_fit(M - M.mean(0), G - G.mean(0), with_scale=False)
        Gc = G - G.mean(0)
        for k, j in enumerate(joints):
            recs.append((float(np.linalg.norm(Ma[k] - Gc[k])), fi, j))
    recs.sort(reverse=True)
    return recs[:top]


def native_patch(capdir, fi, u_native, v_native, r=1):
    import cv2
    dpath = os.path.join(capdir, "depth", f"{fi:06d}.png")
    cpath = os.path.join(capdir, "confidence", f"{fi:06d}.png")
    if not os.path.isfile(dpath):
        return None
    d = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
    c = cv2.imread(cpath, cv2.IMREAD_UNCHANGED) if os.path.isfile(cpath) else None
    h, w = d.shape
    ui, vi = int(round(u_native)), int(round(v_native))
    u0, u1 = max(0, ui - r), min(w, ui + r + 1)
    v0, v1 = max(0, vi - r), min(h, vi + r + 1)
    dp = d[v0:v1, u0:u1].astype(float)
    cp = c[v0:v1, u0:u1] if c is not None else None
    nz = dp[dp > 0]
    return dict(dmin=nz.min() / 1000 if nz.size else float("nan"),
                dmed=np.median(nz) / 1000 if nz.size else float("nan"),
                dmax=nz.max() / 1000 if nz.size else float("nan"),
                conf=sorted(set(cp.flatten().tolist())) if cp is not None else None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mono_csv")
    p.add_argument("rgbd_csv")
    p.add_argument("--capture-dir", help="default: two levels up from rgbd_csv")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--conf-excl", type=float, default=0.5)
    p.add_argument("--depth-res", type=int, nargs=2, default=[256, 192])
    p.add_argument("--color-res", type=int, nargs=2, default=[1920, 1440])
    args = p.parse_args()

    capdir = args.capture_dir or os.path.dirname(os.path.dirname(
        os.path.abspath(args.rgbd_csv)))
    fx, fy, cx, cy = load_intrinsics(capdir)
    dw, dh = args.depth_res
    cw, ch = args.color_res

    mono = bc.load_frames(args.mono_csv, "m")   # positions in mm
    rgbd = bc.load_frames(args.rgbd_csv, "m")
    top = worst_errors(mono, rgbd, args.top, args.conf_excl)

    print(f"capture: {capdir}")
    print(f"{'err_mm':>8} {'frame':>6} {'joint':<15} {'rgbd_z':>7} {'body_z':>7} "
          f"{'patch_min':>9} {'patch_med':>9} {'patch_max':>9}  conf")
    out = []
    for err, fi, j in top:
        fr = rgbd[fi]
        zj = fr.pos[j][2] / 1000.0
        body = [fr.pos[b][2] / 1000.0 for b in
                ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
                if b in fr.pos]
        body_z = float(np.median(body)) if body else float("nan")
        xm, ym, zm = fr.pos[j] / 1000.0
        u = fx * xm / zm + cx
        v = fy * ym / zm + cy
        patch = native_patch(capdir, fi, u * dw / cw, v * dh / ch) or {}
        print(f"{err:8.0f} {fi:6d} {j:<15} {zj:7.2f} {body_z:7.2f} "
              f"{patch.get('dmin', float('nan')):9.2f} "
              f"{patch.get('dmed', float('nan')):9.2f} "
              f"{patch.get('dmax', float('nan')):9.2f}  {patch.get('conf')}")
        out.append(dict(err_mm=round(err, 1), frame=fi, joint=j,
                        rgbd_z=round(zj, 3), body_z=round(body_z, 3),
                        patch_min=round(patch.get("dmin", float("nan")), 3),
                        patch_med=round(patch.get("dmed", float("nan")), 3),
                        patch_max=round(patch.get("dmax", float("nan")), 3)))

    outp = os.path.join(os.path.dirname(os.path.abspath(args.rgbd_csv)),
                        "tail_diagnosis.csv")
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {outp}")
    print("\nEdge bleed signature: rgbd_z ~ patch_max (background) while "
          "patch_min ~ body_z (the limb). Nearest/confidence sampling fixes it.")


if __name__ == "__main__":
    main()
