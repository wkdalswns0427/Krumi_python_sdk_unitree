#!/usr/bin/env python3
"""block1_compare.py - per-joint 3D pose error, monocular vs RGBD reference.

Block 1 of the Humanoids 2026 protocol. Takes two per-frame 3D joint
position files (the MONOCULAR pipeline output and the RGBD reference),
aligns both to the pelvis/root frame, and reports the relative 3D joint
error per joint.

Claim language (per protocol): this is "relative 3D joint error against an
RGBD reference," NOT ground-truth accuracy.

Inputs (auto-detected by extension, see experiments/README.md):
  * canonical long CSV:  frame,joint,x,y,z[,conf]
  * MediaPipe world-landmark JSON (json2pose.py motion format)

Evaluated joints (6): left/right shoulder, elbow, wrist.
Root for alignment: pelvis (or midpoint of left_hip/right_hip).

Usage:
    python3 block1_compare.py MONO.csv RGBD.csv --out per_joint_error.csv
    python3 block1_compare.py mono.json rgbd.json --conf-threshold 0.5
    python3 block1_compare.py --demo          # self-test on synthetic data

Only the MONO file's confidence is used for the exclusion rule.
"""

import argparse
import csv
import json
import os
import sys
import tempfile

import numpy as np


# Block 1 scores the 6 retargeted arm joints. --joints full widens the scored
# set to the whole exported skeleton (bag_to_joints --joints full); the arm
# default keeps Block 1 numbers unaffected by the schema extension.
ARM_EVAL_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
FULL_EVAL_JOINTS = ARM_EVAL_JOINTS + [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
# The active scored set. Mutated IN PLACE (never rebound) by main() so every
# module reference sees the selection.
EVAL_JOINTS = list(ARM_EVAL_JOINTS)

# MediaPipe Pose world-landmark index -> canonical joint name.
# Face landmarks (0, 7, 8) are excluded from the pipeline export.
MP_INDEX_TO_JOINT = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow",    14: "right_elbow",
    15: "left_wrist",    16: "right_wrist",
    23: "left_hip",      24: "right_hip",
    25: "left_knee",     26: "right_knee",
    27: "left_ankle",    28: "right_ankle",
}

UNIT_TO_MM = {"m": 1000.0, "cm": 10.0, "mm": 1.0}


class Frame:
    """One frame's joint positions (mm) and per-joint confidence."""

    __slots__ = ("pos", "conf", "pelvis")

    def __init__(self):
        self.pos = {}      # joint -> np.array([x, y, z]) in mm
        self.conf = {}     # joint -> float in [0, 1]
        self.pelvis = None  # np.array([x, y, z]) in mm, or None


# ── Loading ──────────────────────────────────────────────────────────────────
def _derive_pelvis(pos):
    if "pelvis" in pos:
        return pos["pelvis"]
    if "left_hip" in pos and "right_hip" in pos:
        return (pos["left_hip"] + pos["right_hip"]) / 2.0
    return None


def load_csv(path, scale):
    """Canonical long CSV -> {frame_index: Frame}."""
    frames = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame", "joint", "x", "y", "z"}
        if not required.issubset(reader.fieldnames or []):
            sys.exit(f"{path}: CSV must have columns {sorted(required)}; "
                     f"got {reader.fieldnames}")
        has_conf = "conf" in (reader.fieldnames or [])
        for row in reader:
            fi = int(float(row["frame"]))
            fr = frames.setdefault(fi, Frame())
            joint = row["joint"].strip()
            p = np.array([float(row["x"]), float(row["y"]), float(row["z"])],
                         dtype=float) * scale
            fr.pos[joint] = p
            if joint in EVAL_JOINTS:
                fr.conf[joint] = float(row["conf"]) if has_conf and row["conf"] != "" else 1.0
    for fr in frames.values():
        fr.pelvis = _derive_pelvis(fr.pos)
    return frames


def load_mediapipe_json(path, scale):
    """json2pose.py motion JSON -> {frame_index: Frame}. Positions in metres."""
    with open(path) as f:
        data = json.load(f)
    if not (isinstance(data, list) and data and isinstance(data[0], dict)
            and "landmarks" in data[0]):
        sys.exit(f"{path}: not a MediaPipe motion JSON (list of frames with "
                 f"'landmarks'). Use the canonical CSV format instead.")
    frames = {}
    for i, frame in enumerate(data):
        if not frame.get("pose_detected", True) or not frame.get("landmarks"):
            continue
        fi = int(frame.get("frame_index", i))
        fr = frames.setdefault(fi, Frame())
        for lm in frame["landmarks"]:
            idx = lm.get("landmark_index")
            joint = MP_INDEX_TO_JOINT.get(idx)
            if joint is None:
                continue
            fr.pos[joint] = np.array([lm["x"], lm["y"], lm["z"]], float) * scale
            if joint in EVAL_JOINTS:
                fr.conf[joint] = float(
                    lm.get("visibility", lm.get("presence", 1.0)))
        fr.pelvis = _derive_pelvis(fr.pos)
    return frames


def load_frames(path, in_units):
    scale = UNIT_TO_MM[in_units]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return load_mediapipe_json(path, scale)
    return load_csv(path, scale)


# ── Alignment ────────────────────────────────────────────────────────────────
def _root_offset(fr, mode):
    if mode == "pelvis":
        return fr.pelvis  # may be None -> caller handles
    if mode == "centroid":
        pts = [fr.pos[j] for j in EVAL_JOINTS if j in fr.pos]
        return np.mean(pts, axis=0) if pts else None
    return np.zeros(3)  # mode == "none"


def _procrustes_fit(src, dst, with_scale):
    """Best-fit rotation (Kabsch), optionally with scale, mapping src->dst.
    Both Nx3 and already centred. Returns the transformed src.

    with_scale=True gives the standard Procrustes-aligned (PA-MPJPE) fit
    used in 3D human-pose work: rotation + isotropic scale + translation.
    """
    H = src.T @ dst
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt                     # applied as (src @ R) ≈ dst
    s = 1.0
    if with_scale:
        var = float((src ** 2).sum())
        s = (float(S @ np.diag(D)) / var) if var > 0 else 1.0
    return s * (src @ R)


# ── Comparison ───────────────────────────────────────────────────────────────
def compare(mono, rgbd, root_mode, conf_thresh, conf_reduce, align):
    common = sorted(set(mono) & set(rgbd))
    errors = {j: [] for j in EVAL_JOINTS}
    n_common = len(common)
    n_excluded = 0
    n_no_root = 0

    for fi in common:
        fm, fr = mono[fi], rgbd[fi]

        # MediaPipe confidence exclusion (mono file only).
        confs = [fm.conf.get(j, 1.0) for j in EVAL_JOINTS if j in fm.pos]
        if confs:
            c = min(confs) if conf_reduce == "min" else float(np.mean(confs))
            if c < conf_thresh:
                n_excluded += 1
                continue

        joints = [j for j in EVAL_JOINTS if j in fm.pos and j in fr.pos]
        if not joints:
            continue
        Mraw = np.array([fm.pos[j] for j in joints])
        Graw = np.array([fr.pos[j] for j in joints])

        if align == "none":
            # Plain root-relative error: translate both to the pelvis frame.
            om = _root_offset(fm, root_mode)
            og = _root_offset(fr, root_mode)
            if om is None or og is None:
                n_no_root += 1
                continue
            M, G = Mraw - om, Graw - og
        else:
            # Procrustes (PA-MPJPE): centre each set on its OWN centroid, then
            # solve rotation (+scale). Kabsch requires centroid-centred data.
            G = Graw - Graw.mean(axis=0)
            M = Mraw - Mraw.mean(axis=0)
            if len(joints) >= 3:
                M = _procrustes_fit(M, G, with_scale=(align == "sim"))

        for k, j in enumerate(joints):
            errors[j].append(float(np.linalg.norm(M[k] - G[k])))

    return errors, dict(n_common=n_common, n_excluded=n_excluded,
                        n_no_root=n_no_root)


def axis_errors(mono, rgbd, conf_thresh, conf_reduce):
    """Per-joint per-axis abs error (mm) in the RGBD camera frame.

    Uses a SINGLE global Kabsch rotation fit over all matched frames, not the
    per-frame procrustes the scalar MPJPE uses. This is deliberate: a
    per-frame rotation reorients each frame to minimise total residual and so
    ROTATES the depth disagreement into the in-plane axes, hiding the very
    anisotropy F1 is about (measured: per-frame gives depth 0.67x in-plane,
    global gives 1.45x on M1_R_1). The global fit removes only the fixed
    mono-world vs rgbd-camera frame offset, preserving per-frame axis
    structure. Axes: [0]=horizontal (x), [1]=vertical (y), [2]=depth (z).
    """
    common = sorted(set(mono) & set(rgbd))
    Ms, Gs, Js = [], [], []
    for fi in common:
        fm, fr = mono[fi], rgbd[fi]
        confs = [fm.conf.get(j, 1.0) for j in EVAL_JOINTS if j in fm.pos]
        if confs:
            c = min(confs) if conf_reduce == "min" else float(np.mean(confs))
            if c < conf_thresh:
                continue
        js = [j for j in EVAL_JOINTS if j in fm.pos and j in fr.pos]
        if len(js) < 3:
            continue
        M = np.array([fm.pos[j] for j in js])
        G = np.array([fr.pos[j] for j in js])
        Ms.append(M - M.mean(axis=0))
        Gs.append(G - G.mean(axis=0))
        Js.append(js)
    errors_axis = {j: [] for j in EVAL_JOINTS}
    if not Ms:
        return errors_axis
    allM, allG = np.vstack(Ms), np.vstack(Gs)
    H = allM.T @ allG
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt                  # one global rotation
    for M, G, js in zip(Ms, Gs, Js):
        Ma = M @ R
        for k, j in enumerate(js):
            errors_axis[j].append(np.abs(Ma[k] - G[k]))
    return errors_axis


def summarize(errors):
    """Per-joint stats (mm) + overall (mean of per-joint means)."""
    rows = []
    joint_means = []
    for j in EVAL_JOINTS:
        e = np.array(errors[j], float)
        if e.size == 0:
            rows.append(dict(joint=j, n=0, mean=float("nan"), std=float("nan"),
                             median=float("nan"), p95=float("nan"), max=float("nan")))
            continue
        rows.append(dict(joint=j, n=int(e.size), mean=float(e.mean()),
                         std=float(e.std()), median=float(np.median(e)),
                         p95=float(np.percentile(e, 95)), max=float(e.max())))
        joint_means.append(e.mean())
    overall_mean = float(np.mean(joint_means)) if joint_means else float("nan")
    pooled = np.concatenate([np.array(errors[j], float) for j in EVAL_JOINTS]) \
        if any(errors[j] for j in EVAL_JOINTS) else np.array([])
    rows.append(dict(joint="OVERALL", n=int(pooled.size), mean=overall_mean,
                     std=float(pooled.std()) if pooled.size else float("nan"),
                     median=float(np.median(pooled)) if pooled.size else float("nan"),
                     p95=float(np.percentile(pooled, 95)) if pooled.size else float("nan"),
                     max=float(pooled.max()) if pooled.size else float("nan")))
    return rows, overall_mean


AXES = ("horizontal", "vertical", "depth")


def summarize_axes(errors_axis):
    """Per-joint per-axis stats (mm) + overall, + depth:in-plane ratio.

    In-plane error magnitude is sqrt(dx^2 + dy^2); depth is |dz|. The ratio
    of depth-axis to in-plane error is the quantitative backbone of F1.
    """
    rows = []
    pooled = {a: [] for a in AXES}
    pooled_inplane = []
    for j in EVAL_JOINTS:
        stacked = np.array(errors_axis[j], float)          # n x 3
        if stacked.size == 0:
            rows.append(dict(joint=j, n=0, **{a: float("nan") for a in AXES},
                             depth_inplane_ratio=float("nan")))
            continue
        per_axis = {a: stacked[:, i] for i, a in enumerate(AXES)}
        inplane = np.sqrt(stacked[:, 0] ** 2 + stacked[:, 1] ** 2)
        for i, a in enumerate(AXES):
            pooled[a].append(stacked[:, i])
        pooled_inplane.append(inplane)
        ratio = (per_axis["depth"].mean() / inplane.mean()
                 if inplane.mean() > 0 else float("nan"))
        row = dict(joint=j, n=int(stacked.shape[0]),
                   depth_inplane_ratio=float(ratio))
        for a in AXES:
            row[f"{a}_mean"] = float(per_axis[a].mean())
            row[f"{a}_median"] = float(np.median(per_axis[a]))
            row[f"{a}_p95"] = float(np.percentile(per_axis[a], 95))
        rows.append(row)

    ov = dict(joint="OVERALL")
    if pooled[AXES[0]]:
        cat = {a: np.concatenate(pooled[a]) for a in AXES}
        inp = np.concatenate(pooled_inplane)
        ov["n"] = int(cat["depth"].size)
        for a in AXES:
            ov[f"{a}_mean"] = float(cat[a].mean())
            ov[f"{a}_median"] = float(np.median(cat[a]))
            ov[f"{a}_p95"] = float(np.percentile(cat[a], 95))
        ov["depth_inplane_ratio"] = (float(cat["depth"].mean() / inp.mean())
                                     if inp.mean() > 0 else float("nan"))
    rows.append(ov)
    return rows


def print_axis_summary(rows):
    print("-" * 68)
    print("  Per-axis error in the RGBD camera frame (mm), mean [median]:")
    print(f"  {'joint':<16}{'horiz':>12}{'vert':>12}{'depth':>12}"
          f"{'depth/inplane':>14}")
    for r in rows:
        if "horizontal_mean" not in r:
            continue
        if r["joint"] == "OVERALL":
            print("  " + "-" * 64)
        print(f"  {r['joint']:<16}"
              f"{r['horizontal_mean']:>7.1f}[{r['horizontal_median']:>4.0f}]"
              f"{r['vertical_mean']:>7.1f}[{r['vertical_median']:>4.0f}]"
              f"{r['depth_mean']:>7.1f}[{r['depth_median']:>4.0f}]"
              f"{r['depth_inplane_ratio']:>14.2f}")
    print("=" * 68)
    ov = rows[-1]
    if "depth_inplane_ratio" in ov and ov["depth_inplane_ratio"] == ov["depth_inplane_ratio"]:
        print(f"  F1 backbone: overall depth-axis error is "
              f"{ov['depth_inplane_ratio']:.2f}x the in-plane error")
        print("=" * 68)


def write_axis_table(rows, path, tag):
    fields = ["joint", "n",
              "horizontal_mean_mm", "horizontal_median_mm", "horizontal_p95_mm",
              "vertical_mean_mm", "vertical_median_mm", "vertical_p95_mm",
              "depth_mean_mm", "depth_median_mm", "depth_p95_mm",
              "depth_inplane_ratio"]
    if tag is not None:
        fields = ["tag"] + fields
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            if "horizontal_mean" not in r:
                continue
            vals = [r["joint"], r["n"]]
            for a in AXES:
                vals += [f"{r[f'{a}_mean']:.3f}", f"{r[f'{a}_median']:.3f}",
                         f"{r[f'{a}_p95']:.3f}"]
            vals += [f"{r['depth_inplane_ratio']:.4f}"]
            if tag is not None:
                vals = [tag] + vals
            w.writerow(vals)


def write_table(rows, path, tag):
    fields = ["joint", "n", "mean_mm", "std_mm", "median_mm", "p95_mm", "max_mm"]
    if tag is not None:
        fields = ["tag"] + fields
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            vals = [r["joint"], r["n"], f"{r['mean']:.3f}", f"{r['std']:.3f}",
                    f"{r['median']:.3f}", f"{r['p95']:.3f}", f"{r['max']:.3f}"]
            if tag is not None:
                vals = [tag] + vals
            w.writerow(vals)


def print_summary(rows, meta, overall_mean, conf_thresh):
    total = meta["n_common"]
    used = total - meta["n_excluded"] - meta["n_no_root"]
    excl_rate = (meta["n_excluded"] / total * 100.0) if total else 0.0
    print("=" * 68)
    print("Block 1 - relative 3D joint error (monocular vs RGBD reference)")
    print("=" * 68)
    print(f"  common frames        : {total}")
    print(f"  excluded (conf<{conf_thresh:g}) : {meta['n_excluded']}  "
          f"({excl_rate:.1f}% exclusion rate)")
    if meta["n_no_root"]:
        print(f"  skipped (no root)    : {meta['n_no_root']}")
    print(f"  frames used          : {used}")
    print("-" * 68)
    print(f"  {'joint':<16}{'n':>7}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}  (mm)")
    for r in rows:
        sep = "  " + "-" * 66 if r["joint"] == "OVERALL" else ""
        if sep:
            print(sep)
        print(f"  {r['joint']:<16}{r['n']:>7}{r['mean']:>9.2f}"
              f"{r['median']:>9.2f}{r['p95']:>9.2f}{r['max']:>9.2f}")
    print("=" * 68)
    print(f"  ABSTRACT PLACEHOLDER (overall mean across joints): "
          f"{overall_mean:.2f} mm")
    print("=" * 68)


# ── Demo / self-test ─────────────────────────────────────────────────────────
def _write_demo_pair(dirpath):
    """Synthetic mono/rgbd CSV pair: rgbd is truth, mono = truth + noise."""
    rng = np.random.default_rng(0)
    base = {
        "pelvis": (0.0, 0.0, 3.0),
        "left_shoulder": (0.20, 0.55, 3.0), "right_shoulder": (-0.20, 0.55, 3.0),
        "left_elbow": (0.30, 0.30, 3.0), "right_elbow": (-0.30, 0.30, 3.0),
        "left_wrist": (0.35, 0.05, 3.0), "right_wrist": (-0.35, 0.05, 3.0),
    }
    mono_p = os.path.join(dirpath, "demo_mono.csv")
    rgbd_p = os.path.join(dirpath, "demo_rgbd.csv")
    with open(mono_p, "w", newline="") as fm, open(rgbd_p, "w", newline="") as fg:
        wm, wg = csv.writer(fm), csv.writer(fg)
        wm.writerow(["frame", "joint", "x", "y", "z", "conf"])
        wg.writerow(["frame", "joint", "x", "y", "z", "conf"])
        for fi in range(200):
            low_conf = (fi % 25 == 0)  # ~4% of frames below threshold
            for j, (x, y, z) in base.items():
                wg.writerow([fi, j, x, y, z, 1.0])
                noise = rng.normal(0, 0.015, 3)  # ~15 mm sigma
                conf = 0.2 if (low_conf and j in EVAL_JOINTS) else 0.95
                wm.writerow([fi, j, x + noise[0], y + noise[1], z + noise[2], conf])
    return mono_p, rgbd_p


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mono_file", nargs="?", help="monocular pipeline output")
    p.add_argument("rgbd_file", nargs="?", help="RGBD reference output")
    p.add_argument("--out", help="write per-joint error table to this CSV")
    p.add_argument("--out-axis", help="write the per-axis (horizontal/vertical/"
                   "depth) error table here (default: <out>_peraxis.csv)")
    p.add_argument("--tag", help="label column added to the output CSV rows "
                                 "(for later per-motion aggregation)")
    p.add_argument("--joints", choices=["arm", "full"], default="arm",
                   help="scored joint set: arm = the 6 retargeted joints "
                        "(Block 1 default), full = whole exported skeleton")
    p.add_argument("--in-units", choices=list(UNIT_TO_MM), default="m",
                   help="input position units (default: m)")
    p.add_argument("--root", choices=["pelvis", "centroid", "none"],
                   default="pelvis", help="root-alignment mode (default: pelvis)")
    p.add_argument("--conf-threshold", type=float, default=0.5,
                   help="exclude frames whose mono confidence is below this")
    p.add_argument("--conf-reduce", choices=["min", "mean"], default="min",
                   help="reduce per-joint mono conf to one value per frame")
    p.add_argument("--procrustes", action="store_true",
                   help="rotation-align (Kabsch) after root translation - use "
                        "for mono-world vs rgbd-camera frame mismatch")
    p.add_argument("--procrustes-scale", action="store_true",
                   help="rotation + isotropic scale align (standard PA-MPJPE); "
                        "removes global scale error too")
    p.add_argument("--demo", action="store_true",
                   help="run on generated synthetic data (no inputs needed)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.joints == "full":
        EVAL_JOINTS[:] = FULL_EVAL_JOINTS   # in place; see comment at definition

    tmp = None
    if args.demo:
        tmp = tempfile.mkdtemp(prefix="block1_demo_")
        args.mono_file, args.rgbd_file = _write_demo_pair(tmp)
        print(f"[demo] synthetic pair in {tmp}\n")
    elif not (args.mono_file and args.rgbd_file):
        sys.exit("error: provide MONO and RGBD files (or --demo)")

    for pth in (args.mono_file, args.rgbd_file):
        if not os.path.isfile(pth):
            sys.exit(f"error: file not found: {pth}")

    mono = load_frames(args.mono_file, args.in_units)
    rgbd = load_frames(args.rgbd_file, args.in_units)
    if not mono or not rgbd:
        sys.exit("error: one of the inputs has no usable frames")

    align = "sim" if args.procrustes_scale else ("rot" if args.procrustes else "none")
    errors, meta = compare(mono, rgbd, args.root, args.conf_threshold,
                           args.conf_reduce, align)
    rows, overall = summarize(errors)
    print_summary(rows, meta, overall, args.conf_threshold)
    errors_axis = axis_errors(mono, rgbd, args.conf_threshold, args.conf_reduce)
    axis_rows = summarize_axes(errors_axis)
    print_axis_summary(axis_rows)

    if args.out:
        write_table(rows, args.out, args.tag)
        print(f"\nWrote per-joint table: {args.out}")
    elif args.demo:
        out = os.path.join(tmp, "per_joint_error.csv")
        write_table(rows, out, args.tag)
        print(f"\n[demo] wrote per-joint table: {out}")

    # per-axis table goes to its own file so the default --out stays identical
    axis_out = args.out_axis
    if axis_out is None and args.out:
        base, ext = os.path.splitext(args.out)
        axis_out = base + "_peraxis" + ext
    elif axis_out is None and args.demo:
        axis_out = os.path.join(tmp, "per_axis_error.csv")
    if axis_out:
        write_axis_table(axis_rows, axis_out, args.tag)
        print(f"Wrote per-axis table: {axis_out}")


if __name__ == "__main__":
    main()
