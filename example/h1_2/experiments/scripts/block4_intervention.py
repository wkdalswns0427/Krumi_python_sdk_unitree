#!/usr/bin/env python3
"""block4_intervention.py - Block 4 as an intervention study, not a measurement.

The old Block 4 question was "do mono and RGBD agree on RULA/REBA risk
category". Now that F1 (depth is the noisy monocular axis) exists, the sharper
question is causal: does the axis-aware depth conditioning MAKE them agree?
That is what this measures, on all captures, at zero robot cost.

Three conditions per capture (RULA and REBA, each scored by rula_reba_score):
  (i)   mono raw               vs RGBD raw
  (ii)  mono depth-conditioned vs RGBD raw
  (iii) mono depth-conditioned vs RGBD depth-conditioned   (CONTROL)

If (ii) > (i), the conditioning improves agreement. Condition (iii) is the
control: a filter that merely flattens both signals toward their mean would
inflate agreement here too, exposing a fake result. The depth conditioning is
the SAME 0.8 Hz depth-axis low-pass used in retargeting (finding F1), a fixed
parameter tuned on nothing here; per-motion output lets held-out motions
(M2 bricklaying, M3 lifting, if the policy was set on M1) be read separately.

GUARD, reported not assumed: the trunk-flexion distribution before vs after
conditioning. If the low-pass biases trunk angle toward its mean it could
manufacture category agreement that will not generalize, so the distribution
shift is printed and must stay small.

Usage:
    /usr/bin/python3 block4_intervention.py \
        --caps M1_R_1 M1_R_2 M1_R_3 M2_1 M2_2 M2_3 M3_1 M3_2 M3_3 \
        --motion-of "M1_R_1=M1,M1_R_2=M1,...,M2_1=M2,...,M3_3=M3" \
        --out-dir $IROS/block4
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)          # scripts/ -> experiments/ (data root)
sys.path.insert(0, HERE)
import rula_reba_score as S              # noqa: E402
import rula_reba_agreement as AG         # noqa: E402
from retarget_arm import filter_depth_axis  # noqa: E402
import angles as A                       # noqa: E402


def load_full(path):
    """{frame: {joint: np.array xyz}} and {frame: {joint: conf}}."""
    pos = defaultdict(dict)
    conf = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_conf = "conf" in (reader.fieldnames or [])
        for r in reader:
            fi = int(float(r["frame"]))
            pos[fi][r["joint"].strip()] = np.array(
                [float(r["x"]), float(r["y"]), float(r["z"])])
            if has_conf and r.get("conf", "") != "":
                conf[fi][r["joint"].strip()] = float(r["conf"])
    return pos, conf


def condition_depth(pos, fps, cutoff):
    """Apply the F1 depth-axis low-pass to a copy of the frames dict."""
    cp = {fi: {j: v.copy() for j, v in d.items()} for fi, d in pos.items()}
    filter_depth_axis(cp, fps, cutoff)
    return cp


def score_frames(pos, conf, conf_min, up):
    """{frame: (rula_cat, reba_cat)} using the scorer + agreement banding."""
    ru, re = {}, {}
    for fi in sorted(pos):
        ang = A.compute_frame_angles(pos[fi], conf.get(fi) or None, conf_min, up)
        legs = A.legs_support_info(pos[fi], conf.get(fi) or None, conf_min, up)
        ru[fi] = AG.band("rula", S.rula_grand(ang, legs))
        re[fi] = AG.band("reba", S.reba_grand(ang, legs))
    return ru, re


def agreement(mono_cat, rgbd_cat, metric):
    cats = AG.CATEGORIES[metric]
    mat, _ = AG.confusion(mono_cat, rgbd_cat, cats)
    total = int(mat.sum())
    agree = int(np.trace(mat))
    return (agree / total * 100.0 if total else float("nan"),
            AG.cohen_kappa(mat), mat, total)


def trunk_dist(pos, conf, conf_min, up):
    vals = []
    for fi in sorted(pos):
        t = A.trunk_flexion(pos[fi], conf.get(fi) or None, conf_min, up)
        if t is not None:
            vals.append(t)
    return np.array(vals)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caps", nargs="+", required=True)
    p.add_argument("--motion-of", required=True,
                   help="cap=motion comma list, e.g. M1_R_1=M1,M2_1=M2")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--z-cutoff", type=float, default=0.8)
    p.add_argument("--conf-min", type=float, default=0.5)
    p.add_argument("--up", default="0,-1,0")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    up = tuple(float(v) for v in args.up.split(","))
    motion_of = dict(kv.split("=") for kv in args.motion_of.split(","))
    os.makedirs(args.out_dir, exist_ok=True)

    # per capture: agreement in the 3 conditions + trunk-dist shift
    per_cap = {}
    trunk_shift = {}
    for cap in args.caps:
        base = os.path.join(EXP_DIR, "iphone_data", cap, "b2g")
        mp = os.path.join(base, "iph_mono.csv")
        rp = os.path.join(base, "iph_rgbd.csv")
        if not (os.path.isfile(mp) and os.path.isfile(rp)):
            print(f"  skip {cap}: missing mono or rgbd joints")
            continue
        mpos, mconf = load_full(mp)
        rpos, rconf = load_full(rp)
        mcond = condition_depth(mpos, args.fps, args.z_cutoff)
        rcond = condition_depth(rpos, args.fps, args.z_cutoff)

        m_raw = score_frames(mpos, mconf, args.conf_min, up)
        m_cnd = score_frames(mcond, mconf, args.conf_min, up)
        r_raw = score_frames(rpos, rconf, args.conf_min, up)
        r_cnd = score_frames(rcond, rconf, args.conf_min, up)

        conds = {}
        for metric, mi in (("rula", 0), ("reba", 1)):
            conds[metric] = {
                "i": agreement(m_raw[mi], r_raw[mi], metric),
                "ii": agreement(m_cnd[mi], r_raw[mi], metric),
                "iii": agreement(m_cnd[mi], r_cnd[mi], metric),
            }
        per_cap[cap] = conds
        t0 = trunk_dist(mpos, mconf, args.conf_min, up)
        t1 = trunk_dist(mcond, mconf, args.conf_min, up)
        trunk_shift[cap] = (t0, t1)

    # ── report ──
    print("=" * 78)
    print("BLOCK 4 intervention: does depth conditioning make mono agree with RGBD?")
    print("Category agreement % (RULA / REBA), per capture, 3 conditions")
    print("  (i) mono_raw vs rgbd_raw   (ii) mono_cond vs rgbd_raw   "
          "(iii) mono_cond vs rgbd_cond [control]")
    print("=" * 78)
    hdr = (f"  {'cap':8}{'mot':4}"
           f"{'RULA i':>8}{'ii':>6}{'iii':>6}   {'REBA i':>8}{'ii':>6}{'iii':>6}")
    print(hdr)
    by_motion = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rows_csv = []
    for cap in args.caps:
        if cap not in per_cap:
            continue
        mot = motion_of.get(cap, "?")
        c = per_cap[cap]
        def g(metric, k):
            return c[metric][k][0]
        print(f"  {cap:8}{mot:4}"
              f"{g('rula','i'):>8.0f}{g('rula','ii'):>6.0f}{g('rula','iii'):>6.0f}   "
              f"{g('reba','i'):>8.0f}{g('reba','ii'):>6.0f}{g('reba','iii'):>6.0f}")
        for metric in ("rula", "reba"):
            for k in ("i", "ii", "iii"):
                by_motion[mot][metric][k].append(c[metric][k][0])
                rows_csv.append(dict(cap=cap, motion=mot, metric=metric,
                                     condition=k, agreement_pct=round(c[metric][k][0], 2),
                                     kappa=round(c[metric][k][1], 4),
                                     n=c[metric][k][3]))
    print("-" * 78)
    print("  Per-motion mean agreement % (held-out motions read separately):")
    print(f"  {'motion':8}{'RULA i':>8}{'ii':>6}{'iii':>6}   "
          f"{'REBA i':>8}{'ii':>6}{'iii':>6}   delta(ii-i)")
    for mot in sorted(by_motion):
        r = by_motion[mot]["rula"]
        e = by_motion[mot]["reba"]
        rm = {k: np.mean(r[k]) for k in ("i", "ii", "iii")}
        em = {k: np.mean(e[k]) for k in ("i", "ii", "iii")}
        print(f"  {mot:8}{rm['i']:>8.0f}{rm['ii']:>6.0f}{rm['iii']:>6.0f}   "
              f"{em['i']:>8.0f}{em['ii']:>6.0f}{em['iii']:>6.0f}   "
              f"RULA {rm['ii'] - rm['i']:+.0f}  REBA {em['ii'] - em['i']:+.0f}")
    print("-" * 78)
    print("  GUARD - trunk-flexion distribution shift under conditioning "
          "(mean / std, deg):")
    print(f"  {'cap':8}{'raw mean':>10}{'cond mean':>11}{'raw std':>9}"
          f"{'cond std':>10}{'mean shift':>12}")
    for cap in args.caps:
        if cap not in trunk_shift:
            continue
        t0, t1 = trunk_shift[cap]
        if len(t0) == 0:
            continue
        print(f"  {cap:8}{t0.mean():>10.1f}{t1.mean():>11.1f}{t0.std():>9.1f}"
              f"{t1.std():>10.1f}{t1.mean() - t0.mean():>12.2f}")
    print("  (a large std collapse or mean shift = the filter is flattening the")
    print("   trunk signal, which would manufacture agreement; expect small)")
    print("=" * 78)

    out = os.path.join(args.out_dir, "block4_intervention.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"wrote {out}")

    # per-motion pooled confusion matrices (conditions i and ii), each metric
    print("-" * 78)
    print("  Confusion matrices (rows = mono, cols = RGBD), pooled per motion:")
    for mot in sorted(by_motion):
        for metric in ("rula", "reba"):
            cats = AG.CATEGORIES[metric]
            for k in ("i", "ii"):
                mat = np.zeros((len(cats), len(cats)), dtype=int)
                for cap in args.caps:
                    if cap in per_cap and motion_of.get(cap) == mot:
                        mat += per_cap[cap][metric][k][2]
                path = os.path.join(
                    args.out_dir, f"confusion_{mot}_{metric}_{k}.csv")
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([f"mono\\rgbd"] + [f"cat_{c}" for c in cats])
                    for i, c in enumerate(cats):
                        w.writerow([f"cat_{c}"] + list(mat[i]))
    print(f"  wrote confusion_<motion>_<metric>_<i|ii>.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
