#!/usr/bin/env python3
"""block2_report.py - Block 2 aggregate: continuous tracking error, before/after.

Restructures Block 2 around the error-budget finding F2 (gravity sag). The
binary 5 cm success criterion is a coin flip near the feedforward's predicted
4 cm, so the PRIMARY result is the continuous end-effector tracking error and
its improvement WITH vs WITHOUT the gravity feedforward. Binary success is
reported as a secondary at a tolerance frozen from the post-fix pilot.

Inputs: one or more replay logs, each tagged motion:condition:path, where
condition is off (no gravity feedforward) or on. Example:
    block2_report.py --tolerance 0.05 \
        --log M1:off:$IROS/training_data/replay_M1_T1_pilot_20260713.csv \
        --log M1:on:$IROS/block2/replay_M1_T2.csv \
        --out $IROS/block2/block2_summary.csv

PRIMARY table (per motion): mean / median / p95 / max continuous EE tracking
error (FK of executed joints vs FK of the commanded/retargeted reference),
off vs on, and the paired improvement.

SECONDARY: fraction of frames within --tolerance (CLI, not baked in).

F2 evidence (the gravity story, free from the same logs):
  - per-joint mean/max commanded-vs-executed error (deg)
  - the one-signed fraction per joint (sag signature; ~97% on shoulder
    pitch in the off pilot, expected to fall with the feedforward on)
  - at static holds, the measured holding torque (kp*err + commanded
    feedforward tau) vs the URDF gravity model dU/dq: correlation, sign
    agreement, per-arm magnitude scale

Self-test builds synthetic off/on logs and checks the before/after math:
    block2_report.py --self-test
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.expanduser("~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments")
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
from h12_experiments.fk_h12 import H12ArmFK              # noqa: E402
from h12_experiments.joints import ARM_CHAIN            # noqa: E402
from replay_arm import GravityFF, KP, FROZEN_TOLERANCE_M  # noqa: E402

SIDES = ("left", "right")
STATIC_VEL = 0.15        # rad/s: below this the joint is "holding"


def read_log(path):
    """Columns as dict of np arrays; missing tau -> nan."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"{path}: empty")
    cols = rows[0].keys()

    def col(name):
        if name not in cols:
            return None
        return np.array([float(r[name]) if r[name] not in ("", "nan")
                         else np.nan for r in rows])
    return {c: col(c) for c in cols}, cols


def paired_mask(d, side):
    """Frames where all of this arm's cmd+exe are finite."""
    chain = ARM_CHAIN[side]
    ok = np.ones(len(d["t_rel"]), dtype=bool)
    for j in chain:
        ok &= np.isfinite(d[f"cmd_{j}"]) & np.isfinite(d[f"exe_{j}"])
    return ok


def ee_tracking(fk, d, side, mask):
    chain = ARM_CHAIN[side]
    qc = np.stack([d[f"cmd_{j}"][mask] for j in chain], axis=1)
    qe = np.stack([d[f"exe_{j}"][mask] for j in chain], axis=1)
    ref = np.array([fk.ee_position(qc[i]) for i in range(len(qc))])
    exe = np.array([fk.ee_position(qe[i]) for i in range(len(qe))])
    return np.linalg.norm(exe - ref, axis=1)             # metres per frame


def joint_stats(d, side, mask):
    chain = ARM_CHAIN[side]
    out = {}
    for j in chain:
        err = d[f"cmd_{j}"][mask] - d[f"exe_{j}"][mask]
        out[j] = dict(mean=float(np.degrees(np.abs(err)).mean()),
                      max=float(np.degrees(np.abs(err)).max()),
                      one_signed=float(max(np.mean(err > 0),
                                           np.mean(err < 0))))
    return out


def gravity_f2(d, side, mask, gff):
    """At static holds: measured holding torque (kp*err + cmdtau) vs model
    dU/dq. Returns dict per shoulder-pitch/elbow with corr, sign, scale."""
    chain = ARM_CHAIN[side]
    qe = np.stack([d[f"exe_{j}"][mask] for j in chain], axis=1)
    t = d["t_rel"][mask]
    # executed shoulder-pitch velocity as the static-hold gate
    sp = qe[:, 0]
    vel = np.abs(np.gradient(sp, t)) if len(sp) > 3 else np.zeros(len(sp))
    static = vel < STATIC_VEL
    res = {}
    idx = {"shoulder_pitch": 0, "elbow": 3}
    for seg, k in idx.items():
        j = f"{side}_{seg}"
        kp = KP[seg]
        err = d[f"cmd_{j}"][mask] - d[f"exe_{j}"][mask]
        tau_ff = d.get(f"cmdtau_{j}")
        ff = (tau_ff[mask] if tau_ff is not None else np.zeros(len(err)))
        ff = np.nan_to_num(ff)
        measured = kp * err + ff                          # total holding torque
        model = np.array([gff.tau(qe[i])[k] for i in range(len(qe))])
        m = static & np.isfinite(measured) & np.isfinite(model)
        if m.sum() < 10:
            res[seg] = None
            continue
        mo, me = model[m], measured[m]
        corr = float(np.corrcoef(mo, me)[0, 1])
        slope = float(np.polyfit(mo, me, 1)[0])
        sign_ok = bool(np.sign(mo.mean()) == np.sign(me.mean()))
        res[seg] = dict(n=int(m.sum()), corr=corr, scale=slope,
                        sign_agree=sign_ok,
                        model_mean=float(mo.mean()),
                        measured_mean=float(me.mean()))
    return res


def analyze_log(path, side, tolerance, gff, urdf):
    d, cols = read_log(path)
    fk = H12ArmFK(urdf, side=side)
    mask = paired_mask(d, side)
    if mask.sum() < 2:
        sys.exit(f"{path}: <2 paired {side}-arm frames")
    dev = ee_tracking(fk, d, side, mask)
    return dict(
        path=path, n=int(mask.sum()),
        mean=float(dev.mean() * 100), median=float(np.median(dev) * 100),
        p95=float(np.percentile(dev, 95) * 100), max=float(dev.max() * 100),
        frac_within=float(np.mean(dev <= tolerance)),
        joints=joint_stats(d, side, mask),
        f2=gravity_f2(d, side, mask, gff))


# ── reporting ────────────────────────────────────────────────────────────────
def report(by_motion, tolerance, side):
    print("=" * 74)
    print(f"BLOCK 2 aggregate  ({side} arm)   tolerance {tolerance * 100:.1f} cm")
    print("=" * 74)
    print("PRIMARY: continuous EE tracking error (cm), gravity feedforward off vs on")
    print(f"  {'motion':7} {'cond':4} {'n':>5} {'mean':>7} {'median':>7} "
          f"{'p95':>7} {'max':>7} {'within%':>8}")
    rows_out = []
    for motion in sorted(by_motion):
        conds = by_motion[motion]
        for cond in ("off", "on"):
            if cond not in conds:
                continue
            r = conds[cond]
            print(f"  {motion:7} {cond:4} {r['n']:>5} {r['mean']:>7.2f} "
                  f"{r['median']:>7.2f} {r['p95']:>7.2f} {r['max']:>7.2f} "
                  f"{r['frac_within'] * 100:>7.1f}%")
            rows_out.append(dict(motion=motion, condition=cond, n=r["n"],
                                 mean_cm=round(r["mean"], 3),
                                 median_cm=round(r["median"], 3),
                                 p95_cm=round(r["p95"], 3),
                                 max_cm=round(r["max"], 3),
                                 frac_within_tol=round(r["frac_within"], 4)))
        if "off" in conds and "on" in conds:
            do = conds["off"]["mean"] - conds["on"]["mean"]
            print(f"  {motion:7} IMPROVEMENT mean {conds['off']['mean']:.2f}"
                  f" -> {conds['on']['mean']:.2f} cm  (delta {-do:+.2f}, "
                  f"{100 * do / conds['off']['mean']:+.0f}%)")
    print("-" * 74)
    print("F2 gravity evidence (shoulder_pitch, static holds): measured "
          "kp*err+cmdtau vs model dU/dq")
    print(f"  {'motion':7} {'cond':4} {'n':>5} {'corr':>6} {'sign':>5} "
          f"{'scale':>6} {'model':>8} {'measured':>9}")
    for motion in sorted(by_motion):
        for cond in ("off", "on"):
            if cond not in by_motion[motion]:
                continue
            f2 = by_motion[motion][cond]["f2"].get("shoulder_pitch")
            if not f2:
                continue
            print(f"  {motion:7} {cond:4} {f2['n']:>5} {f2['corr']:>6.2f} "
                  f"{str(f2['sign_agree']):>5} {f2['scale']:>6.2f} "
                  f"{f2['model_mean']:>7.1f}N {f2['measured_mean']:>8.1f}N")
    print("-" * 74)
    print("Per-joint |cmd-exe| and one-signed fraction (sag signature)")
    for motion in sorted(by_motion):
        for cond in ("off", "on"):
            if cond not in by_motion[motion]:
                continue
            print(f"  [{motion} {cond}]  {'joint':<20}{'mean deg':>9}"
                  f"{'max deg':>9}{'1-signed%':>10}")
            for j, s in by_motion[motion][cond]["joints"].items():
                print(f"           {j:<20}{s['mean']:>9.2f}{s['max']:>9.2f}"
                      f"{s['one_signed'] * 100:>10.0f}")
    print("=" * 74)
    return rows_out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", default=[],
                   help="motion:condition:path (condition = off|on), repeatable")
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--tolerance", type=float, default=FROZEN_TOLERANCE_M,
                   help=f"binary success tolerance (m); frozen "
                        f"{FROZEN_TOLERANCE_M} (secondary metric; 86%% of "
                        f"frames within it at the frozen gravity gain)")
    p.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"))
    p.add_argument("--out", help="summary CSV")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return self_test(args)
    if not args.log:
        sys.exit("provide at least one --log motion:condition:path")

    gff = {s: GravityFF(args.urdf, s) for s in SIDES}[args.side]
    by_motion = {}
    for spec in args.log:
        parts = spec.split(":", 2)
        if len(parts) != 3 or parts[1] not in ("off", "on"):
            sys.exit(f"bad --log '{spec}'; want motion:off|on:path")
        motion, cond, path = parts
        if not os.path.isfile(path):
            sys.exit(f"log not found: {path}")
        by_motion.setdefault(motion, {})[cond] = analyze_log(
            path, args.side, args.tolerance, gff, args.urdf)

    rows = report(by_motion, args.tolerance, args.side)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


def self_test(args):
    """Synthetic off/on logs: 'on' cancels most of a modeled sag. Checks the
    before/after improvement is detected and positive."""
    import tempfile
    from replay_arm import SDK_INDEX  # noqa: F401  (import sanity)
    from h12_experiments.joints import LOG_JOINTS
    urdf = args.urdf
    gff = GravityFF(urdf, "right")
    d = tempfile.mkdtemp(prefix="b2report_")
    rng = np.random.default_rng(0)

    def make(path, sag_frac):
        n = 400
        t = np.linspace(0, 8, n)
        chain = ARM_CHAIN["right"]
        hdr = (["t_wall", "t_rel", "weight", "mode_machine"]
               + [f"cmd_{j}" for j in LOG_JOINTS]
               + [f"exe_{j}" for j in LOG_JOINTS]
               + [f"err_{j}" for j in LOG_JOINTS]
               + [f"cmdtau_{j}" for j in LOG_JOINTS])
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            for i in range(n):
                cmd = {j: 0.5 * math.sin(0.3 * t[i]) for j in LOG_JOINTS}
                q = [cmd[j] for j in chain]
                tau_g = gff.tau(q)                       # model gravity torque
                exe = dict(cmd)
                tau_ff = {j: 0.0 for j in LOG_JOINTS}
                for k, seg in enumerate(("shoulder_pitch", "shoulder_roll",
                                         "shoulder_yaw", "elbow")):
                    j = f"right_{seg}"
                    kp = KP[seg]
                    # residual sag after feedforward supplies (1-sag_frac)
                    ff = (1 - sag_frac) * tau_g[k]
                    resid_tau = tau_g[k] - ff
                    exe[j] = cmd[j] - resid_tau / kp + rng.normal(0, 0.002)
                    tau_ff[j] = ff
                row = (["0", f"{t[i]:.4f}", "1.0", "6"]
                       + [f"{cmd[j]:.6f}" for j in LOG_JOINTS]
                       + [f"{exe[j]:.6f}" for j in LOG_JOINTS]
                       + [f"{cmd[j] - exe[j]:.6f}" for j in LOG_JOINTS]
                       + [f"{tau_ff[j]:.6f}" for j in LOG_JOINTS])
                w.writerow(row)

    off, on = os.path.join(d, "off.csv"), os.path.join(d, "on.csv")
    make(off, sag_frac=1.0)      # no feedforward: full sag
    make(on, sag_frac=0.1)       # feedforward cancels 90%
    by_motion = {"M1": {
        "off": analyze_log(off, "right", args.tolerance, gff, urdf),
        "on": analyze_log(on, "right", args.tolerance, gff, urdf)}}
    report(by_motion, args.tolerance, "right")
    off_mean = by_motion["M1"]["off"]["mean"]
    on_mean = by_motion["M1"]["on"]["mean"]
    f2 = by_motion["M1"]["off"]["f2"]["shoulder_pitch"]
    assert on_mean < 0.5 * off_mean, "feedforward should cut EE error > 2x"
    assert f2["corr"] > 0.9 and f2["sign_agree"], "F2 corr/sign should hold"
    print(f"\n[self-test] PASS: off {off_mean:.2f} -> on {on_mean:.2f} cm, "
          f"F2 corr {f2['corr']:.2f} sign_ok {f2['sign_agree']}")


if __name__ == "__main__":
    main()
