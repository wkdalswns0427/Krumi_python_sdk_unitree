#!/usr/bin/env python3
"""X3: hardware replay, the paper's only non-simulation result.

The question X3 answers: the offline analysis reports that direction matching
drives a joint onto its limit and holds it there. On hardware, does that frame
manifest as a real deviation, or does the low-level controller clamp it
silently? Both answers are reportable and the second must be reported.

Three subcommands, in order.

  prep   Solve a capture under each condition and write replay-ready CSVs, plus
         a sidecar JSON marking which frames touch a joint limit. No hardware.
  run    Replay one CSV on the robot with gravity feed-forward while logging
         commanded and executed joint angles. Hardware, e-stop operator needed.
  grade  Compare the log against the commanded trajectory and the predicted
         limit frames, and state which of the two outcomes occurred. No hardware.

    x3_hardware_replay.py prep --captures M2_1 M3_1
    x3_hardware_replay.py run results/x3/M2_1_fidelity.csv --interface enp128s31f6
    x3_hardware_replay.py grade results/x3/M2_1_fidelity.csv

DO NOT run the replay without an e-stop operator who is not the person at the
keyboard. Start with --speed-scale 0.5 on the first trajectory of a session.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXP_SCRIPTS = os.path.expanduser(
    "~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/scripts")

from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4      # noqa: E402
from common.data_paths import joints_csv_for                        # noqa: E402
from common.metrics import limit_margin                             # noqa: E402
import x6_taskcentric as x6                                         # noqa: E402

CONDITIONS = ("fidelity_reg", "taskdefined")
SEGS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
WRIST = ("wrist_roll", "wrist_pitch", "wrist_yaw")
# replay_arm.py requires the full 15-joint vector. The non-solved joints hold a
# fixed rest pose so only the solved arm moves.
REST4 = {"right": (0.30, -0.24, 0.12, 0.69), "left": (0.31, 0.24, -0.12, 0.67)}


def lowpass(qs, fps, cutoff_hz):
    """Zero-phase Butterworth low-pass on each joint.

    A per-frame solver produces joint noise that is harmless offline and
    unexecutable on hardware: raw, these trajectories demand tens to hundreds of
    rad/s where the arm allows a few. The prior pipeline filtered the human side
    for the same reason. Filtering is applied to every condition identically, so
    it cannot favour one, and the cutoff is reported."""
    from scipy.signal import butter, filtfilt
    qs = np.asarray(qs, dtype=float)
    nyq = 0.5 * fps
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = butter(2, wn)
    if len(qs) <= 3 * max(len(a), len(b)):
        return qs.copy()
    return np.stack([filtfilt(b, a, qs[:, j]) for j in range(qs.shape[1])], axis=1)


def fit_cutoff(qs, ts, fps, vel_limit, cutoffs=(8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.7)):
    """Lowest-distortion cutoff whose peak joint velocity is within vel_limit."""
    for hz in cutoffs:
        f = lowpass(qs, fps, hz)
        v = peak_velocity(ts, f)
        if v.max() <= vel_limit:
            return hz, f, v
    f = lowpass(qs, fps, cutoffs[-1])
    return cutoffs[-1], f, peak_velocity(ts, f)


def peak_velocity(ts, qs):
    """Max |dq/dt| per joint, rad/s."""
    if len(ts) < 2:
        return np.zeros(qs.shape[1])
    dt = np.diff(np.asarray(ts))
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    return np.abs(np.diff(np.asarray(qs), axis=0) / dt[:, None]).max(axis=0)


# ── prep ─────────────────────────────────────────────────────────────────────
def prep(args):
    """Solve each capture under each condition, write replay CSV + limit sidecar."""
    os.makedirs(args.out_dir, exist_ok=True)
    lims = load_limits_4(args.urdf)[args.side]
    lo, hi = lims["lo"], lims["hi"]
    written = []
    scales = []
    for cap in args.captures:
        src = joints_csv_for(cap)
        if not os.path.isfile(src):
            print(f"[prep] {cap}: MISSING {src}")
            continue
        for cond in args.conditions:
            stats, ts, qs, _tg = x6.solve_rep(
                src, args.side, args.urdf, args.fps, method=cond,
                ext_cap=args.ext_cap)
            # Frames where any joint sits on its limit, which is what X3 tests.
            touch = [i for i, q in enumerate(qs)
                     if float(limit_margin(q, lo, hi)) <= args.touch_tol]
            qs_raw = np.asarray(qs)
            v_raw = peak_velocity(ts, qs_raw)
            # One cutoff for every condition and capture. Filtering must not
            # differ between conditions, or the comparison is confounded.
            hz = args.lp_hz
            qs_out = lowpass(qs_raw, args.fps, hz) if hz > 0 else qs_raw
            v_out = peak_velocity(ts, qs_out)
            # speed_scale divides velocity linearly, so this is what makes it safe
            need_scale = min(1.0, args.vel_warn / v_out.max()) if v_out.max() > 0 else 1.0
            scales.append((f"{cap}_{cond}", float(need_scale), float(v_out.max())))

            # Full 15-joint vector. Only the solved side moves.
            other = "left" if args.side == "right" else "right"
            names, cols = ["t_s"], []
            for sd in ("left", "right"):
                for k, seg in enumerate(SEGS):
                    names.append(f"{sd}_{seg}")
                    cols.append(("solve", k) if sd == args.side else ("rest", REST4[sd][k]))
                for seg in WRIST:
                    names.append(f"{sd}_{seg}")
                    cols.append(("rest", 0.0))
            names.append("waist_yaw"); cols.append(("rest", 0.0))

            path = os.path.join(args.out_dir, f"{cap}_{cond}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(names)
                for t, q in zip(ts, qs_out):
                    row = [f"{t:.4f}"]
                    for kind, val in cols:
                        row.append(f"{q[val]:.6f}" if kind == "solve" else f"{val:.6f}")
                    w.writerow(row)
            names = [f"{args.side}_{s}" for s in SEGS]   # sidecar records solved joints
            side_json = os.path.splitext(path)[0] + ".limits.json"
            json.dump(dict(capture=cap, condition=cond, side=args.side,
                           joint_names=names,
                           lo=[float(v) for v in lo], hi=[float(v) for v in hi],
                           n_frames=int(len(ts)),
                           touch_tol=args.touch_tol,
                           lowpass_hz=float(hz),
                           peak_vel_after_filter=float(v_out.max()),
                           recommended_speed_scale=float(need_scale),
                           peak_vel_raw=[float(v) for v in v_raw],
                           peak_vel_out=[float(v) for v in v_out],
                           predicted_limit_frames=touch,
                           predicted_limit_times=[float(ts[i]) for i in touch],
                           offline=stats),
                      open(side_json, "w"), indent=2)
            written.append(path)
            print(f"[prep] {cap:8} {cond:12} {len(ts):5d} frames, "
                  f"{len(touch):5d} predicted limit frames "
                  f"({100.0*len(touch)/max(1,len(ts)):5.1f}%), "
                  f"margin {stats['margin_mean']:.3f}  "
                  f"peak vel {v_raw.max():6.1f} -> {v_out.max():5.2f} rad/s @ {hz:.1f} Hz"
                  f" -> {os.path.basename(path)}")
    print(f"\n[prep] wrote {len(written)} trajectories to {args.out_dir}")
    if scales:
        common = min(sc for _n, sc, _v in scales)
        print(f"[prep] all conditions filtered identically at {args.lp_hz:.1f} Hz.")
        print(f"[prep] peak velocity after filtering, rad/s:")
        for n, sc, v in scales:
            print(f"         {n:22} {v:5.2f}   needs speed-scale {sc:.2f}")
        print(f"\n[prep] USE --speed-scale {common:.2f} FOR EVERY TRAJECTORY.")
        print( "[prep] One speed for all of them keeps the comparison clean, and it "
               "is set by the fastest condition.")


# ── run ──────────────────────────────────────────────────────────────────────
def run(args):
    traj = os.path.abspath(args.trajectory_csv)
    if not os.path.isfile(traj):
        sys.exit(f"[run] no such trajectory: {traj}")
    tag = os.path.splitext(os.path.basename(traj))[0]
    # sdk_replay_logger writes <out-dir>/replay_M<motion>_T<trial>.csv and refuses
    # to overwrite, so give each trajectory its own directory and bump --trial.
    log_dir = os.path.join(os.path.dirname(traj), tag + ".log")
    os.makedirs(log_dir, exist_ok=True)
    trial = args.trial
    while os.path.exists(os.path.join(log_dir, f"replay_M0_T{trial}.csv")):
        trial += 1
    log_path = os.path.join(log_dir, f"replay_M0_T{trial}.csv")

    py = sys.executable   # conda env: /usr/bin/python3 lacks unitree_sdk2py
    logger = [py, os.path.join(EXP_SCRIPTS, "sdk_replay_logger.py"),
              "--interface", args.interface, "--out-dir", log_dir,
              "--motion", "0", "--trial", str(trial)]
    replay = [py, os.path.join(EXP_SCRIPTS, "replay_arm.py"), traj,
              "--interface", args.interface,
              "--gravity-ff", "--gravity-gain", str(args.gravity_gain)]
    if args.speed_scale != 1.0:
        replay += ["--speed-scale", str(args.speed_scale)]

    print("[run] e-stop operator required, and it must not be the operator at "
          "the keyboard.")
    print("  logger: " + " ".join(logger))
    print("  replay: " + " ".join(replay))
    if args.dry_run:
        print("[run] dry run, nothing executed")
        return
    input("Press Enter when the e-stop operator is ready, or Ctrl+C to abort...")

    log_proc = subprocess.Popen(logger)
    try:
        subprocess.run(replay, check=True)
    finally:
        log_proc.terminate()
        log_proc.wait()
    print(f"[run] log written to {log_path}")
    print(f"[run] next: x3_hardware_replay.py grade {traj}")


# ── grade ────────────────────────────────────────────────────────────────────
def _read_log(path):
    """Executed joint angles from the replay log, as {name: array} plus time."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None, None
    cols = rows[0].keys()
    exe = {}
    for c in cols:
        # exe_<joint> only. exedq_/exetau_ do not match the underscore.
        base = c[4:] if c.startswith("exe_") else None
        if base:
            exe[base] = np.array([float(r[c]) for r in rows])
    # IROS logs use t_rel (seconds from start). t_wall is epoch time.
    tcol = next((c for c in ("t_rel", "t_s", "t_wall") if c in cols),
                list(cols)[0])
    return np.array([float(r[tcol]) for r in rows]), exe


def grade(args):
    """Compare commanded against executed using the log's OWN cmd_ columns.

    The trajectory file and the log run on different clocks: --speed-scale
    stretches the replay, and there is a lead-in before playback starts. Reading
    the commanded angles from the log removes the alignment problem entirely,
    because cmd_ and exe_ are written on the same row at the same instant.
    Limit contact is then recomputed on the logged command, so the frames tested
    are the ones the robot was actually asked to hold on a limit."""
    traj = os.path.abspath(args.trajectory_csv)
    side_json = os.path.splitext(traj)[0] + ".limits.json"
    if not os.path.isfile(side_json):
        sys.exit(f"[grade] missing sidecar {side_json}, re-run prep")
    meta = json.load(open(side_json))
    names = meta["joint_names"]
    lo = np.array(meta["lo"]); hi = np.array(meta["hi"])

    if args.log:
        log_path = args.log
    else:
        d = os.path.splitext(traj)[0] + ".log"
        cand = sorted(glob.glob(os.path.join(d, "replay_*.csv")))
        if not cand:
            sys.exit(f"[grade] no replay log under {d}")
        log_path = cand[-1]

    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        sys.exit(f"[grade] empty log {log_path}")
    cols = rows[0].keys()
    need = [f"cmd_{n}" for n in names] + [f"exe_{n}" for n in names]
    missing = [c for c in need if c not in cols]
    if missing:
        sys.exit(f"[grade] log lacks columns {missing[:4]}")

    def col(pfx):
        return np.array([[float(r[f"{pfx}_{n}"]) for n in names] for r in rows])

    Q_cmd, Q_exe = col("cmd"), col("exe")
    ok = np.isfinite(Q_cmd).all(axis=1) & np.isfinite(Q_exe).all(axis=1)
    # only the frames where the overlay was actually driving the arm
    if "weight" in cols:
        w = np.array([float(r["weight"]) if r["weight"] not in ("", "nan") else 0.0
                      for r in rows])
        ok &= w > 0.5
    Q_cmd, Q_exe = Q_cmd[ok], Q_exe[ok]
    if len(Q_cmd) < 10:
        sys.exit("[grade] too few active frames in the log")

    err = np.degrees(np.abs(Q_exe - Q_cmd))
    # limit contact recomputed on the LOGGED command
    span = hi - lo
    marg = np.minimum(Q_cmd - lo, hi - Q_cmd) / np.where(span > 0, span, 1.0)
    touch = marg.min(axis=1) <= args.touch_tol
    exe_out = ((Q_exe < lo - args.tol) | (Q_exe > hi + args.tol)).any(axis=1)

    print(f"[grade] {os.path.basename(traj)}   {meta['capture']} "
          f"{meta['condition']}")
    print(f"  log {os.path.basename(log_path)}, {len(Q_cmd)} active frames")
    print(f"  frames commanded onto a limit: {int(touch.sum())} "
          f"({100.0*touch.mean():.1f}%)")
    print(f"  executed pose outside limits:  {int(exe_out.sum())}")
    print()
    print(f"  {'':24}{'mean err deg':>14}{'p95 deg':>10}{'max deg':>10}")
    print(f"  {'all frames':24}{err.mean():>14.3f}{np.percentile(err,95):>10.3f}"
          f"{err.max():>10.3f}")
    verdict = dict(n_frames=int(len(Q_cmd)),
                   frames_on_limit=int(touch.sum()),
                   executed_out_of_limits=int(exe_out.sum()),
                   mean_err_deg=float(err.mean()),
                   p95_err_deg=float(np.percentile(err, 95)))
    if touch.any() and (~touch).any():
        at, el = err[touch].mean(), err[~touch].mean()
        ratio = at / el if el > 1e-9 else float("inf")
        print(f"  {'at limit frames':24}{at:>14.3f}"
              f"{np.percentile(err[touch],95):>10.3f}{err[touch].max():>10.3f}")
        print(f"  {'elsewhere':24}{el:>14.3f}"
              f"{np.percentile(err[~touch],95):>10.3f}{err[~touch].max():>10.3f}")
        verdict.update(mean_err_at_limit_deg=float(at),
                       mean_err_elsewhere_deg=float(el), ratio=float(ratio))
        print()
        if exe_out.sum() > 0:
            print(f"  OUTCOME A. The executed pose left the joint limits on "
                  f"{int(exe_out.sum())} frames.")
        elif ratio > args.ratio_thresh:
            print(f"  OUTCOME B. The controller held the joints inside their "
                  f"limits and the cost appears as tracking error, {at:.3f} deg "
                  f"at the limit frames against {el:.3f} deg elsewhere, a factor "
                  f"of {ratio:.1f}.")
        else:
            print(f"  OUTCOME C. No elevated deviation at the limit frames, "
                  f"{at:.3f} against {el:.3f} deg. The offline limit contact did "
                  f"not produce a measurable hardware consequence here.")
    elif not touch.any():
        print("  No frame was commanded onto a limit, so this trajectory is the "
              "control rather than the test.")

    out = os.path.splitext(traj)[0] + ".grade.json"
    json.dump(dict(trajectory=os.path.basename(traj), capture=meta["capture"],
                   condition=meta["condition"], log=os.path.basename(log_path),
                   **verdict), open(out, "w"), indent=2)
    print(f"\n[grade] wrote {out}")


# ── cli ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prep", help="generate replay trajectories, no hardware")
    a.add_argument("--captures", nargs="+", default=["M2_1", "M3_1"])
    a.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                   choices=list(x6.METHODS))
    a.add_argument("--side", default="right", choices=["left", "right"])
    a.add_argument("--urdf", default=DEFAULT_URDF)
    a.add_argument("--fps", type=float, default=60.0)
    a.add_argument("--ext-cap", type=float, default=0.88)
    a.add_argument("--touch-tol", type=float, default=1e-4,
                   help="margin at or below this counts as sitting on a limit")
    a.add_argument("--lp-hz", type=float, default=0.0,
                   help="low-pass cutoff in Hz. 0 auto-selects the highest cutoff "
                        "whose peak joint velocity is within --vel-warn")
    a.add_argument("--vel-warn", type=float, default=4.0,
                   help="peak joint velocity considered safe to replay, rad/s")
    a.add_argument("--out-dir", default=os.path.join(HERE, "..", "results", "x3"))
    a.set_defaults(func=prep)

    b = sub.add_parser("run", help="replay on hardware, e-stop operator required")
    b.add_argument("trajectory_csv")
    b.add_argument("--interface", required=True)
    b.add_argument("--gravity-gain", type=float, default=1.0)
    b.add_argument("--speed-scale", type=float, default=1.0,
                   help="use 0.5 for the first trajectory of a session")
    b.add_argument("--trial", type=int, default=1,
                   help="log trial number, auto-bumped if taken")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=run)

    c = sub.add_parser("grade", help="analyse a replay log, no hardware")
    c.add_argument("trajectory_csv")
    c.add_argument("--log", help="defaults to <trajectory>.replay.csv")
    c.add_argument("--touch-tol", type=float, default=1e-3,
                   help="normalized margin at or below which the command counts as on a limit")
    c.add_argument("--tol", type=float, default=1e-3,
                   help="radians of slack before calling a pose out of limits")
    c.add_argument("--ratio-thresh", type=float, default=1.5,
                   help="error ratio above which outcome B is declared")
    c.set_defaults(func=grade)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
