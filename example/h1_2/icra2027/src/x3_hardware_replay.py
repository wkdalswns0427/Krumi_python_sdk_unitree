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
from common.self_collision import (                                 # noqa: E402
    scan as sc_scan, verdict as sc_verdict, HAND_LEN_M, SAFE_MARGIN_M,
    torso_clearance, repair_torso)
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
    """Solve each capture under each condition, write replay CSV + limit sidecar.

    --side both solves the left and the right arm and drives them together. The
    motions are bimanual, and the supporting arm covers 54 to 77 percent of the
    reaching arm's wrist travel, so a single-arm replay leaves half the motion
    on the table. Solving both also lets one replay test the side asymmetry the
    offline analysis reports."""
    os.makedirs(args.out_dir, exist_ok=True)
    sides = ("right", "left") if args.side == "both" else (args.side,)
    x6f = {sd: ArmChainFK(args.urdf, sd) for sd in ("right", "left")}
    L = load_limits_4(args.urdf)
    written = []
    scales = []
    unsafe = []
    for cap in args.captures:
        src = joints_csv_for(cap)
        if not os.path.isfile(src):
            print(f"[prep] {cap}: MISSING {src}")
            continue
        for cond in args.conditions:
            per = {}
            for sd in sides:
                stats, ts, qs, _tg = x6.solve_rep(
                    src, sd, args.urdf, args.fps, method=cond,
                    ext_cap=args.ext_cap)
                lo, hi = L[sd]["lo"], L[sd]["hi"]
                # Lift the arm off the trunk before anything else looks at it.
                # The demonstrator's posture is faithful but the robot's torso
                # is proportionally broader, so an adducted arm can end up
                # inside the chest. The same procedure runs on every condition,
                # and what it applied is recorded in the sidecar.
                tors_before = min(torso_clearance(x6f[sd], q) for q in qs)
                rep_deg = 0.0
                rep_frames = 0
                if args.repair_torso:
                    qs, need = repair_torso(x6f[sd], qs, sd,
                                            margin=args.torso_margin)
                    rep_deg = float(np.degrees(need.max()))
                    rep_frames = int((need > 0).sum())
                tors_after = min(torso_clearance(x6f[sd], q) for q in qs)
                # Frames where any joint sits on its limit, which is what X3 tests.
                touch = [i for i, q in enumerate(qs)
                         if float(limit_margin(q, lo, hi)) <= args.touch_tol]
                qs_raw = np.asarray(qs)
                v_raw = peak_velocity(ts, qs_raw)
                # One cutoff for every condition, capture and side. Filtering
                # must not differ between them, or the comparison is confounded.
                hz = args.lp_hz
                qs_out = lowpass(qs_raw, args.fps, hz) if hz > 0 else qs_raw
                # The solver respects the limits but the low-pass does not: it
                # overshoots at turning points and can push a joint a fraction
                # of a degree past its stop. That is enough to make the robot
                # look as though it left its limits when it was simply obeying
                # the command, so clamp before anything downstream sees it.
                n_over = int(((qs_out < lo) | (qs_out > hi)).any(axis=1).sum())
                qs_out = np.clip(qs_out, lo, hi)
                v_out = peak_velocity(ts, qs_out)
                per[sd] = dict(stats=stats, ts=ts, qs=qs_out, touch=touch,
                               lo=lo, hi=hi, v_raw=v_raw, v_out=v_out, hz=hz,
                               clamped_frames=n_over,
                               torso_before=tors_before, torso_after=tors_after,
                               repair_deg=rep_deg, repair_frames=rep_frames)
            # Both sides come from one capture at one fps, so the frame counts
            # match, but truncate to the shorter if conditioning dropped frames.
            n = min(len(per[sd]["ts"]) for sd in sides)
            ts = per[sides[0]]["ts"][:n]
            vmax = max(float(per[sd]["v_out"].max()) for sd in sides)
            # speed_scale divides velocity linearly, so this is what makes it safe
            need_scale = min(1.0, args.vel_warn / vmax) if vmax > 0 else 1.0
            scales.append((f"{cap}_{cond}", float(need_scale), vmax))

            # Full 15-joint vector. Solved sides move, the rest hold at rest.
            names, cols = ["t_s"], []
            for sd in ("left", "right"):
                for k, seg in enumerate(SEGS):
                    names.append(f"{sd}_{seg}")
                    cols.append((sd, k) if sd in per else ("rest", REST4[sd][k]))
                for seg in WRIST:
                    names.append(f"{sd}_{seg}")
                    cols.append(("rest", 0.0))
            names.append("waist_yaw"); cols.append(("rest", 0.0))

            path = os.path.join(args.out_dir, f"{cap}_{cond}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(names)
                for i in range(n):
                    row = [f"{ts[i]:.4f}"]
                    for kind, val in cols:
                        row.append(f"{val:.6f}" if kind == "rest"
                                   else f"{per[kind]['qs'][i][val]:.6f}")
                    w.writerow(row)
            # Bar-aware self-collision. The wrist frame is not the end of the
            # arm: a 0.20 m bar hand extends along the forearm, and the lifting
            # motions bring both forearms toward the midline. Wrist-frame
            # clearance therefore understates the real gap badly.
            sc = None
            if len(sides) == 2:
                fkR = ArmChainFK(args.urdf, "right")
                fkL = ArmChainFK(args.urdf, "left")
                sc = sc_scan(fkR, fkL, per["right"]["qs"][:n], per["left"]["qs"][:n],
                             hand_len=args.hand_len, margin=args.sc_margin)
                v = sc_verdict(sc, args.sc_margin)
                if v != "ok":
                    unsafe.append((os.path.basename(path), sc, v))
                sc = {k: val for k, val in sc.items() if k != "per_frame"}

            side_json = os.path.splitext(path)[0] + ".limits.json"
            json.dump(dict(capture=cap, condition=cond,
                           side=args.side, sides=list(sides),
                           joint_names={sd: [f"{sd}_{s}" for s in SEGS] for sd in sides},
                           lo={sd: [float(v) for v in per[sd]["lo"]] for sd in sides},
                           hi={sd: [float(v) for v in per[sd]["hi"]] for sd in sides},
                           n_frames=int(n),
                           touch_tol=args.touch_tol,
                           lowpass_hz=float(args.lp_hz),
                           peak_vel_after_filter=vmax,
                           recommended_speed_scale=float(need_scale),
                           peak_vel_raw={sd: [float(v) for v in per[sd]["v_raw"]] for sd in sides},
                           peak_vel_out={sd: [float(v) for v in per[sd]["v_out"]] for sd in sides},
                           predicted_limit_frames={sd: per[sd]["touch"] for sd in sides},
                           predicted_limit_times={sd: [float(per[sd]["ts"][i])
                                                       for i in per[sd]["touch"]] for sd in sides},
                           offline={sd: per[sd]["stats"] for sd in sides},
                           self_collision=sc,
                           torso={sd: dict(before_m=per[sd]["torso_before"],
                                           after_m=per[sd]["torso_after"],
                                           repair_deg=per[sd]["repair_deg"],
                                           repair_frames=per[sd]["repair_frames"])
                                  for sd in sides}),
                      open(side_json, "w"), indent=2)
            written.append(path)
            stats = per[sides[0]]["stats"]
            ts = per[sides[0]]["ts"]
            touch = per[sides[0]]["touch"]
            v_raw = per[sides[0]]["v_raw"]
            v_out = per[sides[0]]["v_out"]
            hz = args.lp_hz
            bits = "  ".join(
                f"{sd[0].upper()}: {len(per[sd]['touch']):4d} lim, "
                f"marg {per[sd]['stats']['margin_mean']:.3f}, "
                f"vel {per[sd]['v_out'].max():5.2f}, "
                f"torso {per[sd]['torso_after']*100:+.1f}cm" for sd in sides)
            print(f"[prep] {cap:8} {cond:12} {n:5d} fr   {bits}"
                  f"  -> {os.path.basename(path)}")
    if unsafe:
        print("\n" + "="*72)
        print("SELF-COLLISION: these trajectories must not be replayed as they are")
        print("="*72)
        for name, sc, v in unsafe:
            print(f"  {name:28} {v}")
            print(f"      closest {sc['min_m']*100:5.1f} cm at frame {sc['argmin_frame']}"
                  f" between {sc['pair']},  {sc['frames_under_margin']} frames inside the margin")
        print("  The bar hand extends 0.20 m past the wrist along the forearm, and")
        print("  the segment model gives it no width, so the real gap is smaller.")
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
    are the ones the robot was actually asked to hold on a limit.

    Grades every side the prep solved and every trial in the log directory, so
    a two-speed session is summarised in one pass."""
    traj = os.path.abspath(args.trajectory_csv)
    side_json = os.path.splitext(traj)[0] + ".limits.json"
    if not os.path.isfile(side_json):
        sys.exit(f"[grade] missing sidecar {side_json}, re-run prep")
    meta = json.load(open(side_json))

    # The sidecar carries per-side dicts. Older single-side files used flat
    # lists, so accept both rather than forcing a re-prep.
    jn = meta["joint_names"]
    if isinstance(jn, dict):
        sides = list(jn)
        NAMES = jn
        LO = {k: np.array(v) for k, v in meta["lo"].items()}
        HI = {k: np.array(v) for k, v in meta["hi"].items()}
    else:
        sd = meta.get("side", "right")
        sides, NAMES = [sd], {sd: jn}
        LO = {sd: np.array(meta["lo"])}
        HI = {sd: np.array(meta["hi"])}

    if args.log:
        logs = [args.log]
    else:
        d = os.path.splitext(traj)[0] + ".log"
        logs = sorted(glob.glob(os.path.join(d, "replay_*.csv")))
        if not logs:
            sys.exit(f"[grade] no replay log under {d}")

    all_out = []
    for log_path in logs:
        rows = list(csv.DictReader(open(log_path)))
        if not rows:
            print(f"[grade] empty log {log_path}"); continue
        cols = rows[0].keys()
        print(f"\n[grade] {os.path.basename(traj)}   {meta['capture']} "
              f"{meta['condition']}   log {os.path.basename(log_path)}")
        for sd in sides:
            names = NAMES[sd]
            need = [f"{p}_{n}" for p in ("cmd", "exe") for n in names]
            missing = [c for c in need if c not in cols]
            if missing:
                print(f"  {sd}: log lacks {missing[:3]}"); continue
            def col(pfx):
                return np.array([[float(r[f"{pfx}_{n}"]) for n in names]
                                 for r in rows])
            Q_cmd, Q_exe = col("cmd"), col("exe")
            ok = np.isfinite(Q_cmd).all(axis=1) & np.isfinite(Q_exe).all(axis=1)
            if "weight" in cols:
                w = np.array([float(r["weight"]) if r["weight"] not in ("", "nan")
                              else 0.0 for r in rows])
                ok &= w > 0.5
            Q_cmd, Q_exe = Q_cmd[ok], Q_exe[ok]
            if len(Q_cmd) < 10:
                print(f"  {sd}: too few active frames"); continue

            err = np.degrees(np.abs(Q_exe - Q_cmd))
            lo, hi = LO[sd], HI[sd]
            span = hi - lo
            marg = np.minimum(Q_cmd - lo, hi - Q_cmd) / np.where(span > 0, span, 1.0)
            touch = marg.min(axis=1) <= args.touch_tol
            # Only count an excursion the robot made on its own. If the
            # command was already outside, following it is obedience, not
            # failure.
            cmd_out = ((Q_cmd < lo - args.tol) | (Q_cmd > hi + args.tol))
            exe_out = (((Q_exe < lo - args.tol) | (Q_exe > hi + args.tol))
                       & ~cmd_out).any(axis=1)
            cmd_out_n = int(cmd_out.any(axis=1).sum())

            rec = dict(side=sd, log=os.path.basename(log_path),
                       n_frames=int(len(Q_cmd)),
                       frames_on_limit=int(touch.sum()),
                       executed_out_of_limits=int(exe_out.sum()),
                       commanded_out_of_limits=cmd_out_n,
                       mean_err_deg=float(err.mean()),
                       p95_err_deg=float(np.percentile(err, 95)),
                       max_err_deg=float(err.max()))
            print(f"  {sd}: {len(Q_cmd)} active frames, "
                  f"{int(touch.sum())} commanded onto a limit "
                  f"({100.0*touch.mean():.1f}%), {int(exe_out.sum())} executed outside "
                  f"unprompted" + (f", {cmd_out_n} commanded outside" if cmd_out_n else ""))
            print(f"       {'all frames':20}{err.mean():>10.3f}{np.percentile(err,95):>10.3f}"
                  f"{err.max():>10.3f}   (mean / p95 / max deg)")
            if touch.any() and (~touch).any():
                at, el = err[touch].mean(), err[~touch].mean()
                ratio = at / el if el > 1e-9 else float("inf")
                print(f"       {'at limit frames':20}{at:>10.3f}"
                      f"{np.percentile(err[touch],95):>10.3f}{err[touch].max():>10.3f}")
                print(f"       {'elsewhere':20}{el:>10.3f}"
                      f"{np.percentile(err[~touch],95):>10.3f}{err[~touch].max():>10.3f}")
                rec.update(mean_err_at_limit_deg=float(at),
                           mean_err_elsewhere_deg=float(el), ratio=float(ratio))
                if exe_out.sum() > 0:
                    rec["outcome"] = "A"
                    print(f"       OUTCOME A. Executed pose left the limits on "
                          f"{int(exe_out.sum())} frames.")
                elif ratio > args.ratio_thresh:
                    rec["outcome"] = "B"
                    print(f"       OUTCOME B. Controller clamped. Error {at:.3f} deg "
                          f"at limit frames against {el:.3f} elsewhere, "
                          f"factor {ratio:.1f}.")
                else:
                    rec["outcome"] = "C"
                    print(f"       OUTCOME C. No elevated deviation, {at:.3f} "
                          f"against {el:.3f} deg.")
            elif not touch.any():
                rec["outcome"] = "control"
                print("       No frame commanded onto a limit. This is the control.")
            all_out.append(rec)

    out = os.path.splitext(traj)[0] + ".grade.json"
    json.dump(dict(trajectory=os.path.basename(traj), capture=meta["capture"],
                   condition=meta["condition"], runs=all_out),
              open(out, "w"), indent=2)
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
    a.add_argument("--side", default="both", choices=["left", "right", "both"],
                   help="both solves and drives the two arms together, which is "
                        "what the bimanual motions need")
    a.add_argument("--urdf", default=DEFAULT_URDF)
    a.add_argument("--fps", type=float, default=60.0)
    a.add_argument("--ext-cap", type=float, default=0.88)
    a.add_argument("--touch-tol", type=float, default=1e-4,
                   help="margin at or below this counts as sitting on a limit")
    a.add_argument("--lp-hz", type=float, default=0.0,
                   help="low-pass cutoff in Hz. 0 auto-selects the highest cutoff "
                        "whose peak joint velocity is within --vel-warn")
    a.add_argument("--repair-torso", action="store_true",
                   help="abduct the shoulder just enough to lift the arm off the "
                        "trunk, applied identically to every condition")
    a.add_argument("--torso-margin", type=float, default=0.02,
                   help="clearance to leave against the trunk, metres")
    a.add_argument("--hand-len", type=float, default=HAND_LEN_M,
                   help="bar hand length beyond the wrist, metres")
    a.add_argument("--sc-margin", type=float, default=SAFE_MARGIN_M,
                   help="minimum arm-to-arm clearance to accept, metres")
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
