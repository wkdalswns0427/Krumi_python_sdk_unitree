#!/usr/bin/env python3
"""x6_taskcentric.py - ICRA Phase-0 critical ablation (Experiment X6).

QUESTION: does resolving the arm redundancy with a task-centric objective plus
a competent (multi-start) solver eliminate the shoulder-yaw branch flips that
the fidelity objective produces, WITHOUT any learning?

Everything downstream (whether learning is a contribution or a footnote) hangs
on the answer. Outcomes (see icra2027/PLAN.md Phase 0):
  1. flips persist  -> learning justified (distill a global solution).
  2. flips vanish   -> learning drops to latency + robustness.
  3. flips vanish + cheap online -> maybe no learning at all.

METHOD (task held identical, only redundancy resolution differs), right arm:
  clean B1  = retarget_arm with the yaw hack ON  -> the smooth EE task target
              x_t (wrist position) and a_t (approach/forearm direction).
  B1 nohack = retarget_arm with the yaw hack OFF -> the FIDELITY baseline; its
              greedy direction-matching flips. This is what X6 must beat.
  X6        = per frame  argmin_q  w_pos||wrist(q)-x_t||^2
                                  + w_dir||fdir(q)-a_t||^2
                                  + R_robot(q, q_prev),
              R_robot = joint-limit margin + (1/manipulability) + continuity,
              solved MULTI-START (seeds: prev frame, mid-range, N random),
              lowest-cost kept. No yaw-neutral trick; no learning.

METRICS (right_shoulder_yaw): flip count (|dyaw|/dt > 6 rad/s), peak yaw vel,
percent of frames within 0.1 rad of a limit. X6 also reports task-tracking
error (it must still hit the EE path) so "no flips" cannot be won by ignoring
the task.

Gravity belongs in R_robot for the X2 cost-of-fidelity study; the flip question
is resolved by limit-margin + continuity + manipulability, so v1 uses those.

Usage:
    /usr/bin/python3 x6_taskcentric.py                  # M2 + M3, rep 1
    /usr/bin/python3 x6_taskcentric.py --reps 1 2 3
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments", "scripts")
sys.path.insert(0, EXP)
from retarget_arm import ArmChainFK, load_limits, _unit          # noqa: E402

DATA = os.path.join(EXP, "..", "iphone_data", "MotionsDataset0713")
URDF = os.path.expanduser("~/mj_ws/assets/h1_2_description/h1_2.urdf")
RSEG = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
RCOLS = [f"right_{s}" for s in RSEG]
FLIP_RADS = 6.0                       # single-frame yaw jump threshold
LIM_MARGIN = 0.10                     # "near a limit" band (rad)


def retarget(joints_csv, out_csv, yaw_neutral, yaw_reg):
    subprocess.run(
        ["/usr/bin/python3", os.path.join(EXP, "retarget_arm.py"), joints_csv,
         "--out", out_csv, "--yaw-neutral", str(yaw_neutral),
         "--yaw-reg", str(yaw_reg)],
        check=True, capture_output=True)


def read_traj(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t_s"]) for r in rows])
    q = np.array([[float(r[c]) for c in RCOLS] for r in rows])
    return t, q


def yaw_metrics(t, q, lo, hi):
    yaw = q[:, 2]
    dt = np.diff(t)
    dt[dt <= 0] = np.median(dt[dt > 0])
    v = np.abs(np.diff(yaw)) / dt
    near = np.mean((yaw < lo + LIM_MARGIN) | (yaw > hi - LIM_MARGIN)) * 100
    return dict(flips=int((v > FLIP_RADS).sum()), peak_vel=float(v.max()),
                near_limit=float(near))


def x6_solve(fk, x_t, a_t, limits, weights, n_random=1, seed=0, greedy=False):
    """Task-centric multi-start solve over the whole trajectory.

    Speed: manipulability is NOT in the inner objective (its nested finite
    difference dominated cost). The inner objective (1 FK/eval) is task pos +
    approach dir + limit margin + continuity; manipulability enters only when
    RANKING the multi-start candidates, so a low-manipulability (near-singular
    or branch-flipped) solution loses to a healthier one. That still lets the
    robot-native term break the branch tie, at a fraction of the cost."""
    lo = np.array([l for l, _ in limits])
    hi = np.array([h for _, h in limits])
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    wp, wd, wl, wm, wc = (weights[k] for k in
                          ("pos", "dir", "lim", "manip", "cont"))
    rng = np.random.default_rng(seed)

    def manip(q):                         # Yoshikawa index of the 3x4 pos Jac
        wr = fk.positions(q)[2]
        J = np.zeros((3, 4))
        for k in range(4):
            qp = q.copy(); qp[k] += 1e-5
            J[:, k] = (fk.positions(qp)[2] - wr) / 1e-5
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def cost(q, xt, at, qprev):           # cheap: 1 FK per eval
        a, e, w = fk.positions(q)
        fdir = _unit(w - e)
        L = wp * np.sum((w - xt) ** 2) + wl * np.sum(((q - mid) / half) ** 4) \
            + wc * np.sum((q - qprev) ** 2)
        if fdir is not None:
            L += wd * np.sum((fdir - at) ** 2)
        return L

    bounds = list(zip(lo, hi))
    Q = np.zeros((len(x_t), 4))
    qprev = mid.copy()
    for i in range(len(x_t)):
        # greedy = causal warm-start only (the cheap online solver, no global
        # fallback). multi = warm-start + mid anchor + random restarts.
        seeds = [qprev] if greedy else (
            [qprev, mid] + [lo + rng.random(4) * (hi - lo)
                            for _ in range(n_random)])
        best, bestrank = None, np.inf
        for s in seeds:
            r = minimize(cost, np.clip(s, lo, hi), args=(x_t[i], a_t[i], qprev),
                         method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=40))
            rank = r.fun + wm / (manip(r.x) + 1e-4)     # manip only at ranking
            if rank < bestrank:
                bestrank, best = rank, r.x
        Q[i] = best
        qprev = best
    return Q


def task_error_cm(fk, Q, x_t):
    P = np.array([fk.positions(q)[2] for q in Q])
    return float(np.linalg.norm(P - x_t, axis=1).mean() * 100)


def run_rep(motion, rep, weights):
    tag = f"{motion}_{rep}"
    joints = os.path.join(DATA, tag, "b2g", "iph_mono.csv")
    if not os.path.exists(joints):
        return None
    fk = ArmChainFK(URDF, "right")
    lim = [load_limits(URDF)[f"right_{s}_joint"] for s in RSEG]
    lo, hi = lim[2]                                    # yaw limits for metrics
    with tempfile.TemporaryDirectory() as td:
        clean = os.path.join(td, "clean.csv")
        nohack = os.path.join(td, "nohack.csv")
        retarget(joints, clean, 0.02, 0.05)           # smooth task target
        retarget(joints, nohack, 0.0, 0.0)            # fidelity baseline
        t, qc = read_traj(clean)
        _, qn = read_traj(nohack)
        # task targets from the clean B1: wrist pos + forearm (approach) dir
        pos = np.array([fk.positions(q)[2] for q in qc])
        adir = np.array([_unit(fk.positions(q)[2] - fk.positions(q)[1])
                         for q in qc])
        Qg = x6_solve(fk, pos, adir, lim, weights, greedy=True)   # causal
        Qm = x6_solve(fk, pos, adir, lim, weights, greedy=False)  # +global
    return dict(tag=tag, n=len(t),
                fid=yaw_metrics(t, qn, lo, hi),
                greedy=yaw_metrics(t, Qg, lo, hi), greedy_err=task_error_cm(fk, Qg, pos),
                multi=yaw_metrics(t, Qm, lo, hi), multi_err=task_error_cm(fk, Qm, pos))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motions", nargs="+", default=["M2", "M3"])
    p.add_argument("--reps", nargs="+", type=int, default=[1])
    p.add_argument("--w-pos", type=float, default=50.0)
    p.add_argument("--w-dir", type=float, default=1.0)
    p.add_argument("--w-lim", type=float, default=0.05)
    p.add_argument("--w-manip", type=float, default=2e-4)
    p.add_argument("--w-cont", type=float, default=0.2)
    args = p.parse_args()
    weights = dict(pos=args.w_pos, dir=args.w_dir, lim=args.w_lim,
                   manip=args.w_manip, cont=args.w_cont)

    print("X6 critical ablation, right shoulder-yaw. flip = |dyaw|/dt > "
          f"{FLIP_RADS} rad/s")
    print("  fidelity  = B1 direction-matching, greedy, yaw hack OFF")
    print("  TC-greedy = task-centric objective, causal warm-start ONLY "
          "(cheap online)")
    print("  TC-multi  = task-centric objective, multi-start global solver\n")
    hdr = (f"{'rep':8}{'fid flips':>10}{'':3}{'TC-greedy':>12}{'err':>7}"
           f"{'':3}{'TC-multi':>11}{'err':>7}")
    print(hdr)
    print(f"{'':21}{'flips  pk':>15}{'cm':>7}{'':3}{'flips  pk':>14}{'cm':>7}")
    print("-" * len(hdr))
    for m in args.motions:
        for r in args.reps:
            res = run_rep(m, r, weights)
            if res is None:
                print(f"{m}_{r}: (missing)"); continue
            f, g, x = res["fid"], res["greedy"], res["multi"]
            print(f"{res['tag']:8}{f['flips']:>7}{f['peak_vel']:>8.1f}   "
                  f"{g['flips']:>6}{g['peak_vel']:>6.1f}{res['greedy_err']:>7.2f}   "
                  f"{x['flips']:>6}{x['peak_vel']:>6.1f}{res['multi_err']:>7.2f}")


if __name__ == "__main__":
    main()
