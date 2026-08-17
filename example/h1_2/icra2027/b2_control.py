#!/usr/bin/env python3
"""b2_control.py - baseline B2, the "is learning needed / is C2 a method" control.

B2 is textbook task-space inverse kinematics: damped least squares on the 3D
end-effector POSITION target, with joint-limit avoidance resolved in the null
space (Liegeois 1977). It uses NO posture-fidelity term, NO R_robot health
objective, NO yaw hack, and NO learning. It is warm-started from the previous
frame (causal, deployable online).

The question it settles: does a generic, off-the-shelf task-space IK already
remove the branch flips? If yes, our task-centric formulation (C2) is an
instantiation of a known idea, not a novel method, and the paper is C1 + C3 +
C4. The one thing that could keep C2 alive: B2 targets position ONLY, so it may
sacrifice task-relevant ORIENTATION at contact events (a strike/place cares
about approach direction). We therefore also report the approach-direction
error at event frames; if B2 ignores orientation while our formulation keeps it,
that is C2's surviving contribution.

Comparison, right arm, M2/M3: fidelity (direction-matching, yaw hack OFF, the
flip baseline) vs B2. Metrics: yaw flips, EE position error, and approach-
direction error at contact events (low EE speed).

Usage:
    /usr/bin/python3 b2_control.py --motions M2 M3 --reps 1 2 3
"""

import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments", "scripts")
sys.path.insert(0, HERE)
sys.path.insert(0, EXP)
from retarget_arm import ArmChainFK, load_limits, _unit          # noqa: E402
from x6_taskcentric import retarget, read_traj, RSEG, DATA, URDF  # noqa: E402

FLIP_RADS = 6.0
EVENT_SPEED = 0.04


def pos_jac(fk, q):
    wr = fk.positions(q)[2]
    J = np.zeros((3, 4))
    for k in range(4):
        qp = q.copy(); qp[k] += 1e-5
        J[:, k] = (fk.positions(qp)[2] - wr) / 1e-5
    return J, wr


def b2_solve(fk, x_t, limits, lam=0.05, knull=0.4, iters=10):
    """Causal DLS position IK + null-space joint-limit avoidance."""
    lo = np.array([l for l, _ in limits]); hi = np.array([h for _, h in limits])
    mid = 0.5 * (lo + hi); span = (hi - lo)
    I4 = np.eye(4)
    Q = np.zeros((len(x_t), 4))
    q = mid.copy()
    for i in range(len(x_t)):
        for _ in range(iters):
            J, wr = pos_jac(fk, q)
            e = x_t[i] - wr
            Jp = J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(3))
            z = -knull * (q - mid) / (0.5 * span)      # push toward mid-range
            dq = Jp @ e + (I4 - Jp @ J) @ z
            q = np.clip(q + dq, lo, hi)
        Q[i] = q
    return Q


def yaw_flips(t, Q):
    yaw = Q[:, 2]; dt = np.diff(t); dt[dt <= 0] = np.median(dt[dt > 0])
    return int((np.abs(np.diff(yaw)) / dt > FLIP_RADS).sum())


def run_rep(motion, rep):
    tag = f"{motion}_{rep}"
    joints = os.path.join(DATA, tag, "b2g", "iph_mono.csv")
    if not os.path.exists(joints):
        return None
    fk = ArmChainFK(URDF, "right")
    lim = [load_limits(URDF)[f"right_{s}_joint"] for s in RSEG]
    with tempfile.TemporaryDirectory() as td:
        clean = os.path.join(td, "clean.csv")
        nohack = os.path.join(td, "nohack.csv")
        retarget(joints, clean, 0.02, 0.05)         # smooth EE target
        retarget(joints, nohack, 0.0, 0.0)          # fidelity flip baseline
        t, qc = read_traj(clean)
        _, qn = read_traj(nohack)
    pos = np.array([fk.positions(q)[2] for q in qc])
    adir = np.array([_unit(fk.positions(q)[2] - fk.positions(q)[1]) for q in qc])
    dt = np.diff(t); dt[dt <= 0] = np.median(dt[dt > 0])
    speed = np.concatenate([[0], np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt])
    event = speed < EVENT_SPEED
    Qb = b2_solve(fk, pos, lim)
    # metrics
    Pb = np.array([fk.positions(q)[2] for q in Qb])
    pos_err = float(np.linalg.norm(Pb - pos, axis=1).mean() * 100)
    bdir = np.array([_unit(fk.positions(q)[2] - fk.positions(q)[1]) for q in Qb])
    ev = event & np.isfinite(bdir).all(1) & np.isfinite(adir).all(1)
    dir_err = float(np.degrees(np.arccos(np.clip(
        np.sum(bdir[ev] * adir[ev], axis=1), -1, 1))).mean()) if ev.any() else float("nan")
    return dict(tag=tag, fid_flips=yaw_flips(t, qn), b2_flips=yaw_flips(t, Qb),
                b2_pos=pos_err, b2_dir=dir_err, ev=float(event.mean() * 100))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motions", nargs="+", default=["M2", "M3"])
    p.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    args = p.parse_args()
    print("B2 control: generic task-space DLS IK + null-space joint-limit "
          "avoidance (no fidelity, no R_robot, no learning). Right arm.\n")
    hd = (f"{'rep':7}{'fidelity flips':>16}{'B2 flips':>10}{'B2 pos err cm':>15}"
          f"{'B2 event-dir err deg':>22}")
    print(hd); print("-" * len(hd))
    for m in args.motions:
        for r in args.reps:
            res = run_rep(m, r)
            if res is None:
                print(f"{m}_{r}: (missing)"); continue
            print(f"{res['tag']:7}{res['fid_flips']:>16}{res['b2_flips']:>10}"
                  f"{res['b2_pos']:>15.2f}{res['b2_dir']:>22.1f}")
    print("\nRead: B2 flips ~ 0 => generic IK already fixes the flips (the "
          "objective, not our specific R_robot, is what matters; C2 is an "
          "instantiation). B2 event-dir error LARGE => generic position IK "
          "sacrifices task-relevant orientation that our formulation keeps, "
          "which is C2's surviving contribution. Compare X6 task-centric: "
          "0 flips AND orientation respected at events.")


if __name__ == "__main__":
    main()
