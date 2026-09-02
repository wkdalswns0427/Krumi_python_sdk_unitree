#!/usr/bin/env python3
"""Is the task still accomplished when the reach magnitude is re-set?

The task-defined target deliberately does not put the hand where the
demonstrator's hand was, so the obvious objection is that limit contact was
avoided by not performing the task. Measuring hand-position error against the
demonstrator would not answer it, because that measures fidelity, which is the
thing being argued against.

This asks the deployment question instead. A worker who cannot reach a point
steps closer rather than over-extending, and a legged robot can do the same.
So: is there a base placement from which the robot reaches every point the task
requires, while staying inside its well-conditioned range?

For each capture we take the point the demonstrator's configuration asks the
hand to occupy, expressed relative to the shoulder, and find the single
horizontal base translation that brings the whole trajectory within the usable
radius ext_cap * arm_length. One translation for the whole capture, not a new
one per frame, because a stance shift is a step and a per-frame shift is not.

Reported per capture:
  step_cm        magnitude of that one base translation
  residual_cm    how far outside the usable radius the worst frame still is
                 after the step. Zero means the whole task is reachable.
  naive_over_cm  worst-case overshoot with no step, for comparison
  within_tol     whether the residual falls inside the task-attainment
                 tolerance our prior work used, 8 cm

    task_reachability.py --side right
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.fk import ArmChainFK, DEFAULT_URDF                 # noqa: E402
from common.data_paths import joints_csv_for                    # noqa: E402
from common.human_directions import (                           # noqa: E402
    load_frames, per_frame_inputs)

DEFAULT_CAPS = [f"M{m}_{r}" for m in (1, 2, 3, 4, 5) for r in range(1, 6)]


def targets_rel_shoulder(fk, rows, q_rest):
    """Where the demonstrator's configuration asks the hand to be, relative to
    the shoulder, at the robot's own limb lengths. This is the task point."""
    a, e, w = fk.positions(q_rest)
    l_up = float(np.linalg.norm(e - a))
    l_fo = float(np.linalg.norm(w - e))
    P = np.array([l_up * u + l_fo * f for _t, u, f, *_r in rows])
    return P, l_up + l_fo


def best_step(P, R):
    """Smallest horizontal translation b bringing every |P - b| within R.

    Minimizes the worst-case radius, which is the 1-center problem, then reports
    what is left over. The translation is horizontal because the base cannot
    move vertically, so any residual would need a squat or a lean."""
    def worst(b):
        d = P - np.array([b[0], b[1], 0.0])
        return float(np.linalg.norm(d, axis=1).max())
    b0 = P.mean(axis=0)[:2]
    res = minimize(worst, b0, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-5, maxiter=2000))
    b = np.array([res.x[0], res.x[1], 0.0])
    return b, worst(res.x)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", nargs="+", default=DEFAULT_CAPS)
    p.add_argument("--sides", nargs="+", default=["right", "left"])
    p.add_argument("--ext-cap", type=float, default=0.88)
    p.add_argument("--tol-cm", type=float, default=8.0,
                   help="task-attainment tolerance, the value our prior work used")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "task_reach", "reachability.json"))
    args = p.parse_args()

    out = {}
    for side in args.sides:
        fk = ArmChainFK(args.urdf, side)
        q0 = np.array([0.30 if side == "right" else 0.31,
                       -0.24 if side == "right" else 0.24,
                       0.12 if side == "right" else -0.12,
                       0.69 if side == "right" else 0.67])
        rows_out = []
        print(f"\n=== {side} arm, usable radius = {args.ext_cap:.2f} of arm ===")
        print(f"{'capture':9}{'step_cm':>9}{'residual_cm':>13}"
              f"{'naive_over_cm':>15}{'within_tol':>12}")
        for cap in args.captures:
            path = joints_csv_for(cap)
            if not os.path.isfile(path):
                continue
            rows = list(per_frame_inputs(load_frames(path, fps=args.fps),
                                         side, args.fps))
            if len(rows) < 10:
                continue
            P, arm = targets_rel_shoulder(fk, rows, q0)
            R = args.ext_cap * arm
            naive = float(np.linalg.norm(P, axis=1).max()) - R
            b, worst = best_step(P, R)
            residual = max(0.0, worst - R)
            r = dict(capture=cap, step_cm=float(np.linalg.norm(b) * 100),
                     residual_cm=residual * 100,
                     naive_over_cm=max(0.0, naive) * 100,
                     within_tol=bool(residual * 100 <= args.tol_cm),
                     n_frames=len(rows))
            rows_out.append(r)
            print(f"  {cap:7}{r['step_cm']:9.1f}{r['residual_cm']:13.2f}"
                  f"{r['naive_over_cm']:15.1f}{'yes' if r['within_tol'] else 'NO':>12}")
        out[side] = rows_out
        if rows_out:
            st = np.array([r["step_cm"] for r in rows_out])
            rs = np.array([r["residual_cm"] for r in rows_out])
            nv = np.array([r["naive_over_cm"] for r in rows_out])
            full = int((rs < 0.5).sum())
            print(f"  ---- mean step {st.mean():.1f} cm, p95 {np.percentile(st,95):.1f} cm")
            print(f"  ---- fully reachable after one step: {full}/{len(rows_out)} captures")
            print(f"  ---- mean residual {rs.mean():.2f} cm, worst {rs.max():.2f} cm")
            wt = sum(r["within_tol"] for r in rows_out)
            print(f"  ---- within the {args.tol_cm:.0f} cm task tolerance of our prior "
                  f"work: {wt}/{len(rows_out)} captures")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(ext_cap=args.ext_cap, sides=args.sides, per_capture=out),
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
