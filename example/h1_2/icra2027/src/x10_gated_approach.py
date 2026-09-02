#!/usr/bin/env python3
"""X10: does gating approach direction to contact events buy tolerance cheaply?

C3's frontier currently looks lopsided: H2O and OKAMI lose 0.1 deg of approach
direction, task-centric loses 22.2. That is because L_task was wrist position
only. This measures the gated approach term (common/ik_taskcentric.py) which is
supposed to buy back approach accuracy WHERE THE TASK NEEDS IT without paying
posture-matching's feasibility cost everywhere else.

Three conditions on identical targets:
  free        w_approach = 0                       (the old task-centric)
  gated       approach constrained on contact frames only
  always-on   approach constrained on every frame  (the failure mode, shown
              on purpose: 6 residuals on 4 DOF kills the null space)

Reported separately ON and OFF contact frames, because a rep-mean hides the
whole point.
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4
from common.ik_taskcentric import solve_taskcentric
from common.human_directions import load_frames, per_frame_inputs, robot_scaled_wrist_target
from common.metrics import limit_margin, limit_violation, approach_direction_error_deg
from common.contact_events import gate_for_capture
from common.data_paths import joints_csv_for

REST = np.array([0.30, -0.24, 0.12, 0.69])
CONDS = [("free", 0.0, "none"), ("gated", 0.5, "contact"), ("always-on", 0.5, "all")]


def run(cap, side, urdf, fps, w_app, mode):
    fk = ArmChainFK(urdf, side); lims = load_limits_4(urdf)[side]
    lo, hi = lims["lo"], lims["hi"]
    rows = list(per_frame_inputs(load_frames(joints_csv_for(cap), fps=fps), side, fps))
    ts = np.array([r[0] for r in rows]); dt = float(np.median(np.diff(ts)))

    # gate is built from the HUMAN wrist path, which is what defines the task
    q0 = REST.copy(); human_wrist = []
    for _t, u, f, _w in rows:
        a, e, w = fk.positions(q0)
        human_wrist.append(a + np.linalg.norm(e-a)*u + np.linalg.norm(w-e)*f)
    gate, src = gate_for_capture(np.array(human_wrist), dt, capture=cap)
    if mode == "none":   gate = np.zeros(len(rows), bool)
    elif mode == "all":  gate = np.ones(len(rows), bool)

    q_prev = REST.copy(); M, V, A, T = [], [], [], []
    for i, (_t, u_hat, f_hat, _w) in enumerate(rows):
        tgt = robot_scaled_wrist_target(fk, u_hat, f_hat)(q_prev)
        q, _r, _i = solve_taskcentric(
            fk, tgt, q_prev, lims, lambda_limit=0.05, lambda_cont=0.02,
            target_forearm=(f_hat if gate[i] else None), w_approach=w_app)
        M.append(limit_margin(q, lo, hi)); V.append(limit_violation(q, lo, hi))
        A.append(approach_direction_error_deg(q, fk, f_hat))
        T.append(float(np.linalg.norm(fk.tip(q) - tgt)) * 1000)
        q_prev = q
    A = np.array(A); M = np.array(M); T = np.array(T); V = np.array(V)
    on = gate if gate.any() else np.zeros(len(rows), bool)
    return dict(capture=cap, gate_source=src, gate_frac=float(gate.mean()),
                margin=float(M.mean()), viol=bool(V.any()), tip_mm=float(T.mean()),
                appr_all=float(np.nanmean(A)),
                appr_on=float(np.nanmean(A[on])) if on.any() else float("nan"),
                appr_off=float(np.nanmean(A[~on])) if (~on).any() else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captures", nargs="+",
                   default=["M2_1","M2_2","M2_3","M3_1","M3_2","M3_3"])
    p.add_argument("--side", default="right"); p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out", default=os.path.join(HERE,"..","results","x10","gated_approach.json"))
    a = p.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = {}
    print(f"{'condition':<12s}{'margin':>8s}{'viol':>7s}{'tip_mm':>8s}"
          f"{'appr@contact':>14s}{'appr@free':>11s}{'gate%':>7s}")
    print("-"*68)
    for name, w, mode in CONDS:
        rs = [run(c, a.side, a.urdf, a.fps, w, mode) for c in a.captures]
        out[name] = rs
        print(f"{name:<12s}{np.mean([r['margin'] for r in rs]):>8.3f}"
              f"{sum(r['viol'] for r in rs):>5d}/{len(rs):<2d}"
              f"{np.mean([r['tip_mm'] for r in rs]):>8.2f}"
              f"{np.nanmean([r['appr_on'] for r in rs]):>14.1f}"
              f"{np.nanmean([r['appr_off'] for r in rs]):>11.1f}"
              f"{100*np.mean([r['gate_frac'] for r in rs]):>7.1f}")
    json.dump(out, open(a.out,"w"), indent=2)
    print(f"\ngate source: {rs[0]['gate_source']}  (annotated > detected; see open_questions item 5)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
