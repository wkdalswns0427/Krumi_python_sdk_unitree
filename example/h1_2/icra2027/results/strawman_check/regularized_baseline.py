"""Is B1 a strawman? Run fidelity WITH the IROS yaw regularizers.

decisions.md 2026-08-25 flags this as the reviewer objection and offers the
regularized numbers as an optional 'middle column'. This measures it.
"""
import sys, os
import numpy as np
sys.path.insert(0, "src")
from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4
from common.ik_baseline import solve_fidelity
from common.ik_taskcentric import solve_taskcentric
from common.human_directions import load_frames, per_frame_inputs, robot_scaled_wrist_target
from common.metrics import flip_count, limit_violation, limit_margin
from common.data_paths import joints_csv_for

SIDE, FPS = "right", 60.0
CAPS = ["M2_1", "M2_2", "M2_3", "M3_1", "M3_2", "M3_3"]
REST = np.array([0.30, -0.24, 0.12, 0.69])

# (label, kind, kwargs)
ARMS = [
    ("fidelity naive",     "fid",  dict(yaw_reg=0.0,  yaw_neutral=0.0)),
    ("fidelity IROS-reg",  "fid",  dict(yaw_reg=0.05, yaw_neutral=0.02)),
    ("fidelity strong-reg","fid",  dict(yaw_reg=0.20, yaw_neutral=0.10)),
    ("task-centric",       "task", dict()),
]

fk = ArmChainFK(DEFAULT_URDF, SIDE)
lims = load_limits_4(DEFAULT_URDF)[SIDE]
lo, hi = lims["lo"], lims["hi"]

agg = {lab: dict(flips=0, viol_reps=0, margins=[], tip=[], reps=0) for lab, _, _ in ARMS}
per = {lab: [] for lab, _, _ in ARMS}

for cap in CAPS:
    frames = load_frames(joints_csv_for(cap), fps=FPS)
    inputs = list(per_frame_inputs(frames, SIDE, FPS))
    for lab, kind, kw in ARMS:
        q_prev = REST.copy()
        ts, qs, viols, margins, tips = [], [], [], [], []
        for t, u_hat, f_hat, _w in inputs:
            target = robot_scaled_wrist_target(fk, u_hat, f_hat)(q_prev)
            if kind == "fid":
                q, _, _ = solve_fidelity(fk, u_hat, f_hat, q_prev, lims, **kw)
            else:
                q, _, _ = solve_taskcentric(fk, target, q_prev, lims,
                                            lambda_limit=0.05, lambda_cont=0.02)
            ts.append(t); qs.append(q)
            viols.append(limit_violation(q, lo, hi))
            margins.append(limit_margin(q, lo, hi))
            tips.append(float(np.linalg.norm(fk.tip(q) - target)))
            q_prev = q
        ts = np.asarray(ts); qs = np.asarray(qs)
        dt = float(np.median(np.diff(ts)))
        f = flip_count(qs, dt, threshold_rad_per_s=6.0, joint_index=2)
        a = agg[lab]
        a["flips"] += f; a["viol_reps"] += int(np.any(viols)); a["reps"] += 1
        a["margins"].append(float(np.mean(margins))); a["tip"].append(float(np.mean(tips))*1000)
        per[lab].append((cap, f, int(np.sum(viols))))

print(f"{'condition':<20s} {'flips':>6s} {'viol reps':>10s} {'margin':>8s} {'tip mm':>8s}")
print("-"*56)
for lab, _, _ in ARMS:
    a = agg[lab]
    print(f"{lab:<20s} {a['flips']:>6d} {a['viol_reps']:>8d}/6 "
          f"{np.mean(a['margins']):>8.3f} {np.mean(a['tip']):>8.2f}")
print()
print("per-rep flips (viol frames):")
print(f"{'':<20s}" + "".join(f"{c:>12s}" for c in CAPS))
for lab, _, _ in ARMS:
    print(f"{lab:<20s}" + "".join(f"{f:>6d}({v:<4d})" for _, f, v in per[lab]))
