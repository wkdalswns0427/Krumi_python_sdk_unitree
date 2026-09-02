"""Does posture fidelity cost feasibility, or does limit avoidance buy it?

The B3 sweep showed margin barely moves as OKAMI's posture weight drops three
decades (0.241 -> 0.261), then jumps to 0.333 for task-centric. Task-centric
differs in TWO ways at once: no posture term AND null-space limit avoidance.
That confound is exactly the reviewer objection PLAN_v2.md anticipates.

This crosses the two factors independently inside one solver family, so only
one thing changes per cell.
"""
import sys, itertools, json, argparse, os
import numpy as np
sys.path.insert(0, "src")
from common.fk import ArmChainFK, DEFAULT_URDF, load_limits_4
from common.ik_keypoint import solve_okami_style
from common.human_directions import load_frames, per_frame_inputs, robot_scaled_arm_targets
from common.metrics import flip_count, limit_violation, limit_margin
from common.data_paths import joints_csv_for

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--captures", nargs="+",
                 default=[f"M{m}_{r}" for m in (2, 3) for r in range(1, 6)])
_ap.add_argument("--side", default="right", choices=["left", "right"])
_ap.add_argument("--fps", type=float, default=60.0)
_ap.add_argument("--out", default="results/b3/factorial.json")
_args = _ap.parse_args()

SIDE, FPS = _args.side, _args.fps
CAPS = _args.captures
REST = np.array([0.30, -0.24, 0.12, 0.69] if SIDE == "right" else [0.31, 0.24, -0.12, 0.67])

POSTURE = [("posture ON  (OKAMI pub)", 1.0), ("posture OFF (1e-3 gain)", 0.001)]
LIMITAV = [("limits OFF", 0.0), ("limits w=0.05", 0.05),
           ("limits w=0.20", 0.20), ("limits w=0.50", 0.50)]

fk = ArmChainFK(DEFAULT_URDF, SIDE)
lims = load_limits_4(DEFAULT_URDF)[SIDE]
lo, hi = lims["lo"], lims["hi"]

cache = {c: list(per_frame_inputs(load_frames(joints_csv_for(c), fps=FPS), SIDE, FPS))
         for c in CAPS}

print(f"{'posture':<24s}{'limit avoid':<16s}{'margin':>8s}{'viol':>7s}{'flips':>7s}{'tip_mm':>9s}{'appr':>7s}")
print("-" * 78)
grid = {}
per_rep = {}
N = len(CAPS)
for (plab, gain), (llab, wl) in itertools.product(POSTURE, LIMITAV):
    M, V, F, T, A = [], 0, 0, [], []
    reps = []
    for cap in CAPS:
        q_prev = REST.copy(); qs, ms, vs, ts, ap = [], [], [], [], []
        for t, u_hat, f_hat, _w in cache[cap]:
            _e, wt = robot_scaled_arm_targets(fk, u_hat, f_hat)(q_prev)
            q, _r, _i = solve_okami_style(fk, wt, u_hat, f_hat, q_prev, lims,
                                          w_shoulder=0.04*gain, w_elbow=0.08*gain,
                                          w_limit=wl)
            qs.append(q); ms.append(limit_margin(q, lo, hi))
            vs.append(limit_violation(q, lo, hi))
            ts.append(float(np.linalg.norm(fk.tip(q)-wt)))
            a, e, w = fk.positions(q); v = w - e
            ap.append(np.degrees(np.arccos(np.clip(np.dot(v/np.linalg.norm(v), f_hat), -1, 1))))
            q_prev = q
        qs = np.asarray(qs)
        reps.append(dict(capture=cap, margin=float(np.mean(ms)),
                         viol_frames=int(np.sum(vs)), any_violation=bool(np.any(vs)),
                         tip_mm=float(np.mean(ts)*1000), appr_deg=float(np.mean(ap))))
        M.append(np.mean(ms)); V += int(np.any(vs))
        F += flip_count(qs, float(np.median(np.diff([t for t,_,_,_ in cache[cap]]))), 6.0, 2); T.append(np.mean(ts)*1000); A.append(np.mean(ap))
    grid[(plab, llab)] = (np.mean(M), V, F, np.mean(T), np.mean(A))
    per_rep[f'{plab.strip()} | {llab}'] = reps
    print(f"{plab:<24s}{llab:<16s}{np.mean(M):>8.3f}{V:>4d}/{N:<2d}{F:>7d}{np.mean(T):>9.2f}{np.mean(A):>7.1f}")

print()
print("MAIN EFFECTS on limit margin")
for llab, _ in LIMITAV:
    on  = grid[(POSTURE[0][0], llab)][0]; off = grid[(POSTURE[1][0], llab)][0]
    print(f"  at {llab:<15s} posture ON {on:.3f} -> OFF {off:.3f}   posture costs {off-on:+.3f}")
for plab, _ in POSTURE:
    a = grid[(plab, LIMITAV[0][0])][0]; b = grid[(plab, LIMITAV[-1][0])][0]
    print(f"  at {plab:<24s} limits OFF {a:.3f} -> w=0.50 {b:.3f}   limit avoidance buys {b-a:+.3f}")

os.makedirs(os.path.dirname(os.path.abspath(_args.out)), exist_ok=True)
json.dump(dict(side=SIDE, fps=FPS, captures=CAPS, n=N,
               cells=[dict(posture=p_, limit=l_, margin=v[0], reps_violating=v[1],
                           flips=v[2], tip_mm=v[3], appr_deg=v[4])
                      for (p_, l_), v in grid.items()],
               per_rep=per_rep),
          open(_args.out, "w"), indent=2)
print(f"\nwrote {_args.out}  (n={N} captures)")
