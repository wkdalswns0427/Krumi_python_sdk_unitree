#!/usr/bin/env python3
"""x2_robot_optimal.py - X2 headline, robot-optimal variant (contribution C3).

X2 v1 (x2_costoffidelity.py) found ~0 manipulability/gravity cost because the
task-centric config matched the approach direction EVERY frame, over-
constraining the arm so there was no null space to exploit. This variant frees
the redundancy: it enforces the approach (forearm) direction ONLY at contact
EVENTS (low end-effector speed = dwell/strike/place); between events the swivel
is free, and the solver picks the ROBOT-OPTIMAL configuration on the task
manifold.

Method, right arm, per frame:
  - many seeds spanning the shoulder-yaw range (to cover the 1-DOF redundancy)
    plus the previous frame and mid-range,
  - each minimized on a CHEAP objective (EE position + limit margin + continuity
    + approach dir only if this frame is an event), 1 FK per eval,
  - candidates that hit the EE target within TOL are RANKED by robot health
    (manipulability up, gravity load down, limit margin up); the healthiest is
    kept. If none hit TOL, the closest is kept.

This isolates the true cost of fidelity: for the identical task, how much
manipulability margin, gravity load, and joint-limit margin does insisting on
the human configuration give up versus a robot-optimal one.

Usage:
    /usr/bin/python3 x2_robot_optimal.py --motions M2 M3 --reps 1 2 3
"""

import argparse
import os
import sys
import tempfile

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments", "scripts")
sys.path.insert(0, HERE)
sys.path.insert(0, EXP)
from retarget_arm import ArmChainFK, load_limits, _unit          # noqa: E402
from replay_arm import GravityFF                                 # noqa: E402
from x6_taskcentric import retarget, read_traj, RSEG, DATA, URDF  # noqa: E402
from x2_costoffidelity import manip, metrics                     # noqa: E402
import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8})
FID_C, RO_C = "#d1495b", "#00798c"
TOL_M = 0.03                                   # task position tolerance
EVENT_SPEED = 0.04                             # m/s below this = contact event
W = dict(pos=50.0, dir=1.0, lim=0.05, cont=0.12)
# robot-health ranking weights (scaled so terms are O(1))
RH = dict(manip=1 / 0.02, grav=1 / 12.0, margin=0.5)


def robot_optimal(fk, gff, x_t, a_t, event, limits, n_seed=4, seed=0):
    lo = np.array([l for l, _ in limits]); hi = np.array([h for _, h in limits])
    mid = 0.5 * (lo + hi); half = 0.5 * (hi - lo)
    yaw_grid = np.linspace(lo[2] + 0.1, hi[2] - 0.1, n_seed)
    rng = np.random.default_rng(seed)
    bounds = list(zip(lo, hi))

    def cost(q, xt, at, qprev, use_dir):
        a, e, w = fk.positions(q)
        L = W["pos"] * np.sum((w - xt) ** 2) \
            + W["lim"] * np.sum(((q - mid) / half) ** 4) \
            + W["cont"] * np.sum((q - qprev) ** 2)
        if use_dir:
            fd = _unit(w - e)
            if fd is not None:
                L += W["dir"] * np.sum((fd - at) ** 2)
        return L

    def health(q):
        m = manip(fk, q)
        g = sum(abs(x) for x in gff.tau(list(q) + [0, 0, 0]))
        margin = float(np.min(np.minimum(q - lo, hi - q)))
        return -RH["manip"] * m + RH["grav"] * g + RH["margin"] / (margin + 0.02)

    Q = np.zeros((len(x_t), 4))
    qprev = mid.copy()
    for i in range(len(x_t)):
        seeds = [qprev, mid]
        for y in yaw_grid:                      # span the redundancy
            s = mid.copy(); s[2] = y; seeds.append(s)
        cands = []
        for s in seeds:
            r = minimize(cost, np.clip(s, lo, hi),
                         args=(x_t[i], a_t[i], qprev, bool(event[i])),
                         method="L-BFGS-B", bounds=bounds, options=dict(maxiter=30))
            err = np.linalg.norm(fk.positions(r.x)[2] - x_t[i])
            cands.append((r.x, err))
        ok = [c for c in cands if c[1] < TOL_M]
        pool = ok if ok else [min(cands, key=lambda c: c[1])]
        best = min(pool, key=lambda c: health(c[0]))[0]
        Q[i] = best
        qprev = best
    return Q


def run_rep(motion, rep):
    tag = f"{motion}_{rep}"
    joints = os.path.join(DATA, tag, "b2g", "iph_mono.csv")
    if not os.path.exists(joints):
        return None
    fk = ArmChainFK(URDF, "right"); gff = GravityFF(URDF, "right")
    lim = [load_limits(URDF)[f"right_{s}_joint"] for s in RSEG]
    with tempfile.TemporaryDirectory() as td:
        clean = os.path.join(td, "clean.csv")
        retarget(joints, clean, 0.02, 0.05)
        t, qfid = read_traj(clean)
    pos = np.array([fk.positions(q)[2] for q in qfid])
    adir = np.array([_unit(fk.positions(q)[2] - fk.positions(q)[1]) for q in qfid])
    dt = np.diff(t); dt[dt <= 0] = np.median(dt[dt > 0])
    speed = np.concatenate([[0], np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt])
    event = speed < EVENT_SPEED
    Qro = robot_optimal(fk, gff, pos, adir, event, lim)
    ro_err = float(np.linalg.norm(np.array([fk.positions(q)[2] for q in Qro])
                                  - pos, axis=1).mean() * 100)
    return dict(tag=tag, t=t, fk=fk, qfid=qfid, qro=Qro, ro_err=ro_err,
                ev=float(event.mean() * 100),
                fid=metrics(fk, gff, qfid, t, lim),
                ro=metrics(fk, gff, Qro, t, lim))


def config_figure(res, stem):
    fk = res["fk"]
    gap = res["ro"]["w"] - res["fid"]["w"]
    i = int(np.argmax(gap))
    fig, (axs, axt) = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for ax, (a, b, xl, yl, ttl) in (
            (axs, (0, 2, "x  (forward +, m)", "z  (up +, m)", "side view")),
            (axt, (1, 0, "y  (left +, m)", "x  (forward +, m)", "top view"))):
        for q, col, lab in ((res["qfid"][i], FID_C, "fidelity (human pose)"),
                            (res["qro"][i], RO_C, "robot-optimal")):
            p = np.array(fk.positions(q))
            ax.plot(p[:, a], p[:, b], "-o", color=col, lw=2.2, ms=5, label=lab)
        ee = fk.positions(res["qfid"][i])[2]
        ax.scatter(ee[a], ee[b], s=90, marker="*", color="#edae49",
                   edgecolor="k", zorder=5, label="EE target")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.set_aspect("equal", "box"); ax.grid(alpha=0.3, lw=0.5)
    axs.legend(loc="best", fontsize=7)
    fig.suptitle(f"Same task instant, {res['tag']}: fidelity vs robot-optimal "
                 f"configuration", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[x2ro] wrote {os.path.basename(stem)}.pdf + .png (frame {i}, "
          f"manip gap {gap[i]:.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motions", nargs="+", default=["M2", "M3"])
    p.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--fig-rep", default="M2_1")
    args = p.parse_args()
    print("X2 robot-optimal: fidelity (human pose) vs robot-optimal, same task, "
          "right arm. Approach dir enforced at contact events only.\n")
    hd = (f"{'rep':7}{'event%':>7}{'manip mean f/ro':>18}{'manip min f/ro':>17}"
          f"{'grav mean f/ro':>17}{'lim min f/ro':>15}{'ro err':>8}")
    print(hd); print("-" * len(hd))
    agg = {k: [] for k in ("wf", "wr", "gf", "gr", "mf", "mr")}
    figrep = None
    for m in args.motions:
        for r in args.reps:
            res = run_rep(m, r)
            if res is None:
                print(f"{m}_{r}: (missing)"); continue
            f, ro = res["fid"], res["ro"]
            print(f"{res['tag']:7}{res['ev']:>6.0f}%"
                  f"{f['w_mean']:>10.4f}/{ro['w_mean']:.4f}"
                  f"{f['w_min']:>9.4f}/{ro['w_min']:.4f}"
                  f"{f['g_mean']:>9.1f}/{ro['g_mean']:<5.1f}"
                  f"{f['margin_min']:>8.3f}/{ro['margin_min']:.3f}{res['ro_err']:>7.2f}")
            agg["wf"].append(f["w_mean"]); agg["wr"].append(ro["w_mean"])
            agg["gf"].append(f["g_mean"]); agg["gr"].append(ro["g_mean"])
            agg["mf"].append(f["margin_min"]); agg["mr"].append(ro["margin_min"])
            if res["tag"] == args.fig_rep:
                figrep = res
    if agg["wf"]:
        wf, wr = np.mean(agg["wf"]), np.mean(agg["wr"])
        gf, gr = np.mean(agg["gf"]), np.mean(agg["gr"])
        mf, mr = np.mean(agg["mf"]), np.mean(agg["mr"])
        print("\n=== COST OF FIDELITY vs robot-optimal (mean over reps) ===")
        print(f"  manipulability: {wf:.4f} vs {wr:.4f}  -> fidelity gives up "
              f"{100*(wr-wf)/wr:+.0f}% of the margin")
        print(f"  gravity load  : {gf:.1f} vs {gr:.1f} Nm  -> {gf-gr:+.1f} Nm")
        print(f"  limit margin  : {mf:.3f} vs {mr:.3f} rad  -> "
              f"{100*(mr-mf)/mr:+.0f}% closer to a limit")
    if figrep is not None:
        config_figure(figrep, os.path.join(HERE, "x2_robotopt_figure"))


if __name__ == "__main__":
    main()
