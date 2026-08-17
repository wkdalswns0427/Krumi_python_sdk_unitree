#!/usr/bin/env python3
"""x2_costoffidelity.py - ICRA headline experiment (X2 / contribution C3).

For the SAME task, what does reproducing the human's CONFIGURATION cost the
robot, versus a task-centric configuration that achieves the same end-effector
path? Task achievement is held fixed (both hit the clean B1 wrist path), so any
difference is the price of insisting on the human pose, for no task benefit.

Two configurations per frame, right arm, M2 (bricklaying) and M3 (lifting):
  fidelity   = the clean B1 retarget (matches the human limb directions). This
               is the deployable fidelity retargeter, flip-free (yaw hack ON),
               so the cost measured is the cost of the POSE, not of flips.
  task-cent. = the X6 causal task-centric solve on the same EE target.

Robot-side cost metrics:
  manipulability  Yoshikawa index sqrt(det(J J^T)) of the 3x4 position Jacobian
                  (higher = farther from singularity). Report mean and MIN.
  gravity load    sum |holding torque| over the arm (Nm), IROS-validated URDF
                  model. Report mean and peak.
  limit margin    min distance to any joint limit along the trajectory (rad).
  peak joint vel  max |dq|/dt.

Headline sentence target: "reproducing the worker's configuration costs the
robot X% of its manipulability margin and Y Nm of additional gravity load, for
no task benefit."

Also emits the side-by-side configuration figure at the task instant of largest
manipulability gap (x2_config_figure.pdf/.png) - the picture that makes C1.

Usage:
    /usr/bin/python3 x2_costoffidelity.py --motions M2 M3 --reps 1 2 3
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments", "scripts")
sys.path.insert(0, HERE)
sys.path.insert(0, EXP)
from retarget_arm import ArmChainFK, load_limits, _unit          # noqa: E402
from replay_arm import GravityFF                                 # noqa: E402
from x6_taskcentric import retarget, read_traj, RSEG, DATA, URDF, x6_solve  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8})
WEIGHTS = dict(pos=50.0, dir=1.0, lim=0.05, manip=2e-4, cont=0.2)
FID_C, TC_C = "#d1495b", "#00798c"


def manip(fk, q):
    wr = fk.positions(q)[2]
    J = np.zeros((3, 4))
    for k in range(4):
        qp = q.copy(); qp[k] += 1e-5
        J[:, k] = (fk.positions(qp)[2] - wr) / 1e-5
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def metrics(fk, gff, Q, t, lim):
    lo = np.array([l for l, _ in lim]); hi = np.array([h for _, h in lim])
    w = np.array([manip(fk, q) for q in Q])
    grav = np.array([sum(abs(x) for x in gff.tau(list(q) + [0, 0, 0]))
                     for q in Q])
    margin = np.minimum(Q - lo, hi - Q).min(axis=1)     # per-frame worst margin
    dt = np.diff(t); dt[dt <= 0] = np.median(dt[dt > 0])
    vel = np.abs(np.diff(Q, axis=0)) / dt[:, None]
    return dict(w_mean=float(w.mean()), w_min=float(w.min()),
                g_mean=float(grav.mean()), g_peak=float(grav.max()),
                margin_min=float(margin.min()),
                vel_peak=float(vel.max()), w=w)


def run_rep(motion, rep):
    tag = f"{motion}_{rep}"
    joints = os.path.join(DATA, tag, "b2g", "iph_mono.csv")
    if not os.path.exists(joints):
        return None
    fk = ArmChainFK(URDF, "right")
    gff = GravityFF(URDF, "right")
    lim = [load_limits(URDF)[f"right_{s}_joint"] for s in RSEG]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        clean = os.path.join(td, "clean.csv")
        retarget(joints, clean, 0.02, 0.05)             # fidelity config
        t, qfid = read_traj(clean)
    pos = np.array([fk.positions(q)[2] for q in qfid])
    adir = np.array([_unit(fk.positions(q)[2] - fk.positions(q)[1])
                     for q in qfid])
    qtc = x6_solve(fk, pos, adir, lim, WEIGHTS, greedy=True)      # task-centric
    return dict(tag=tag, t=t, fk=fk, lim=lim, qfid=qfid, qtc=qtc,
                fid=metrics(fk, gff, qfid, t, lim),
                tc=metrics(fk, gff, qtc, t, lim))


def config_figure(res, out_stem):
    """Side-by-side arm config at the instant of largest manipulability gap."""
    fk = res["fk"]
    gap = res["tc"]["w"] - res["fid"]["w"]              # fidelity below TC
    i = int(np.argmax(gap))
    fig, (axs, axt) = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for ax, (a, b, xl, yl, ttl) in (
            (axs, (0, 2, "x  (forward +, m)", "z  (up +, m)", "side view")),
            (axt, (1, 0, "y  (left +, m)", "x  (forward +, m)", "top view"))):
        for q, col, lab in ((res["qfid"][i], FID_C, "fidelity (human pose)"),
                            (res["qtc"][i], TC_C, "task-centric")):
            sh, el, wr = fk.positions(q)
            pts = np.array([sh, el, wr])
            ax.plot(pts[:, a], pts[:, b], "-o", color=col, lw=2.2, ms=5,
                    label=lab)
        ax.scatter(fk.positions(res["qfid"][i])[2][a],
                   fk.positions(res["qfid"][i])[2][b], s=90, marker="*",
                   color="#edae49", edgecolor="k", zorder=5, label="EE target")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.set_aspect("equal", "box"); ax.grid(alpha=0.3, lw=0.5)
    axs.legend(loc="best", fontsize=7)
    fig.suptitle(f"Same task instant, {res['tag']}: fidelity vs task-centric "
                 f"arm configuration", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[x2] wrote {os.path.basename(out_stem)}.pdf + .png "
          f"(frame {i}, manip gap {gap[i]:.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motions", nargs="+", default=["M2", "M3"])
    p.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--fig-rep", default="M2_1", help="rep for the config figure")
    args = p.parse_args()

    print("X2 cost of fidelity: reproducing the human pose vs task-centric, "
          "same task, right arm\n")
    hd = (f"{'rep':7}{'manip mean':>12}{'manip min':>11}{'grav mean Nm':>14}"
          f"{'grav peak':>11}{'limit min rad':>15}")
    print(hd)
    print(f"{'':7}{'fid / tc':>12}{'fid / tc':>11}{'fid / tc':>14}"
          f"{'fid / tc':>11}{'fid / tc':>15}")
    print("-" * len(hd))
    agg = {"wm_f": [], "wm_t": [], "g_f": [], "g_t": [], "mar_f": [], "mar_t": []}
    figrep = None
    for m in args.motions:
        for r in args.reps:
            res = run_rep(m, r)
            if res is None:
                print(f"{m}_{r}: (missing)"); continue
            f, tc = res["fid"], res["tc"]
            print(f"{res['tag']:7}{f['w_mean']:.4f}/{tc['w_mean']:.4f}"
                  f"{f['w_min']:>6.4f}/{tc['w_min']:.4f}"
                  f"{f['g_mean']:>7.1f}/{tc['g_mean']:<5.1f}"
                  f"{f['g_peak']:>6.1f}/{tc['g_peak']:<5.1f}"
                  f"{f['margin_min']:>8.3f}/{tc['margin_min']:.3f}")
            agg["wm_f"].append(f["w_mean"]); agg["wm_t"].append(tc["w_mean"])
            agg["g_f"].append(f["g_mean"]); agg["g_t"].append(tc["g_mean"])
            agg["mar_f"].append(f["margin_min"]); agg["mar_t"].append(tc["margin_min"])
            if res["tag"] == args.fig_rep:
                figrep = res
    if agg["wm_f"]:
        wf, wt = np.mean(agg["wm_f"]), np.mean(agg["wm_t"])
        gf, gt = np.mean(agg["g_f"]), np.mean(agg["g_t"])
        mf, mt = np.mean(agg["mar_f"]), np.mean(agg["mar_t"])
        print("\n=== COST OF FIDELITY (mean over reps) ===")
        print(f"  manipulability: fidelity {wf:.4f} vs task-centric {wt:.4f}  "
              f"-> fidelity gives up {100*(wt-wf)/wt:+.0f}% of the margin")
        print(f"  gravity load  : fidelity {gf:.1f} Nm vs {gt:.1f} Nm  "
              f"-> {gf-gt:+.1f} Nm extra")
        print(f"  limit margin  : fidelity {mf:.3f} rad vs {mt:.3f} rad  "
              f"-> {100*(mt-mf)/mt:+.0f}% closer to a limit")
    if figrep is not None:
        config_figure(figrep, os.path.join(HERE, "x2_config_figure"))


if __name__ == "__main__":
    main()
