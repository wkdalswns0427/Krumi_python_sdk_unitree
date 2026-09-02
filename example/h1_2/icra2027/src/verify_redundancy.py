#!/usr/bin/env python3
"""Is the redundancy account in the problem formulation actually true?

The paper argues, from a dimension count, that the arm-only protocol leaves
exactly one spare degree of freedom. Four joints participate, a wrist-position
target consumes three, and the remainder is the swivel, the rotation of the
elbow about the shoulder-to-wrist axis. It then argues that this remainder
vanishes as the arm approaches full extension, because the swivel circle
contracts to a point when the arm is straight.

Both steps are geometric arguments, not measurements. This checks them against
the task Jacobian at the configurations the captures actually visit.

What is computed, per frame:

  rank      numerical rank of the 3x4 position Jacobian, at a tolerance set
            from the matrix norm. Rank 3 means the target is fully constrained
            and one dimension is left over, which is what the paper claims.
            Rank below 3 means the arm cannot even serve the target locally.
  sigma_min smallest singular value. This is the honest version of the claim,
            since rank is a threshold on it and the interesting behaviour is
            the approach to zero rather than the crossing.
  null_dim  4 minus rank, the dimension genuinely available to any secondary
            objective at that configuration.

The prediction the paper needs is that sigma_min falls as extension rises, and
that it approaches zero near full extension. If sigma_min stays comfortably
away from zero at high extension, the swivel argument is wrong and the
formulation paragraph has to be rewritten as a hypothesis.

    verify_redundancy.py --side right
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.fk import ArmChainFK, DEFAULT_URDF                     # noqa: E402
from common.data_paths import joints_csv_for                        # noqa: E402
from common.human_directions import load_frames, per_frame_inputs   # noqa: E402
import x6_taskcentric as x6                                         # noqa: E402

DEFAULT_CAPS = [f"M{m}_{r}" for m in (1, 2, 3, 4, 5) for r in range(1, 6)]


def position_jacobian(fk, q, eps=1e-6):
    """3x4 wrist-position Jacobian by central differences on the FK."""
    J = np.zeros((3, 4))
    for j in range(4):
        qp = np.array(q, dtype=float); qp[j] += eps
        qm = np.array(q, dtype=float); qm[j] -= eps
        J[:, j] = (fk.positions(qp)[2] - fk.positions(qm)[2]) / (2 * eps)
    return J


def extension(fk, q):
    """|wrist - shoulder| over the arm's own segment sum, so it is scale-free
    and cannot exceed one."""
    a, e, w = fk.positions(q)
    seg = np.linalg.norm(e - a) + np.linalg.norm(w - e)
    return float(np.linalg.norm(w - a) / seg) if seg > 0 else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", nargs="+", default=DEFAULT_CAPS)
    p.add_argument("--side", default="right", choices=["right", "left"])
    p.add_argument("--method", default="fidelity_reg",
                   help="condition whose configurations to inspect")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "redundancy", "jacobian_rank.json"))
    args = p.parse_args()

    fk = ArmChainFK(args.urdf, args.side)
    ext_all, smin_all, rank_all = [], [], []
    per_cap = []

    for cap in args.captures:
        src = joints_csv_for(cap)
        if not os.path.isfile(src):
            continue
        _stats, _ts, qs, _tg = x6.solve_rep(src, args.side, args.urdf,
                                            args.fps, method=args.method)
        e, s, r = [], [], []
        for q in qs:
            J = position_jacobian(fk, q)
            sv = np.linalg.svd(J, compute_uv=False)
            tol = max(J.shape) * np.finfo(float).eps * sv[0]
            e.append(extension(fk, q))
            s.append(float(sv[-1]))
            r.append(int((sv > tol).sum()))
        ext_all += e; smin_all += s; rank_all += r
        per_cap.append(dict(capture=cap, n=len(qs),
                            sigma_min_median=float(np.median(s)),
                            sigma_min_p05=float(np.percentile(s, 5)),
                            rank3_frac=float(np.mean(np.array(r) == 3)),
                            ext_p95=float(np.percentile(e, 95))))
        print(f"  {cap:8} n={len(qs):5d}  rank3 {np.mean(np.array(r)==3)*100:5.1f}%"
              f"  sigma_min med {np.median(s):.4f}  p05 {np.percentile(s,5):.4f}")

    if not ext_all:
        sys.exit("no captures found")
    ext = np.array(ext_all); smin = np.array(smin_all); rank = np.array(rank_all)

    print(f"\n{'='*66}\nCLAIM 1: the task leaves exactly one spare dimension\n{'='*66}")
    for k in (4, 3, 2, 1):
        f = float(np.mean(rank == k))
        if f > 0:
            print(f"  rank {k}, null dimension {4-k}: {f*100:6.2f}% of frames")
    print(f"  -> the dimension count holds on {np.mean(rank==3)*100:.2f}% of frames")

    print(f"\n{'='*66}\nCLAIM 2: that dimension vanishes near full extension\n{'='*66}")
    print(f"{'extension band':>18}{'n':>8}{'sigma_min median':>19}{'p05':>10}")
    edges = [0.0, 0.80, 0.85, 0.90, 0.94, 0.97, 1.01]
    band = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ext >= lo) & (ext < hi)
        if m.sum() < 20:
            continue
        band.append((lo, hi, int(m.sum()), float(np.median(smin[m])),
                     float(np.percentile(smin[m], 5))))
        print(f"{f'{lo:.2f}-{hi:.2f}':>18}{m.sum():>8}{np.median(smin[m]):>19.4f}"
              f"{np.percentile(smin[m],5):>10.4f}")
    ok = np.isfinite(ext) & np.isfinite(smin)
    r = float(np.corrcoef(ext[ok], smin[ok])[0, 1])
    print(f"\n  correlation of sigma_min with extension: {r:+.3f}")
    if band:
        lo_b, hi_b = band[0], band[-1]
        print(f"  sigma_min median falls {lo_b[3]:.4f} -> {hi_b[3]:.4f} "
              f"across {lo_b[0]:.2f} to {hi_b[1]:.2f} extension "
              f"({(1-hi_b[3]/lo_b[3])*100:.0f}% reduction)")
        print("\n  VERDICT: " + (
            "supported. Redundancy contracts as extension rises."
            if r < -0.2 and hi_b[3] < lo_b[3] else
            "NOT supported by this data. Rewrite the paragraph as a hypothesis."))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(side=args.side, method=args.method,
                   n_frames=int(len(ext)),
                   rank_hist={str(k): float(np.mean(rank == k)) for k in (1, 2, 3, 4)},
                   corr_sigma_extension=r,
                   bands=[dict(lo=a, hi=b, n=c, sigma_median=d, sigma_p05=e)
                          for a, b, c, d, e in band],
                   per_capture=per_cap),
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
