#!/usr/bin/env python3
"""Sensitivity of the task-defined target to its extension ceiling.

C1's remedy re-sets the reach magnitude to min(ext_human * ext_scale, ext_cap).
The value 0.88 was read off one subject's reach sweep, where limit contact
switches on near 90 percent extension. A single tuned number carrying a primary
contribution is the first thing a reviewer will press, so this reports the
ceiling as a curve instead.

For each ceiling it reports, per side, the captures showing limit contact, the
mean limit margin, and the tip error against the target the ceiling defines.
Tip error is not a tracking failure here: a lower ceiling deliberately asks for
a less extended pose, so the number to watch is limit contact against ceiling,
and the flat region around the chosen value.

    ext_cap_sweep.py --side right
    ext_cap_sweep.py --caps 0.80 0.84 0.88 0.92 0.96 1.00
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.fk import DEFAULT_URDF                      # noqa: E402
from common.data_paths import joints_csv_for            # noqa: E402
import x6_taskcentric as x6                             # noqa: E402

DEFAULT_CAPS = [f"M{m}_{r}" for m in (1, 2, 3, 4, 5) for r in range(1, 6)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", nargs="+", default=DEFAULT_CAPS)
    p.add_argument("--sides", nargs="+", default=["right", "left"])
    p.add_argument("--caps", nargs="+", type=float,
                   default=[0.78, 0.82, 0.86, 0.88, 0.90, 0.94, 0.98, 1.00])
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "ext_cap", "sweep.json"))
    args = p.parse_args()

    out = {}
    for side in args.sides:
        print(f"\n=== {side} arm ===")
        print(f"{'ext_cap':>8}{'contact':>10}{'margin':>9}{'tip_mm':>9}{'flips':>8}")
        rows = []
        for cap in args.caps:
            viol = marg = tip = flips = n = 0
            per = []
            for c in args.captures:
                path = joints_csv_for(c)
                if not os.path.isfile(path):
                    continue
                st, _ts, _qs, _tg = x6.solve_rep(
                    path, side, args.urdf, args.fps,
                    method="taskdefined", ext_cap=cap)
                viol += 1 if st["any_violation"] else 0
                marg += st["margin_mean"]; tip += st["tip_err_mean_mm"]
                flips += st["flips_yaw"]; n += 1
                per.append(dict(capture=c, **st))
            if not n:
                continue
            r = dict(ext_cap=cap, n=n, contact=viol,
                     margin=marg / n, tip_mm=tip / n, flips=flips, per_rep=per)
            rows.append(r)
            print(f"{cap:>8.2f}{viol:>7d}/{n:<3d}{r['margin']:>9.3f}"
                  f"{r['tip_mm']:>9.2f}{flips:>8d}")
        out[side] = rows
        if rows:
            best = [r for r in rows if r["contact"] == min(x["contact"] for x in rows)]
            lo, hi = min(r["ext_cap"] for r in best), max(r["ext_cap"] for r in best)
            print(f"  lowest contact ({best[0]['contact']}/{best[0]['n']}) over "
                  f"ceilings {lo:.2f} to {hi:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(captures=args.captures, sides=args.sides, sweep=out),
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
