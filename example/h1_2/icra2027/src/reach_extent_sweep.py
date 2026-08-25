#!/usr/bin/env python3
"""Reach-extent sweep: does the fidelity limit-violation rate rise with reach?

Plan quote: "C3 currently rests on 2 of 6 captures, which a reviewer will
call thin. The reach-extent sweep turns 'fidelity sometimes violates joint
limits' into 'violation rate rises with reach extent, and here is the curve,'
which is a mechanism rather than an anecdote."

STATUS: skeleton only. Consumes the reach-extent sweep captures (planned for
the same refilm session as M1/M2/M3, forward reach at 4-5 target distances,
3 reps each). Requires those captures before it can run for real.

For each capture:
  - reach_extent = distance from shoulder to wrist in the torso frame,
    averaged over the frame window that contains the maximum reach.
  - fidelity limit-violation frames per rep
  - task-centric limit-violation frames per rep (should stay 0 across the
    sweep; if not, the argument weakens)
Report: (reach_extent, viol_rate_fid) as a curve, one point per rep.
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-root", required=False,
                   help="parent dir containing REACH_<d>_<r>/ per-target/per-rep"
                        " captures; not yet available")
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "reach_extent", "sweep.json"))
    args = p.parse_args()
    print("[reach-extent] STATUS: skeleton only. Refilm session is a"
          " prerequisite.")
    print("[reach-extent] Once captures exist, iterate over per-target/per-rep"
          " folders using the same reader as x6/x2, then emit a JSON with the"
          " (extent_m, violations, margin_mean) tuple per rep.")


if __name__ == "__main__":
    main()
