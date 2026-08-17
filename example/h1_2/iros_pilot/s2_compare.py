#!/usr/bin/env python3
"""s2_compare.py - score the S2 pilot: human-mimic vs robot-native reach.

Same task, two motions, one metric set. The task (s2_task_spec.json from
s2_reach_native.py): the right end-effector must visit zone A then zone B
(balls of radius tol around the human-derived pick/place centers), dwelling
>= dwell_s each, for N cycles. Both conditions run through the identical
replay + logging stack, so anything that differs is the MOTION.

Reports per source: task PASS/FAIL (the validity gate - the human-mimic
baseline must pass, else the efficiency win is attributable to a broken
baseline, not to robot-native motion), completion time, right-EE path
length, endpoint accuracy (closest approach to each zone center), and, when
the log carries exetau_/exedq_ columns, MEASURED arm mechanical work
(sum |tau_est * dq| dt over the 8 arm joints).

A source is either a replay log (executed joints, real) or a trajectory CSV
(commanded joints, predicted, no energy). Give several with --src label:path.

Usage:
    # baseline from the real 45-trial logs, native from its trajectory:
    /usr/bin/python3 s2_compare.py --spec s2_task_spec.json \
        --src human:$IROS/block2/replay_M2_R1_T1.csv \
        --src native:s2_native_traj.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments"))
from h12_experiments.fk_h12 import H12ArmFK          # noqa: E402
from h12_experiments.joints import ARM_CHAIN         # noqa: E402

RJ = ARM_CHAIN["right"][:4]                          # 4 actuated shoulder+elbow
LJ = ARM_CHAIN["left"][:4]
# One-handed S2 scores energy on the RIGHT arm only (the left-arm human
# activity is task-unnecessary for a one-handed reach). A two-handed task
# (spec arm=="both") scores BOTH arms: both do task-necessary work and the
# block center is the midpoint of the two hands.
ENERGY_JOINTS = ARM_CHAIN["right"]                   # one-handed default
ENERGY_JOINTS_BOTH = ARM_CHAIN["left"] + ARM_CHAIN["right"]


def read_source(path):
    """-> dict with t, per-arm joints (qR, qL if present), and per-frame arm
    dq/tau if a log. Auto-detects: 't_s' header = trajectory (commanded)."""
    rows = list(csv.DictReader(open(path)))
    cols = set(rows[0].keys())
    if "t_s" in cols:                                # trajectory CSV
        t = np.array([float(x["t_s"]) for x in rows])
        qR = np.array([[float(x[j]) for j in RJ] for x in rows])
        qL = (np.array([[float(x[j]) for j in LJ] for x in rows])
              if all(j in cols for j in LJ) else None)
        return dict(t=t, qR=qR, qL=qL, dq=None, tau=None, dqb=None, taub=None,
                    kind="commanded")
    # replay log: use executed joints, drop unpaired rows
    def c(n):
        return np.array([float(x[n]) if x.get(n, "nan") not in ("", "nan")
                         else np.nan for x in rows])
    t = c("t_rel")
    m = np.ones(len(rows), bool)
    for j in RJ:
        m &= np.isfinite(c(f"exe_{j}"))
    qR = np.stack([c(f"exe_{j}")[m] for j in RJ], axis=1)
    qL = (np.stack([c(f"exe_{j}")[m] for j in LJ], axis=1)
          if all(f"exe_{j}" in cols for j in LJ) else None)
    t = t[m]
    def energy(joints):
        if not (f"exedq_{joints[0]}" in cols and f"exetau_{joints[0]}" in cols):
            return None, None
        return (np.stack([c(f"exedq_{j}")[m] for j in joints], axis=1),
                np.stack([c(f"exetau_{j}")[m] for j in joints], axis=1))
    dq, tau = energy(ENERGY_JOINTS)
    dqb, taub = energy(ENERGY_JOINTS_BOTH)
    return dict(t=t, qR=qR, qL=qL, dq=dq, tau=tau, dqb=dqb, taub=taub,
                kind="executed")


def ee_path(urdf, q, side):
    fk = H12ArmFK(urdf, side)
    return np.array([fk.ee_position(list(qi) + [0, 0, 0]) for qi in q])


def score(src, spec, urdf):
    t, qR = src["t"], src["qR"]
    two_hand = spec.get("arm") == "both" and src.get("qL") is not None
    if two_hand:
        P = 0.5 * (ee_path(urdf, qR, "right") + ee_path(urdf, src["qL"], "left"))
        dq, tau = src.get("dqb"), src.get("taub")   # both-arm energy
    else:
        P = ee_path(urdf, qR, "right")
        dq, tau = src["dq"], src["tau"]
    # station list (new "row" spec) or the legacy A/B two-point spec
    if "stations" in spec:
        stations = [np.array(s) for s in spec["stations"]]
        labels = spec.get("labels", [f"s{i}" for i in range(len(stations))])
        need = 1                                          # each visited >= once
    else:
        stations = [np.array(spec["A"]), np.array(spec["B"])]
        labels = ["A", "B"]
        need = spec.get("cycles", 1)
    tol = spec["tol_m"]
    dwell_n = int(spec["dwell_s"] * 50)

    def sustained_visits(d):
        inz = d <= tol
        runs, run = 0, 0
        for v in inz:
            run = run + 1 if v else 0
            if run == dwell_n:
                runs += 1
        return runs, float(d.min())

    dists = [np.linalg.norm(P - s, axis=1) for s in stations]
    visits = [sustained_visits(d) for d in dists]
    hits = [v[0] for v in visits]
    closest = [v[1] for v in visits]
    inzone = np.any([d <= tol for d in dists], axis=0)
    if inzone.any():
        i0, i1 = np.where(inzone)[0][[0, -1]]
    else:
        i0, i1 = 0, len(t) - 1
    active_t = float(t[i1] - t[i0])
    path = float(np.sum(np.linalg.norm(np.diff(P[i0:i1 + 1], axis=0), axis=1)))
    passed = all(h >= need for h in hits)

    energy = None
    if dq is not None:
        tt = t[i0:i1 + 1]
        power = np.sum(np.abs(tau[i0:i1 + 1] * dq[i0:i1 + 1]), axis=1)   # W
        energy = float(np.trapz(power, tt))
    return dict(kind=src["kind"], passed=passed, hits=hits, need=need,
                labels=labels, closest=closest, time=active_t, path=path,
                energy=energy)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--src", action="append", required=True,
                   help="label:path (replay log or trajectory CSV), repeatable")
    p.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"))
    args = p.parse_args()
    spec = json.load(open(args.spec))

    nstations = len(spec.get("stations", [spec.get("A"), spec.get("B")]))
    armtxt = ("both arms (block center)" if spec.get("arm") == "both"
              else "right arm")
    print("=" * 74)
    print(f"S2 pilot comparison   task: visit {nstations} stations "
          f"(tol {spec['tol_m']*100:.0f} cm, dwell {spec['dwell_s']}s), {armtxt}")
    if "stations" in spec:
        for lab, s in zip(spec.get("labels", []), spec["stations"]):
            print(f"    {lab:8} {[round(v,3) for v in s]}")
    print("=" * 74)
    print(f"  {'source':10}{'kind':10}{'task':>7}{'zone hits':>18}"
          f"{'time s':>9}{'path m':>9}{'energy J':>11}")
    res = {}
    for spec_src in args.src:
        label, path = spec_src.split(":", 1)
        r = score(read_source(path), spec, args.urdf)
        res[label] = r
        e = f"{r['energy']:.1f}" if r["energy"] is not None else "-"
        hitstr = "/".join(str(h) for h in r["hits"]) + f" (>={r['need']})"
        print(f"  {label:10}{r['kind']:10}"
              f"{'PASS' if r['passed'] else 'FAIL':>7}{hitstr:>18}"
              f"{r['time']:>9.1f}{r['path']:>9.2f}{e:>11}")
    if "human" in res and "native" in res:
        h, ntv = res["human"], res["native"]
        print("-" * 74)
        print(f"  ADVANTAGE (human / native):  time {h['time']/ntv['time']:.1f}x"
              f"   path {h['path']/ntv['path']:.1f}x", end="")
        if h["energy"] and ntv["energy"]:
            print(f"   energy {h['energy']/ntv['energy']:.1f}x")
        else:
            print("   energy: needs native on hardware (exetau/exedq)")
        if not h["passed"]:
            print("  ** BASELINE FAILED the task - raise --tol in "
                  "s2_reach_native and rebuild before trusting the advantage **")
    print("=" * 74)


if __name__ == "__main__":
    main()
