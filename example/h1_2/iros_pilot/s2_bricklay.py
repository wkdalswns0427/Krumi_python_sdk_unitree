#!/usr/bin/env python3
"""s2_bricklay.py - two-handed bricklaying pilot: native vs anthropomorphic.

The TASK is a real two-handed pick-and-stack, defined by its goal, not by a
human capture: grasp a block from a low central pile with BOTH hands, carry
it, and set it on a growing column of K blocks. Both hands hold the one block
at a fixed grasp width (a rigid two-hand grasp), so the block CENTER is the
thing that must reach each stack level within tol with a dwell. Both motions
accomplish the identical task; only the STRATEGY differs.

  ROBOT-NATIVE  : direct minimum-jerk joint moves pile->stack_i, fast, no
                  wasted height. Crisp and efficient.
  ANTHROPOMORPHIC: the way a person does it - lift the block high, carry it
                  across at height, lower onto the column, and relax the arms
                  between blocks; slower, with dwells. This is a model, not a
                  capture; its speed and dwell are set from the observed M2
                  human motion, and the up-over-down carry is the
                  characteristic human transport. The retargeting + hardware
                  themselves are validated separately on the real M2/M3
                  captures (Blocks 1-2); this pilot studies STRATEGY.

Two-handed => both hands must grasp symmetric points about the block center,
so the center is pinned near the midline (each hand stays in its own reachable
envelope, z in [-0.05,0.15], right y in [-0.28,-0.04], left mirrored). That is
too narrow for a wide lateral row, so the task is a vertical STACK: real
~19 cm lift, clear up-and-place gesture, both arms working. All grasp points
are IK-checked for BOTH arms. Emits two trajectory CSVs for replay_arm.py + a
task spec for s2_compare.py (arm="both", scored on the block center).

Usage:
    /usr/bin/python3 s2_bricklay.py                 # 3 blocks
    /usr/bin/python3 s2_bricklay.py --blocks 4 --grasp-hw 0.10
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments", "scripts")
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.expanduser("~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments"))
from retarget_arm import ArmChainFK, load_limits, OUT_JOINTS   # noqa: E402

RJSEG = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")

# measured robot rest pose (snapshot from the trial sessions); left/right arm
HOME_L = [0.31, 0.24, -0.12, 0.67]
HOME_R = [0.30, -0.24, 0.12, 0.69]
RATE = 50.0


def minjerk(q0, q1, T, rate):
    """Point-to-point minimum-jerk (10s^3-15s^4+6s^5) at `rate` Hz."""
    n = max(2, int(round(T * rate)))
    out = []
    for i in range(1, n + 1):
        s = i / n
        a = 10 * s**3 - 15 * s**4 + 6 * s**5
        out.append([x0 + a * (x1 - x0) for x0, x1 in zip(q0, q1)])
    return out

# task geometry (torso frame, m): central low pile, blocks stacked in a
# rising column. block CENTER positions; each hand grasps center
# +/- [0, grasp_hw, 0]. PHYSICAL-BLOCK layout (9 3/8 x 4 3/4 in): pile and
# layer 1 share a support surface (same center z), layers are exactly one
# block height apart, and the grasp width equals the block width minus a
# small squeeze. Stack x=0.30 / z0=-0.02 chosen from the reachability sweep
# (top layer z0 + 2h = +0.22 reaches at 0.1 mm on the natural branch; the
# elbow-up mirror branch CANNOT reach the upper layers at this narrow grasp
# - straight arm leaves no yaw freedom - so the real-block task runs on the
# natural branch and the ergonomic inversion rides on the crouched stance).
PICK = np.array([0.26, 0.00, -0.02])           # pile (same surface as layer 1)
STACK_X = 0.30
STACK_Z0 = -0.02
# anthropomorphic "carry close to the body": bring the block IN toward the
# chest (x=CHEST_X) then push it OUT to the wall. This real load-management
# behavior makes the human path a big in/out/up loop that the direct native
# motion skips, so the two strategies look different in KIND, not just speed.
CHEST_X = 0.25          # was 0.22; at the real-block narrow grasp the chest
CHEST_Z = 0.13          # carry pulls elbows in - 0.25 keeps them clear
# legacy scaled stations (pre-physical-block): the Block_S2 human takes were
# validity-gated against THESE; kept in the scaled spec for the baseline.
SCALED_STATIONS = [[0.26, 0.0, -0.05], [0.34, 0.0, 0.0],
                   [0.34, 0.0, 0.07], [0.34, 0.0, 0.14]]


class ArmIK:
    """4-DOF wrist-position IK for one arm (damped GN, yaw anchored).

    yaw_anchor selects the IK BRANCH the redundant shoulder yaw settles on:
    0 = the natural human branch; +/-2.5 rad = the mirror "elbow-up" branch
    (same wrist position, elbows high and out - a configuration a human
    cannot sustain but the robot holds for free)."""
    def __init__(self, urdf, side, yaw_anchor=0.0, yaw_w=0.02):
        self.fk = ArmChainFK(urdf, side)
        self.lim = [load_limits(urdf)[f"{side}_{s}_joint"] for s in RJSEG]
        self.anchor = yaw_anchor
        self.w = yaw_w

    def solve(self, target, q0):
        q = np.array(q0, float)
        lo = np.array([l for l, _ in self.lim])
        hi = np.array([h for _, h in self.lim])
        for _ in range(300):
            tip = self.fk.positions(q)[2]
            r = np.asarray(target) - tip
            if np.linalg.norm(r) < 1e-4:
                break
            J = np.zeros((3, 4))
            for k in range(4):
                qp = q.copy()
                qp[k] += 1e-4
                J[:, k] = (self.fk.positions(qp)[2] - tip) / 1e-4
            H = J.T @ J + 1e-2 * np.eye(4)
            g = J.T @ r
            H[2, 2] += self.w                    # anchor yaw = branch select
            g[2] += self.w * (self.anchor - q[2])
            q = np.clip(q + np.linalg.solve(H, g), lo, hi)
        res = float(np.linalg.norm(np.asarray(target) - self.fk.positions(q)[2]))
        return q, res

    def ee(self, q):
        return self.fk.positions(q)[2]


class Grasp:
    """Both arms grasping one block; a pose is the 8-vector [qL(4), qR(4)].

    config "natural" = human-branch arms; "elbow-up" = mirror branch, elbows
    high above the stack: kills the elbow-torso graze class entirely and
    keeps the forearms out of the clutter plane (pile + wall courses)."""
    def __init__(self, urdf, hw, config="natural"):
        if config == "elbow-up":
            self.L = ArmIK(urdf, "left", yaw_anchor=+2.5, yaw_w=0.05)
            self.R = ArmIK(urdf, "right", yaw_anchor=-2.5, yaw_w=0.05)
        else:
            self.L = ArmIK(urdf, "left")
            self.R = ArmIK(urdf, "right")
        self.config = config
        self.hw = hw

    def start(self, side):
        """Warm start on the requested branch."""
        base = HOME_L if side == "L" else HOME_R
        if self.config == "elbow-up":
            q = list(base)
            q[2] = self.L.anchor if side == "L" else self.R.anchor
            return q
        return list(base)

    def solve(self, center, prev=None):
        c = np.asarray(center, float)
        qL0 = self.start("L") if prev is None else prev[0:4]
        qR0 = self.start("R") if prev is None else prev[4:8]
        qL, rL = self.L.solve(c + [0, self.hw, 0], qL0)
        qR, rR = self.R.solve(c + [0, -self.hw, 0], qR0)
        return np.concatenate([qL, qR]), max(rL, rR)

    def center(self, pose):
        return 0.5 * (self.L.ee(pose[0:4]) + self.R.ee(pose[4:8]))


def write_traj(path, poses):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + OUT_JOINTS)
        for i, pose in enumerate(poses):
            full = np.zeros(15)
            full[0:4] = pose[0:4]                # left arm
            full[7:11] = pose[4:8]               # right arm
            w.writerow([f"{i / RATE:.4f}"] + [f"{v:.6f}" for v in full])


def path_time(grasp, poses):
    C = np.array([grasp.center(p) for p in poses])
    return float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))), len(poses) / RATE


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"))
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--block-width", type=float, default=0.238,
                   help="physical block width, m (9 3/8 in = 0.238)")
    p.add_argument("--block-height", type=float, default=0.121,
                   help="physical block height, m (4 3/4 in = 0.121); sets "
                        "the stack layer spacing")
    p.add_argument("--squeeze", type=float, default=0.008,
                   help="total grip squeeze, m: hands commanded this much "
                        "narrower than the block for holding force")
    p.add_argument("--palm-offset", type=float, default=0.05,
                   help="wrist-center to palm-surface offset, m per hand: the "
                        "trajectory positions the WRIST joints, but the palms "
                        "sit inboard of them, so the commanded wrist width = "
                        "block width + 2*palm_offset - squeeze. Calibrate at "
                        "the robot if the grip is loose or tight.")
    p.add_argument("--grasp-hw", type=float, default=None,
                   help="override half WRIST separation, m (default: computed "
                        "from --block-width, --palm-offset and --squeeze)")
    p.add_argument("--native-vmax", type=float, default=0.6,
                   help="native peak joint vel; default MATCHES --human-vmax so "
                        "the time win equals the path win (pure strategy, not "
                        "raw speed). Raise (e.g. 1.0) to also show robot speed.")
    p.add_argument("--native-config", choices=["natural", "elbow-up"],
                   default="natural",
                   help="native arm branch. elbow-up (mirror branch, elbows "
                        "high) CANNOT reach the upper layers of the real-"
                        "block stack (straight arm = no yaw freedom); it "
                        "remains available for reduced/abstract geometries. "
                        "The ergonomic inversion for the real-block task is "
                        "carried by the crouched stance (REBA legs).")
    p.add_argument("--human-vmax", type=float, default=0.6,
                   help="anthropomorphic peak joint vel (slower, from M2)")
    p.add_argument("--tol", type=float, default=0.08,
                   help="station radius, m; 0.08 = quarter block width, "
                        "accommodates transient real-human placements")
    p.add_argument("--out-dir", default=HERE)
    args = p.parse_args()
    hw = args.grasp_hw or ((args.block_width - args.squeeze) / 2.0
                           + args.palm_offset)
    stack = [np.array([STACK_X, 0.0, STACK_Z0 + i * args.block_height])
             for i in range(args.blocks)]

    gH = Grasp(args.urdf, hw, "natural")                  # human model branch
    gN = Grasp(args.urdf, hw, args.native_config)
    home = np.concatenate([HOME_L, HOME_R])
    # human-model poses (natural branch)
    ppick, rp = gH.solve(PICK)
    pchest, _ = gH.solve([CHEST_X, PICK[1], CHEST_Z], ppick)  # carry in to body
    pplace, pover = [], []
    # native poses (task branch; same hand targets)
    npick, nrp = gN.solve(PICK)
    nplace = []
    for s in stack:
        ps, r = gH.solve(s, ppick)
        oz = min(0.26, s[2] + 0.03)              # clear the block below
        po, _ = gH.solve([s[0], s[1], oz], ps)
        ns, nr = gN.solve(s, npick)
        if max(rp, r, nrp, nr) > 0.02:
            sys.exit(f"IK residual too high ({max(rp,r,nrp,nr)*1000:.0f} mm) at {s}")
        pplace.append(ps)
        pover.append(po)
        nplace.append(ns)
    print(f"[bricklay] two-handed, {args.blocks} blocks; grasp width "
          f"{2*hw*100:.1f} cm (block {args.block_width*100:.1f} - squeeze); "
          f"pick z={PICK[2]:+.2f}, layers "
          f"{'/'.join(f'{s[2]:+.2f}' for s in stack)} "
          f"(block height {args.block_height*100:.1f} cm, "
          f"lift {(stack[-1][2]-PICK[2])*100:.0f} cm)")
    print(f"[bricklay] native config: {args.native_config}", end="")
    if args.native_config == "elbow-up":
        elz = [gN.L.fk.positions(p[0:4])[1][2] for p in [npick] + nplace]
        print(f"  (elbow z {min(elz):+.2f}..{max(elz):+.2f} m, above the "
              f"stack top {stack[-1][2]:+.2f}; same hand targets as natural)")
    else:
        print()

    def seg_T(p0, p1, vmax):
        dq = float(np.abs(np.asarray(p0) - np.asarray(p1)).max())
        return max(0.35, 1.875 * dq / vmax)

    def build(waypts, vmax, dwell):
        rows = [np.asarray(waypts[0][1])]
        for i in range(len(waypts) - 1):
            (_, q0), (nm, q1) = waypts[i], waypts[i + 1]
            rows += [np.asarray(q) for q in minjerk(q0, q1, seg_T(q0, q1, vmax), RATE)]
            if nm in ("pick", "place", "reset"):     # reset = pause at rest
                rows += [np.asarray(q1)] * int(dwell * RATE)
        return rows

    # NATIVE: home -> (pick, stack_i) x blocks -> home. Direct, fast, low dwell.
    # elbow-up: switch branches with the hands raised FORWARD (via pair), so
    # the elbow arcs around the front of the body, never beside the torso.
    nat = [("home", home)]
    if args.native_config == "elbow-up":
        READY = [0.28, 0.0, 0.20]
        vN, _ = gH.solve(READY)
        vU, _ = gN.solve(READY)
        nat += [("via", vN), ("via", vU)]
    nat.append(("pick", npick))
    for i, ps in enumerate(nplace):
        nat.append(("place", ps))
        if i < len(nplace) - 1:
            nat.append(("pick", npick))
    if args.native_config == "elbow-up":
        nat += [("via", vU), ("via", vN)]
    nat.append(("home", home))
    nrows = build(nat, args.native_vmax, 0.3)

    # ANTHROPOMORPHIC: per block [pick, carry-IN-to-chest, OUT-over-column,
    # place, lift, relax-home]. The block is pulled in toward the body, carried
    # up, pushed out over the column, lowered, then the arms reset. A big
    # in/out/up loop + rest between, slow. The direct native skips all of it.
    hum = [("home", home)]
    for i, ps in enumerate(pplace):
        last = i == len(pplace) - 1
        hum += [("pick", ppick), ("via", pchest), ("via", pover[i]),
                ("place", ps), ("via", pover[i]),
                ("home" if last else "reset", home)]   # pause at rest between
    hrows = build(hum, args.human_vmax, 0.8)

    npm, nt = path_time(gN, nrows)
    hp, ht = path_time(gH, hrows)
    natf = os.path.join(args.out_dir, "s2_bricklay_native.csv")
    humf = os.path.join(args.out_dir, "s2_bricklay_human.csv")
    write_traj(natf, nrows)
    write_traj(humf, hrows)
    print(f"[bricklay] NATIVE  {os.path.basename(natf)}: {nt:5.1f}s, {npm:.2f}m center-path")
    print(f"[bricklay] HUMAN   {os.path.basename(humf)}: {ht:5.1f}s, {hp:.2f}m center-path")
    print(f"[bricklay] ADVANTAGE  time {ht/nt:.1f}x   path {hp/npm:.1f}x  "
          f"(both stack {args.blocks} blocks, two-handed)")

    labels = ["pick"] + [f"block{i+1}" for i in range(len(stack))]
    spec = dict(arm="both", grasp_hw=hw,
                block_width=args.block_width, block_height=args.block_height,
                native_config=args.native_config,
                stations=[list(map(float, PICK))] + [list(map(float, s)) for s in stack],
                labels=labels,
                tol_m=args.tol, dwell_s=0.15, blocks=args.blocks,
                native_traj=os.path.abspath(natf),
                human_traj=os.path.abspath(humf),
                note="PHYSICAL-block task (real block dims); block center = "
                     "midpoint of the two hands. The retargeted human baseline "
                     "places at ~0.6-scaled stations (embodiment scale) - gate "
                     "it against the SCALED spec and report the mismatch as a "
                     "finding, not a failure of the capture.")
    specf = os.path.join(args.out_dir, "s2_bricklay_spec.json")
    with open(specf, "w") as f:
        json.dump(spec, f, indent=2)
    sspec = dict(spec, stations=SCALED_STATIONS,
                 note="SCALED stations (legacy abstract geometry): what the "
                      "retargeted human baseline actually reaches; the "
                      "Block_S2 takes were validity-gated against these.")
    sspecf = os.path.join(args.out_dir, "s2_bricklay_spec_scaled.json")
    with open(sspecf, "w") as f:
        json.dump(sspec, f, indent=2)
    print(f"[bricklay] wrote {os.path.basename(specf)} + {os.path.basename(sspecf)}")


if __name__ == "__main__":
    main()
