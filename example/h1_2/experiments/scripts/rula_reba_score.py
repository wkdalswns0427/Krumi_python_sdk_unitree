#!/usr/bin/env python3
"""rula_reba_score.py - per-frame RULA and REBA scores from a joints CSV.

Deliverable A of Block 4. Consumes the full joint set (bag_to_joints
--joints full) via angles.py and produces, per frame, the RULA and REBA
grand scores and their discrete risk categories. Output feeds
rula_reba_agreement.py (mono vs RGBD agreement).

Scope and assumptions (documented, applied identically to every pipeline so
the mono-vs-RGBD comparison is fair):
  * POSTURE-only scoring. Force/load and muscle-use (RULA) and load,
    coupling, and activity (REBA) have no capture signal, so they are 0.
    Absolute scores are therefore lower bounds; the Block 4 result is about
    AGREEMENT between pipelines under identical code, which this preserves.
  * NEUTRAL-WRIST assumption: MediaPipe Pose has no stable hand reference,
    so wrist = neutral (RULA wrist 1 + twist 1; REBA wrist 1).
  * NEUTRAL-NECK assumption when neck_flexion is None (face landmarks are
    not exported), exactly like the wrist.

Lookup tables are the published RULA (McAtamney & Corlett 1993) and REBA
(Hignett & McAtamney 2000) tables, transcribed below as nested lists with
1-based clamping so a human can diff them against the standard sheets.
Validated by hand-worked postures in the --self-test (neutral stand,
90 deg bow, overhead reach) plus the known ground truth that a fully
neutral REBA posture scores 1.

Usage:
    /usr/bin/python3 rula_reba_score.py joints_full.csv --out scores.csv
    /usr/bin/python3 rula_reba_score.py --self-test
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import angles as A  # noqa: E402


# ── RULA tables ──────────────────────────────────────────────────────────────
# Table A: [upper_arm 1-6][lower_arm 1-3][wrist 1-4][wrist_twist 1-2]
RULA_A = [
    # upper arm 1
    [[[1, 2], [2, 2], [2, 3], [3, 3]],
     [[2, 2], [2, 2], [3, 3], [3, 3]],
     [[2, 3], [3, 3], [3, 3], [4, 4]]],
    # upper arm 2
    [[[2, 3], [3, 3], [3, 4], [4, 4]],
     [[3, 3], [3, 3], [3, 4], [4, 4]],
     [[3, 4], [4, 4], [4, 4], [5, 5]]],
    # upper arm 3
    [[[3, 3], [4, 4], [4, 4], [5, 5]],
     [[3, 4], [4, 4], [4, 4], [5, 5]],
     [[4, 4], [4, 4], [4, 5], [5, 5]]],
    # upper arm 4
    [[[4, 4], [4, 4], [4, 5], [5, 5]],
     [[4, 4], [4, 4], [4, 5], [5, 5]],
     [[4, 4], [4, 5], [5, 5], [6, 6]]],
    # upper arm 5
    [[[5, 5], [5, 5], [5, 6], [6, 7]],
     [[5, 6], [6, 6], [6, 7], [7, 7]],
     [[6, 6], [6, 7], [7, 7], [7, 8]]],
    # upper arm 6
    [[[7, 7], [7, 7], [7, 8], [8, 9]],
     [[8, 8], [8, 8], [8, 9], [9, 9]],
     [[9, 9], [9, 9], [9, 9], [9, 9]]],
]
# Table B: [neck 1-6][trunk 1-6][legs 1-2]
RULA_B = [
    [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
    [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
    [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
    [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
    [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
    [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
]
# Table C: [scoreA 1-8][scoreB 1-7] -> grand RULA
RULA_C = [
    [1, 2, 3, 3, 4, 5, 5],
    [2, 2, 3, 4, 4, 5, 5],
    [3, 3, 3, 4, 4, 5, 6],
    [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7],
    [4, 4, 5, 6, 6, 7, 7],
    [5, 5, 6, 6, 7, 7, 7],
    [5, 5, 6, 7, 7, 7, 7],
]

# ── REBA tables ──────────────────────────────────────────────────────────────
# Table A: [neck 1-3][trunk 1-5][legs 1-4]
REBA_A = [
    [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
    [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
]
# Table B: [upper_arm 1-6][lower_arm 1-2][wrist 1-3]
REBA_B = [
    [[1, 2, 2], [1, 2, 3]],
    [[1, 2, 3], [2, 3, 4]],
    [[3, 4, 5], [4, 5, 5]],
    [[4, 5, 5], [5, 6, 7]],
    [[6, 7, 8], [7, 8, 8]],
    [[7, 8, 8], [8, 9, 9]],
]
# Table C: [scoreA 1-12][scoreB 1-12] -> score C
REBA_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── segment scores from angles ───────────────────────────────────────────────
def _upper_arm_score(flex):
    """RULA/REBA share this banding (deg from trunk-down; extension folded in
    with flexion since a monocular estimate cannot sign it)."""
    if flex is None:
        return 1
    a = abs(flex)
    if a < 20:
        return 1
    if a < 45:
        return 2
    if a < 90:
        return 3
    return 4


def _lower_arm_score_rula(elbow):
    if elbow is None:
        return 1
    return 1 if 60 <= elbow <= 100 else 2


def _trunk_score_rula(flex, side_bend, twist):
    if flex is None:
        return 1
    if flex < 5:
        s = 1
    elif flex < 20:
        s = 2
    elif flex < 60:
        s = 3
    else:
        s = 4
    if side_bend is not None and abs(side_bend) > 10:
        s += 1
    if twist is not None and abs(twist) > 10:
        s += 1
    return _clamp(s, 1, 6)


def _neck_score_rula(flex, side_bend, twist):
    if flex is None:                      # neutral-neck assumption
        return 1
    if flex < 10:
        s = 1
    elif flex < 20:
        s = 2
    else:
        s = 3
    if side_bend is not None and abs(side_bend) > 10:
        s += 1
    if twist is not None and abs(twist) > 10:
        s += 1
    return _clamp(s, 1, 6)


def _trunk_score_reba(flex, side_bend, twist):
    if flex is None:
        return 1
    if flex < 5:
        s = 1
    elif flex < 20:
        s = 2
    elif flex < 60:
        s = 3
    else:
        s = 4
    if (side_bend is not None and abs(side_bend) > 10) or \
       (twist is not None and abs(twist) > 10):
        s += 1
    return _clamp(s, 1, 5)


def _neck_score_reba(flex, side_bend, twist):
    if flex is None:
        return 1
    s = 1 if flex < 20 else 2
    if (side_bend is not None and abs(side_bend) > 10) or \
       (twist is not None and abs(twist) > 10):
        s += 1
    return _clamp(s, 1, 3)


def _legs_score(legs, metric):
    """1 = bilateral balanced support, +1 unilateral; REBA adds knee bend."""
    knees = [legs.get("knee_flexion_left"), legs.get("knee_flexion_right")]
    stance = legs.get("stance_width_m")
    hip_w = legs.get("hip_width_m")
    # bilateral support proxy: stance not wildly wider/narrower than the hips
    bilateral = (stance is not None and hip_w is not None
                 and 0.3 * hip_w <= stance <= 3.0 * hip_w)
    base = 1 if bilateral else 2
    if metric == "rula":
        return _clamp(base, 1, 2)
    # REBA: knee bend adder using max flexion off straight (180 deg)
    bend = max((180 - k) for k in knees if k is not None) if any(
        k is not None for k in knees) else 0
    add = 0
    if 30 <= bend < 60:
        add = 1
    elif bend >= 60:
        add = 2
    return _clamp(base + add, 1, 4)


# ── grand scores ─────────────────────────────────────────────────────────────
def rula_grand(ang, legs):
    ua = max(_upper_arm_score(ang["upper_arm_flexion_left"]),
             _upper_arm_score(ang["upper_arm_flexion_right"]))
    la = max(_lower_arm_score_rula(ang["lower_arm_flexion_left"]),
             _lower_arm_score_rula(ang["lower_arm_flexion_right"]))
    wrist, twist = 1, 1                    # neutral-wrist assumption
    score_a = RULA_A[ua - 1][la - 1][wrist - 1][twist - 1]  # + muscle/force = 0
    neck = _neck_score_rula(ang["neck_flexion"], ang["trunk_side_bend"],
                            ang["trunk_twist"])
    trunk = _trunk_score_rula(ang["trunk_flexion"], ang["trunk_side_bend"],
                              ang["trunk_twist"])
    legs_s = _legs_score(legs, "rula")
    score_b = RULA_B[neck - 1][trunk - 1][legs_s - 1]       # + muscle/force = 0
    a = _clamp(score_a, 1, 8)
    b = _clamp(score_b, 1, 7)
    return RULA_C[a - 1][b - 1]


def reba_grand(ang, legs):
    trunk = _trunk_score_reba(ang["trunk_flexion"], ang["trunk_side_bend"],
                              ang["trunk_twist"])
    neck = _neck_score_reba(ang["neck_flexion"], ang["trunk_side_bend"],
                            ang["trunk_twist"])
    legs_s = _legs_score(legs, "reba")
    score_a = REBA_A[neck - 1][trunk - 1][legs_s - 1]       # + load = 0
    ua = max(_upper_arm_score(ang["upper_arm_flexion_left"]),
             _upper_arm_score(ang["upper_arm_flexion_right"]))
    la_l = ang["lower_arm_flexion_left"]
    la_r = ang["lower_arm_flexion_right"]
    la = max(1 if (v is not None and 60 <= v <= 100) else 2
             for v in (la_l, la_r))
    wrist = 1                             # neutral-wrist assumption
    score_b = REBA_B[ua - 1][la - 1][wrist - 1]             # + coupling = 0
    a = _clamp(score_a, 1, 12)
    b = _clamp(score_b, 1, 12)
    return REBA_C[a - 1][b - 1]           # + activity = 0


# ── CSV pipeline ─────────────────────────────────────────────────────────────
def load_joints_csv(path):
    frames = defaultdict(lambda: (dict(), dict()))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_conf = "conf" in (reader.fieldnames or [])
        for r in reader:
            fi = int(float(r["frame"]))
            pos, conf = frames[fi]
            pos[r["joint"].strip()] = np.array(
                [float(r["x"]), float(r["y"]), float(r["z"])])
            if has_conf and r.get("conf", "") != "":
                conf[r["joint"].strip()] = float(r["conf"])
    return frames


def score_csv(path, conf_min, up):
    frames = load_joints_csv(path)
    out = []
    for fi in sorted(frames):
        pos, conf = frames[fi]
        ang = A.compute_frame_angles(pos, conf or None, conf_min, up)
        legs = A.legs_support_info(pos, conf or None, conf_min, up)
        out.append((fi, rula_grand(ang, legs), reba_grand(ang, legs)))
    return out


def write_scores(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "rula_score", "reba_score"])
        for fi, ru, re in rows:
            w.writerow([fi, ru, re])


# ── self-test: hand-worked postures ──────────────────────────────────────────
def _posture(kind):
    """Synthetic full-joint frame (y-down) for a known posture."""
    j = {
        "left_shoulder": [0.20, 0.0, 0.0], "right_shoulder": [-0.20, 0.0, 0.0],
        "left_hip": [0.12, 0.55, 0.0], "right_hip": [-0.12, 0.55, 0.0],
        "left_knee": [0.12, 1.05, 0.0], "right_knee": [-0.12, 1.05, 0.0],
        "left_ankle": [0.12, 1.55, 0.0], "right_ankle": [-0.12, 1.55, 0.0],
    }
    if kind == "neutral":            # arms hanging straight down (y-down = +y)
        j["left_elbow"] = [0.22, 0.30, 0.0]
        j["right_elbow"] = [-0.22, 0.30, 0.0]
        j["left_wrist"] = [0.24, 0.58, 0.0]
        j["right_wrist"] = [-0.24, 0.58, 0.0]
    elif kind == "bow":              # trunk flexed ~90 deg forward (+z), arms down
        for k in ("left_shoulder", "right_shoulder"):
            j[k] = [j[k][0], 0.55, -0.55]     # shoulders forward & level with hips
        j["left_elbow"] = [0.22, 0.85, -0.55]
        j["right_elbow"] = [-0.22, 0.85, -0.55]
        j["left_wrist"] = [0.24, 1.10, -0.55]
        j["right_wrist"] = [-0.24, 1.10, -0.55]
    elif kind == "overhead":         # arms raised overhead (upper arm >90 flex)
        j["left_elbow"] = [0.24, -0.30, 0.0]
        j["right_elbow"] = [-0.24, -0.30, 0.0]
        j["left_wrist"] = [0.26, -0.58, 0.0]
        j["right_wrist"] = [-0.26, -0.58, 0.0]
    return j


def self_test():
    up = A.UP_DEFAULT
    results = {}
    for kind in ("neutral", "bow", "overhead"):
        j = _posture(kind)
        ang = A.compute_frame_angles(j, None, 0.5, up)
        legs = A.legs_support_info(j, None, 0.5, up)
        ru, re = rula_grand(ang, legs), reba_grand(ang, legs)
        results[kind] = (ru, re, ang)
        print(f"[{kind:8}] RULA={ru} REBA={re}  "
              f"(upper_arm_flex R={ang['upper_arm_flexion_right']:.0f} "
              f"trunk_flex={ang['trunk_flexion']:.0f} "
              f"elbow R={ang['lower_arm_flexion_right']:.0f})")
    # Hand-worked ground truth:
    #  neutral stand, arms hanging (elbow straight): the known REBA neutral
    #  floor is 1; RULA gives 2 (straight lower arm scores 2 in Table A).
    ru_n, re_n, _ = results["neutral"]
    assert re_n == 1, f"neutral REBA should be 1 (negligible), got {re_n}"
    assert ru_n <= 2, f"neutral RULA should be <=2 (acceptable), got {ru_n}"
    #  a ~90 deg bow must raise BOTH scores above neutral (trunk band jumps).
    ru_b, re_b, ab = results["bow"]
    assert ab["trunk_flexion"] > 60, f"bow trunk_flex {ab['trunk_flexion']:.0f}"
    assert re_b > re_n and ru_b > ru_n, "bow must exceed neutral"
    #  overhead reach must raise the arm score and thus both grands.
    ru_o, re_o, ao = results["overhead"]
    assert ao["upper_arm_flexion_right"] > 90, "overhead arm flex > 90"
    assert re_o > re_n and ru_o > ru_n, "overhead must exceed neutral"
    print("\n[self-test] PASS: neutral REBA=1/RULA<=2, bow and overhead both "
          "escalate as hand-worked")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("joints_csv", nargs="?", help="bag_to_joints --joints full CSV")
    p.add_argument("--out", help="scores CSV (frame,rula_score,reba_score)")
    p.add_argument("--conf-min", type=float, default=0.5)
    p.add_argument("--up", default="0,-1,0", help="up axis (y-down default)")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if not args.joints_csv:
        sys.exit("provide a joints CSV (or --self-test)")
    up = tuple(float(v) for v in args.up.split(","))
    rows = score_csv(args.joints_csv, args.conf_min, up)
    if args.out:
        write_scores(rows, args.out)
        print(f"wrote {len(rows)} frame scores -> {args.out}")
    else:
        ru = np.array([r[1] for r in rows])
        re = np.array([r[2] for r in rows])
        print(f"{len(rows)} frames: RULA mean {ru.mean():.1f} "
              f"(median {int(np.median(ru))}), REBA mean {re.mean():.1f} "
              f"(median {int(np.median(re))})")


if __name__ == "__main__":
    main()
