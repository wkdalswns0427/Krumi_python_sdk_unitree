# ICRA 2027 — Robot-Specific Motion for Construction Humanoids

**Handoff for future Claude sessions.** Read `PLAN.md` first — it is the
canonical plan (verbatim from the user on 2026-08-25, with a sensor-pivot
addendum). This file explains what the repo contains, what has been
implemented, what has been preview-run on the OLD captures, and what still
needs to happen.

**Sensor pivot (2026-08-25):** capture switches from iPhone LiDAR on tripod
(IROS) to ZED at the H1-2 neck (ICRA). ZED RGB + depth feed MediaPipe;
robot stands still; pick-and-stack will be re-captured. `PLAN.md` addendum
+ `data_pointers.md` + `notes/decisions.md` are authoritative. All drivers
route through `src/common/data_paths.py::ACTIVE_DATA_ROOT`, so switching
the capture root is a one-line edit.

## What this paper is

Follow-up to the IROS 2026 workshop paper `../iros_pilot/S2_STATUS.md` and
`IROS_WS_FoC.pdf` (the submitted PDF, attached to the plan). IROS asked
"does departing from a human demonstration help?" and answered with one
hand-designed motion on one sequence (pick-and-stack). ICRA answers the
same question systematically: automated Robot-Specific generation across
M1/M2/M3, with a mechanism explanation (joint-limit margin, branch flips,
in the limit infeasibility) traced to the inherited human posture.

Primary contributions per `PLAN.md`:
- **C2.** Automated Robot-Specific Motion generation across motions
  (was hand-designed for one sequence in IROS).
- **C3.** The feasibility mechanism: fidelity retargeting leaves the arm
  near its own joint limits → branch flips, violations, in the limit
  infeasible motion.
- **C4.** Replica vs replacement modes: fidelity is a choice, not a default.
- **C1.** Restated in one paragraph and cited to the IROS paper.

## Repo layout

```
icra2027/
├── PLAN.md          canonical plan (verbatim, read first)
├── README.md        this file
├── data_pointers.md exact paths to captures + processed outputs (no data dup)
├── src/
│   ├── common/
│   │   ├── data_paths.py             ACTIVE_DATA_ROOT (single point of change)
│   │   ├── fk.py                     ArmChainFK wrapper (reuses fk_h12)
│   │   ├── ik_baseline.py            B1: direction-matching (fidelity)
│   │   ├── ik_taskcentric.py         X6/X1: tip-pos + null-space robot regs
│   │   ├── ik_dls.py                 B2: textbook DLS + null-space limits
│   │   ├── metrics.py                margin, manip, flips, approach err
│   │   └── human_directions.py       reader over iph_mono.csv (uses IROS ret.)
│   ├── x6_taskcentric.py             F-a: flips under fidelity vs task-centric
│   ├── x2_costoffidelity.py          C3/F-b: margin/manip/violation cost
│   ├── x2_robot_optimal.py           writes replay-ready task-optimal trajs
│   ├── b2_control.py                 generic DLS baseline (approach-err cost)
│   ├── x1_generated_across_motions.py  C2 primary; skeleton, needs refilm
│   ├── x3_hardware_replay.py         non-negotiable hardware step; skeleton
│   ├── x7_dual_mode.py               C4 demonstration; skeleton
│   └── reach_extent_sweep.py         C3 curve; needs refilm capture
├── results/                          per-experiment JSON + per-rep CSVs
│   ├── x6/preview.json + one CSV per rep+method
│   ├── x2/preview.json + frame-level CSVs
│   ├── b2/preview.json + per-rep CSVs
│   └── x1/  x3/  x7/  reach_extent/  (empty until run)
├── figures/                          exported figures (empty)
└── notes/
    ├── contributions.md              C→X→result traceability
    ├── decisions.md                  design decisions log
    └── open_questions.md             for advisor conversations
```

## Preview status (2026-08-25)

Scaffolding built and X6, X2, B2 preview-run on the OLD captures at
`~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/iphone_data/MotionsDataset0713/`
(M2_1..M2_3, M3_1..M3_3, right arm). These OLD numbers are **not** paper
numbers — the plan calls for refilming the captures and rerunning
before anything is reported. Preview numbers exist to prove the pipeline
end-to-end.

**Old-capture preview totals** (right arm, 6 reps M2+M3, iph_mono.csv):

| Experiment | Metric | Fidelity | Task-centric / DLS |
| --- | --- | --- | --- |
| X6 | flips (yaw, > 6 rad/s) | 90 total | 0 total |
| X6 | reps with limit violation | 4 / 6 | 0 / 6 |
| X2 | mean limit margin | 0.24 | 0.30 |
| X2 | reps with any violation | 4 / 6 | 0 / 6 |
| B2 (DLS) | flips | — | 0 total |
| B2 (DLS) | mean approach-dir err (deg) | — | ~40 (rep mean, not contact-only) |

Discrepancies vs the numbers cited in `PLAN.md` ("47 flips", "2 of 6 violation
reps", "12 deg approach err at contact events") are expected: those numbers
came from earlier work with a different capture set and different IK
regularizer choices; this scaffold uses the NAIVE fidelity IK (no yaw_reg /
yaw_neutral) so it reports the "before regularizers" baseline the paper
argues against. The refilm + re-run will produce the paper numbers.

Also: X1's preview center-length is implausibly small (0.00-0.02 m) on the
old captures. This is because the retargeter treats each arm independently
and the midpoint of two independent wrists stays near the shoulder midline.
For the IROS pick-and-stack the two hands hold the same block so the
midpoint is meaningful; for M1/M2/M3 as recorded, the midpoint metric may
not be the right efficiency measure — see `notes/open_questions.md`.

## How to run

All scripts use system `/usr/bin/python3` (numpy + scipy only). Working
directory is this folder. The FK utilities come from
`~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments`; the human joints CSVs
are produced by `../experiments/scripts/bag_to_joints.py` (already done
for the OLD captures; the refilm requires re-running it).

```bash
cd ~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/icra2027

# X6: does the objective flip fix flips?
/usr/bin/python3 src/x6_taskcentric.py --side right
# writes results/x6/preview.json + one CSV per (rep, method)

# X2: what does fidelity cost, task held fixed?
/usr/bin/python3 src/x2_costoffidelity.py --side right
# writes results/x2/preview.json + frame-level CSVs

# X2 replay-ready trajectory (per capture):
/usr/bin/python3 src/x2_robot_optimal.py M2_1
# writes results/x2/traj_M2_1.csv (replayable by ../experiments/scripts/replay_arm.py)

# B2: does generic DLS + null-space also fix flips?
/usr/bin/python3 src/b2_control.py --side right
# writes results/b2/preview.json

# X1 (SKELETON, needs refilm): generation across M1/M2/M3
/usr/bin/python3 src/x1_generated_across_motions.py
# writes efficiency-measure previews

# X3 (SKELETON, hardware, DO NOT run without e-stop operator)
/usr/bin/python3 src/x3_hardware_replay.py results/x2/traj_M2_1.csv \
    --interface enp128s31f6 --dry-run

# X7 (SKELETON)
/usr/bin/python3 src/x7_dual_mode.py --capture M2_1

# Reach-extent sweep (SKELETON, needs the refilm capture)
/usr/bin/python3 src/reach_extent_sweep.py
```

## What has to happen next

Ordered by the plan's critical path, updated for the ZED sensor pivot.

0. **First ZED capture session logistics** (see `notes/open_questions.md` §0):
   - Decide SVO vs ROS2 bag as the capture format.
   - Measure the ZED-to-torso_link extrinsic once (robot standing still,
     checkerboard in view). Store in `data_pointers.md`.
   - Run `depth_policy_matrix.py` on one throwaway capture and either
     confirm or re-freeze `--depth-policy relaxed-limb` for the ZED
     confidence distribution.
1. **Refilm M1, M2, M3 and the reach-extent sweep** with the ZED at the
   robot neck, robot standing still (`PLAN.md` §Motion data + Addendum).
   Aug 15 deadline in the plan.
2. **Reprocess** the refilmed captures with
   `../experiments/scripts/bag_to_joints.py` (adapted for ZED input; the
   `rosbag` path likely works with topic renames if capturing to a ROS2
   bag). Output must land as `<capture>/b2g/<name>.csv` in the
   `frame,joint,x,y,z,conf` schema.
3. **Point the repo at the new captures.** Edit `src/common/data_paths.py`:
   set `ZED_DATA_ROOT` to the session folder, flip
   `ACTIVE_DATA_ROOT = ZED_DATA_ROOT`, and update `JOINTS_CSV_REL` if the
   CSV filename changed from `iph_mono.csv`.
4. **Re-run X6 and X2** by pointing `--captures` at the new folder names
   (`src/x6_taskcentric.py` and `src/x2_costoffidelity.py`).
5. **Wire X1** to whichever comparison anchor advisors choose: (a) same
   pick-and-stack sequence as IROS but re-captured on the ZED (needs
   `../iros_pilot/s2_bricklay*.py` as the task spec, and a fresh
   pick-and-stack capture session), or (b) M1/M2/M3 efficiency with an
   independent framing. See `src/x1_generated_across_motions.py` docstring.
6. **X3 hardware sessions.** Reserve the interface flag from the current
   robot's network setup; run each trajectory with `--gravity-ff --gravity-gain 1.0`
   per the IROS bring-up lesson. Grade with `success_check.py`-style FK
   and specifically check whether X2's limit-violation frames appear on
   the executed logs.
7. **X7 dual-mode.** Score both trajectories with
   `../experiments/scripts/rula_reba_score.py` (not yet integrated). If
   both modes land in overlapping posture bands, C4 has to be argued
   rather than demonstrated — flag this to advisors early.

## What NOT to do

- Do not report the preview numbers in this folder as paper numbers.
  Everything under `results/*/preview.json` is a pipeline sanity check on
  captures the plan intends to replace.
- Do not renumber X6/X2/X7/X1/X5 or B2. Filenames encode the plan's
  identifiers; renumbering breaks the traceability table.
- Do not add a fourth construction task. `PLAN.md` explicitly says "ICRA
  should vary the quantity that causes the failure," which is the reach-
  extent sweep, not more task variety.
- Do not resurrect the learned retargeter, the novel-solver claim, or
  the gravity-cost claim. The plan retires all three; introducing them
  contradicts C2/C3.
- Do not run X3 without a human at the e-stop.

## Data pointers

See `data_pointers.md`. Nothing in this repo copies capture data — every
script reads from
`~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/iphone_data/`
directly, so the refilm can drop new folders in place without rewiring.

## Related repos

- `../experiments/scripts/` — IROS analysis tools (retarget_arm.py,
  replay_arm.py, angles.py, rula_reba_*.py). Reuse; do not fork.
- `../iros_pilot/` — S2 pick-and-stack pilot, hand-designed native motion,
  hardware traces. F-c evidence lives here.
- `~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments/` — canonical FK, ROS2
  replay logger, success check.
- `~/mj_ws/assets/h1_2_description/h1_2.urdf` — H1-2 URDF used everywhere.
