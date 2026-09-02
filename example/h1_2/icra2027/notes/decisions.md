# Design decisions log

Append-only. Note the date, the decision, the reason, and (if relevant) how
to revisit it. Do not delete entries; update them.

## 2026-08-25 — Repo scaffolded from a fresh start

- The user cleared the pre-existing `icra2027/` folder and asked for a
  clean setup aligned to the 2026-08-25 plan (`PLAN_v2.md`).
- Existing code reused (via import, not fork): `../experiments/scripts/retarget_arm.py`
  (`load_joints_csv`, `torso_frame`, `arm_directions`, `filter_depth_axis`),
  `~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments/h12_experiments/fk_h12.py`.
- Reasoning: the human-side conditioning (One-Euro filter, dt-aware
  teleport gate, depth low-pass) was chosen and tuned during IROS Blocks
  1-2 and is fine as-is. Rebuilding it here would risk numeric divergence
  from the paper the ICRA one cites.

## 2026-08-25 — Task-centric IK reuses iros_pilot's `ArmIK` idea

- `src/common/ik_taskcentric.py`'s solve loop mirrors
  `../iros_pilot/s2_bricklay.py::ArmIK.solve`: damped Gauss-Newton on tip
  position, yaw-anchor style regularizer moved into a general
  limit-avoidance null-space gradient.
- Reason: `s2_bricklay.py::ArmIK` is the working code that produced the
  IROS S2 pilot's task-optimal poses, so the ICRA solver should be
  numerically similar. Differences: we expose `lambda_limit`,
  `lambda_cont`, `lambda_manip` for weight sensitivity; `s2_bricklay.py`
  used a fixed `yaw_w=0.02`.
- Revisit: if a reviewer asks about the choice of gradient (mid-range vs
  gradient-of-a-barrier), swap `limit_avoidance_gradient` for a barrier
  and rerun X6/X2 to see if the paper numbers move.

## 2026-08-25 — Naive fidelity baseline for the flip counts

- `src/common/ik_baseline.py::solve_fidelity` defaults to `yaw_reg=0.0,
  yaw_neutral=0.0` — the "before regularizers" version.
- Reason: the plan's argument is against the field default (H2O,
  HumanPlus, OmniH2O, OKAMI, Liu&Liu), which does not include the yaw
  regularizers we introduced in IROS. Reporting flips under the
  regularizer-off baseline is the honest comparison for C3/F-a; the
  regularized version can be reported as a sub-result showing that
  patching the baseline gets partway.
- Revisit: if a reviewer objects that the paper is beating a strawman,
  add the regularized-fidelity numbers as a middle column in the results
  table (already easy — `--yaw-reg 0.05 --yaw-neutral 0.02` matches the
  IROS-frozen values).

## 2026-08-25 — Mechanical work uses effort-weighted |dq|, not gravity torque

- `src/x1_generated_across_motions.py::main` uses `sum |dq| * effort` as
  a mechanical-work proxy.
- Reason: a real gravity-torque calc needs link inertia parsing and a
  gravity vector aligned with the robot's actual base pose; that is
  possible from the URDF but adds a dependency (pinocchio or similar) and
  needs verification against the IROS Block 2 torque logs before it is
  worth reporting.
- Revisit: swap for pinocchio's `computeGeneralizedGravity` once the
  refilm is done and X1 numbers matter. Sanity-check against the IROS
  Block 2 measured torques for at least M2_R1_T1.

## 2026-08-25 — Preview run uses OLD captures explicitly

- Ran X6/X2/B2/X1 on `MotionsDataset0713/` even though the plan calls for
  a refilm.
- Reason: proves the pipeline end-to-end so the refilm session's data
  flows straight through the analysis instead of debugging on the
  hardware clock.
- Revisit: delete preview numbers from `README.md` once the real refilm
  numbers land, so nobody accidentally cites the preview.

## 2026-08-25 — Sensor pivot: iPhone LiDAR → ZED on robot neck

- Refilm and all future ICRA captures use the ZED depth camera mounted at
  the robot neck, not the iPhone-on-tripod pipeline the IROS paper used.
- User confirmations (2026-08-25):
  - Skeleton source: **ZED RGB + depth → MediaPipe** (reuse existing
    `bag_to_joints.py`; only the input reader changes).
  - Camera motion: **robot standing still** during capture; ZED-to-base
    extrinsic is fixed for the session.
  - Pick-and-stack (F-c evidence): **re-capture with ZED** so the paper
    does not mix sensors.
- Consequence for the plan text: the "iPhone = human capture, ZED =
  robot perception" sensing discipline is superseded. IROS numbers stay
  in the IROS paper; ICRA numbers are ZED-only. If a comparison against
  IROS's hand-designed pick-and-stack (for the C2 X1 anchor) is needed,
  it must acknowledge the sensor difference or reproduce the IROS
  motion on the ZED.
- Consequence for the repo: `src/common/data_paths.py` is now the single
  source of truth for `ACTIVE_DATA_ROOT`. Every driver imports it, so
  flipping to the ZED session folder is a one-line change.
- Consequence for Track C (extrinsics): D1 (ZED-to-base extrinsic) was
  optional in the plan; now it is required for the ICRA data pipeline.
- Revisit at first ZED capture: the frozen `--depth-policy relaxed-limb`
  thresholds were tuned on iPhone LiDAR confidence (0/1/2 per pixel).
  ZED depth confidence has a different distribution; re-run
  `../experiments/scripts/depth_policy_matrix.py` on the first ZED
  capture and either confirm or re-freeze the policy.
  **VOID 2026-08-26** — the skeleton source is ZED SDK body tracking, which
  does not sample a depth patch per keypoint, so no depth policy applies.
  `depth_policy_matrix.py` also does not exist in `../experiments/scripts/`.


## 2026-08-26 — Skeleton source: ZED SDK body tracking (BODY_38), not MediaPipe

- Supersedes the 2026-08-25 entry, which recorded "ZED RGB + depth →
  MediaPipe (reuse `bag_to_joints.py`; only the input reader changes)".
  The tool actually built, `src/zed/zed_to_joints.py`, is ZED SDK body
  tracking on BODY_38. Those are different algorithms, not different
  readers, and the code is what ships.
- User confirmed 2026-08-26: use the ZED SDK path.
- Reason: it is built and its self-test passes today; it is 3D-native, so
  there is no per-keypoint depth sampling and no depth policy to retune;
  and it already rescales ZED keypoint confidence from 0-100 to 0-1, so the
  emitted CSV matches the `frame,joint,x,y,z,conf` schema that
  `retarget_arm.py` and `rula_reba_score.py` already read.
- Cost, and it must be stated in the paper: the skeleton *algorithm* now
  differs from IROS in addition to the camera. Any cross-paper comparison
  names both, or reproduces the IROS motion on this pipeline.
- Consequence: the depth-policy retune step is void, and the
  `depth_policy_matrix.py` references (README §What has to happen next,
  notes/open_questions.md §0) have been removed because the script does
  not exist.
- Revisit if: BODY_38 keypoint quality on the first real capture is worse
  than the iPhone MediaPipe output at the same reach. Compare wrist-path
  smoothness on one rep before committing the whole session.

## 2026-08-26 — Capture format: SVO2, one file per capture

- User confirmed 2026-08-26. Recorded while streaming, so a session is
  captured and processed in one pass:
  `zed_to_joints.py --record <capture>.svo2 --out <joints>.csv`
- Reason: SVO2 packages RGB + depth + IMU + timestamps in one file that
  `zed_to_joints.py` already reads as a positional argument, and it stays
  re-processable if the body-format or confidence threshold needs changing
  after the session. The ROS2-bag alternative would have required a topic
  rename into `bag_to_joints.py`, which is the MediaPipe path we are not
  using.
- Revisit: only if a session needs to be synchronised against another ROS2
  recording (robot lowstate, for instance), where a bag would align clocks
  more directly.

## 2026-08-26 — Frozen wrist target in four drivers (bug, fixed)

- `x6_taskcentric.py`, `x2_costoffidelity.py`, `x2_robot_optimal.py` and
  `b2_control.py` all built the wrist target once, behind an
  `if build_target is None:` guard, so `robot_scaled_wrist_target` closed
  over frame 0's `u_hat`/`f_hat` and the target never moved. Measured on
  M2_1: 2.8 mm in x, 7.2 mm in z across all 1280 frames.
- Effect: the task-centric and DLS arms held a pose for the whole rep while
  the fidelity arm performed the motion. That produced their 0 flips,
  0.05 mm tip error and higher limit margin. Every `preview.json` in
  `results/` is invalid for this reason.
- Fix: rebuild the target inside the loop from the current frame's
  directions. Results re-run to `results/{x6,x2,b2}/fixed.json`; the old
  `preview.json` files are left in place for comparison and should be
  deleted once ZED numbers land.
- Lesson, same shape as the Kabsch/procrustes fix logged in the IROS
  scaffolds: when one condition scores implausibly perfectly (0 flips,
  0.05 mm, 0 violations), check whether it is being asked to do the task at
  all before believing it.

## 2026-08-26 — Re-run pipeline scripted (`rerun_all.sh`), and the `zed.json` naming trap

- Built `rerun_all.sh` to re-run every data-dependent analysis after a
  capture session: X6, X2, X8, B2, B3, X10, strawman, X1. X9/X3/X7 are
  excluded because they need the sweep captures or hardware. Everything reads
  through `src/common/data_paths.py`, so capture day is a one-line switch of
  `ACTIVE_DATA_ROOT`, then `./rerun_all.sh`.
- Smoke-tested on M2_1 today to prove the chain end-to-end before the refilm
  clock. It passed.
- Trap, recorded so nobody trips it: the script writes to
  `results/*/zed.json`, but `ZED_DATA_ROOT` is still `None` and
  `ACTIVE_DATA_ROOT = MOTIONS_DATA_ROOT` (the iPhone captures), so the current
  `zed.json` files hold the iPhone smoke-test, not ZED data. `results/x6/zed.json`
  is a single rep (M2_1) and its numbers equal `fixed.json` exactly. Do not cite
  any `zed.json` as ZED numbers until `ACTIVE_DATA_ROOT` is flipped to the real
  session folder and the script is re-run; the real captures overwrite these
  files in place.
- Revisit: at Stage 4, after flipping `ACTIVE_DATA_ROOT`, delete both the old
  `preview.json` and these smoke-test `zed.json` files if the re-run has not
  already overwritten them, so nothing stale survives into the paper.

## 2026-08-28 — D1 extrinsic accepted at about 9 mm, hand-eye

- Ran the hand-eye calibration (`src/zed/zed_extrinsic_capture.py` then
  `zed_extrinsic_solve.py`). 11 of 11 poses detected, board reprojection 0.24 to
  0.46 px. Result: ZED optical origin about 0.14 m forward and 0.53 m above
  torso_link, correct for a neck mount.
- Residual 9.0 mm and 1.2 deg, above the roughly 3 mm target. All four hand-eye
  methods agree, and dropping poses does not get below about 6 mm. The floor is
  kinematic (nominal URDF vs the real arm), not detection or method.
- Decision: accept it. The retargeting operates on arm directions in the
  human's own frame, so no experiment depends on the extrinsic. It is used only
  to co-visualize the human and the robot. 9 mm and 1.2 deg is about 4 cm at a
  2 m standing distance, which is fine for that.
- The forward-facing scene camera cannot see the robot's own hands from an
  up-front pose, so the poses reach the hand forward into view. Translation
  diversity is then capped by the forward cone and the arm reach, which is part
  of why the residual does not tighten.
- Revisit only if a human-plus-robot overlay figure needs a tighter fit. That
  would need joint-level kinematic calibration, which is out of scope.
- Stored at `results/extrinsic/zed_to_torso.yaml`. Not yet wired into
  `zed_realtime_node.py`, which still uses the identity placeholder.
