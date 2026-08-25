# Design decisions log

Append-only. Note the date, the decision, the reason, and (if relevant) how
to revisit it. Do not delete entries; update them.

## 2026-08-25 — Repo scaffolded from a fresh start

- The user cleared the pre-existing `icra2027/` folder and asked for a
  clean setup aligned to the 2026-08-25 plan (`PLAN.md`).
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
