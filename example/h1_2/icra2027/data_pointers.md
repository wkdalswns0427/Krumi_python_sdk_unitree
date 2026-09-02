# Data pointers

Nothing in this repo copies capture data. Every script reads from the paths
below directly, so the refilm can drop new folders in place. If a path
moves, update `src/common/data_paths.py` — that is the single source of
truth for `ACTIVE_DATA_ROOT` (which all drivers import).

## Sensor pivot (2026-08-25)

Capture switched from **iPhone LiDAR on tripod** (IROS) to **ZED depth
camera at the robot neck** (ICRA). Confirmed 2026-08-25, refined 2026-08-26
(see `notes/decisions.md`):

- Skeleton source: **ZED SDK body tracking, BODY_38**, via
  `src/zed/zed_to_joints.py` (3D-native; NOT the RGB->MediaPipe path the
  2026-08-25 note first assumed).
- Capture format: **SVO2**, one file per capture, recorded while streaming.
- Camera motion: **robot standing still** during capture; the ZED-to-torso
  extrinsic (D1 below) is fixed for the whole session.
- Pick-and-stack (F-c evidence): to be **re-captured with the ZED** so
  the paper does not mix sensors.

The plan's original "iPhone = human capture, ZED = robot perception"
sensing discipline is superseded by this pivot.

## Human captures (ZED depth camera, robot-mounted, primary going forward)

- **Root (planned):** `~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/zed_data/<session>/`
  (exact folder name TBD; register it in `src/common/data_paths.py::ZED_DATA_ROOT`
  and flip `ACTIVE_DATA_ROOT` when the first session lands).
- **Capture format:** **SVO2**, one file per capture, read by
  `src/zed/zed_to_joints.py` (built; the ZED SDK Python reader). The
  ROS2-bag path via `zed-ros2-wrapper` is not used.
- **Per-capture output must be:** `<capture>/b2g/<name>.csv` with schema
  `frame,joint,x,y,z,conf` (metres, ZED RIGHT_HANDED_Z_UP_X_FWD frame:
  x forward, z up). If the ZED path produces a different filename, update
  `src/common/data_paths.py::JOINTS_CSV_REL`.
- **D1 extrinsic (ZED optical -> torso_link):** done 2026-08-28, accepted at
  about 9 mm and 1.2 deg. Solved by hand-eye calibration
  (`src/zed/zed_extrinsic_capture.py` then `zed_extrinsic_solve.py`), stored at
  `results/extrinsic/zed_to_torso.yaml`. It is used only to co-visualize the
  human and robot in one scene. The retargeting works on directions in the
  human's own frame, so no result depends on it, which is why 9 mm is fine.
  The residual floor is kinematic (URDF vs the real arm), not fixable by more
  poses. Not yet wired into `zed_realtime_node.py`, which still uses an identity
  `world->frame` placeholder.
- **Skeleton sanity check** (Gate 2): before committing the session, compare
  the BODY_38 wrist paths against the iPhone baseline with
  `src/zed/skeleton_quality.py <zed>.csv --compare <iphone>/iph_mono.csv
  --fps 60 --compare-fps 30`. iPhone baseline (M2/M3, 2026-08-27): wrist
  dropout 0.1-8.8% (left wrist worse), jitter 3-10 mm at reach ~0.45-0.50 m.

## Human captures (iPhone LiDAR, IROS-era, retained for reference)

- **Root:** `~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/iphone_data/`
- **Metadata:** `.../iphone_data/metadata.yaml` (subject, device, per-capture role)
- **Captures (retained for the preview runs and historical reference; not
  the ICRA capture set going forward):**
  - `MotionsDataset0713/M1_R_1/`, `M1_R_2/`, `M1_R_3/` — overhead hammering, right arm
  - `MotionsDataset0713/M2_1/`, `M2_2/`, `M2_3/` — bricklaying
  - `MotionsDataset0713/M3_1/`, `M3_2/`, `M3_3/` — lifting
  - `Blockdata0725/Block{1,2,3}_{1,2,3,Full}/`, `Block_S2_{1,2,3}/` — pick-and-stack captures used by the IROS pilot

Each capture folder contains: `rgb.mp4`, `depth/*.png` (16-bit mm),
`confidence/*.png` (0/1/2), `camera_matrix.csv`. Processed 3D joints and
retargeted trajectories live in each capture's `b2g/` subfolder.

## Processed 3D joints (input to this repo's IK)

Each capture's `b2g/iph_mono.csv` is what `src/common/human_directions.py`
reads for the iPhone captures. Same schema will apply to the ZED captures.
Produced by:

```bash
/usr/bin/python3 ../experiments/scripts/bag_to_joints.py <capture_dir> \
    --mode mono --stride 2 --out <capture_dir>/b2g/<name>.csv
```

Frozen depth policy: `--depth-policy relaxed-limb` (default). This applies
to the **iPhone MediaPipe path only**. As of 2026-08-26 the ZED captures use
ZED SDK body tracking (BODY_38) via `src/zed/zed_to_joints.py`, which returns
3D keypoints directly and never samples a depth patch per keypoint, so no
depth policy applies to them and nothing needs re-tuning. Produce ZED joints
with:

```bash
/usr/bin/python3 src/zed/zed_to_joints.py <capture>.svo2 \
    --out <capture_dir>/b2g/<name>.csv --min-conf 20
```

`--min-conf` is the ZED SDK body-detection threshold on a 0-100 scale; 20
rather than the default 40. Per-keypoint confidence is rescaled to 0-1 in the
emitted CSV, matching the schema `retarget_arm.py` reads.

## Retargeted trajectories from IROS (reference; not directly consumed)

Each capture's `b2g/traj_iph_mono.csv` is the IROS retargeter's output at
50 Hz, produced by `../experiments/scripts/retarget_arm.py`. Schema:
`t_s,left_shoulder_pitch,...,right_wrist_yaw,waist_yaw`. This is the B1
output — the fidelity trajectory this paper compares against.

## Hardware replay logs (IROS Block 2/3, unchanged)

- **Root:** `~/mj_ws/h1-2_sensors/experiments/iros2026ws/block2/`
- Filenames: `replay_M{1,2,3}_R{1,2,3}_T{1,2,3}.csv` (motion, rep, hardware trial). Per-frame cmd/exe/err/torque at 50 Hz. Aggregates: `block2_aggregate.csv`, `block2_gain_sweep.csv`.
- Yaw regularizer sweep results: `block2/yaw_sweep/yaw_reg_sweep.csv`.
- These logs are all iPhone-derived retargets replayed on the robot; ZED
  hardware sessions for X3 will produce a new log family under
  `results/x3/`.

## ZED tooling on this machine

- ROS2 wrapper source: `~/mj_ws/zed/zed_ws/src/zed-ros2-wrapper` (built).
- ZED SDK samples: `~/mj_ws/zed/samples/` (body tracking, depth sensing,
  recording — useful references for the SVO reader).

## URDF

`~/mj_ws/assets/h1_2_description/h1_2.urdf`. All FK / IK / limit lookups
use this file (default in every script here).

Key joint limits (relevant to the flip / violation story):

| joint | lower (rad) | upper (rad) | effort (Nm) | velocity (rad/s) |
| --- | --- | --- | --- | --- |
| L/R shoulder_pitch | -3.14 | 1.57 | 40 | 9 |
| L/R shoulder_roll | -0.38 / -3.4 | 3.4 / 0.38 | 40 | 9 |
| L/R shoulder_yaw | -2.66 | 3.01 | 18 | 20 |
| L/R elbow | -0.95 | 3.18 | 18 | 20 |

(Shoulder roll is mirrored across sides.)

## Where new ZED captures should land

1. Choose a session folder, e.g.
   `~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/zed_data/2026-08-XX/`.
2. Save raw ZED capture (SVO or ROS2 bag) inside each per-capture
   subfolder, e.g. `zed_data/2026-08-XX/M1_R_1/`.
3. Produce `b2g/<name>.csv` per capture via
   `src/zed/zed_to_joints.py <capture>.svo2 --out <capture>/b2g/<name>.csv --min-conf 20`.
4. Set `src/common/data_paths.py::ZED_DATA_ROOT` to the session folder
   and set `ACTIVE_DATA_ROOT = ZED_DATA_ROOT`.
5. If the joints CSV filename differs from `iph_mono.csv`, update
   `JOINTS_CSV_REL` in the same file.
6. Re-run everything with `./rerun_all.sh` (X6, X2, X8, B2, B3, X10,
   strawman, X1); pass capture names if the ZED session uses different
   folder names.
