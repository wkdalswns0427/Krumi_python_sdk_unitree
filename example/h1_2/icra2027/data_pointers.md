# Data pointers

Nothing in this repo copies capture data. Every script reads from the paths
below directly, so the refilm can drop new folders in place. If a path
moves, update `src/common/data_paths.py` — that is the single source of
truth for `ACTIVE_DATA_ROOT` (which all drivers import).

## Sensor pivot (2026-08-25)

Capture switched from **iPhone LiDAR on tripod** (IROS) to **ZED depth
camera at the robot neck** (ICRA). Confirmed with user 2026-08-25:

- Skeleton source: **ZED RGB + depth → MediaPipe** (reuses the existing
  `bag_to_joints.py` pipeline; only the input reader changes).
- Camera motion: **robot standing still** during capture; ZED-to-base
  extrinsic is fixed for the whole session.
- Pick-and-stack (F-c evidence): to be **re-captured with the ZED** so
  the paper does not mix sensors.

The plan's original "iPhone = human capture, ZED = robot perception"
sensing discipline is superseded by this pivot.

## Human captures (ZED depth camera, robot-mounted, primary going forward)

- **Root (planned):** `~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/zed_data/<session>/`
  (exact folder name TBD; register it in `src/common/data_paths.py::ZED_DATA_ROOT`
  and flip `ACTIVE_DATA_ROOT` when the first session lands).
- **Capture format:** TBD (ZED SDK SVO file, or ROS2 bag via
  `zed-ros2-wrapper` at `~/mj_ws/zed/zed_ws/src/zed-ros2-wrapper`).
  `../experiments/scripts/bag_to_joints.py` already handles `rosbag`
  input for the D435i pipeline; a small topic-name tweak likely
  suffices for a ZED bag. SVO input needs a new reader (ZED SDK Python
  API required).
- **Per-capture output must be:** `<capture>/b2g/<name>.csv` with schema
  `frame,joint,x,y,z,conf` (metres, MediaPipe person-oriented world frame).
  If the ZED path produces a different filename, update
  `src/common/data_paths.py::JOINTS_CSV_REL`.
- **Extrinsic calibration** to the H1-2 base is required now (previously
  the plan's "D1" was optional). One session with the robot standing
  still and a checkerboard visible in the ZED view is enough; sanity-
  check against a lowstate-anchored reference target.

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

Frozen depth policy: `--depth-policy relaxed-limb` (default). Note that
this policy was tuned on the iPhone LiDAR depth stream (confidence 0/1/2
per pixel); the ZED depth stream carries a different confidence model
(per-pixel depth quality + neural confidence), so the policy thresholds
must be re-evaluated on the first ZED capture before it is trusted.

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
3. Produce `b2g/<name>.csv` per capture via `bag_to_joints.py` (adapted
   for ZED input).
4. Set `src/common/data_paths.py::ZED_DATA_ROOT` to the session folder
   and set `ACTIVE_DATA_ROOT = ZED_DATA_ROOT`.
5. If the joints CSV filename differs from `iph_mono.csv`, update
   `JOINTS_CSV_REL` in the same file.
6. Re-run X6, X2, B2, X1 — the `--captures` default in each driver is
   still M1_R_*/M2_*/M3_*; pass different names via CLI if the ZED
   session uses different folder names.
