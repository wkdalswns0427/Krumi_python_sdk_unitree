# H1-2 experiment tooling (shared IROS 2026 + ICRA 2027)

Offline tooling shared by the IROS 2026 workshop paper and the ICRA 2027
follow-up (`../../icra2027/`). The Block 1 / Block 4 / one-time-sweep
scripts were removed in the 2026-08-25 cleanup with the ICRA pivot — see
`../../icra2027/PLAN.md` for the current scope.

All tools: system `/usr/bin/python3` (conda deactivated), paths as CLI args.

## Tools in this directory

| Tool | Purpose |
|------|---------|
| `bag_to_joints.py` | capture -> per-frame 3D joints CSV (mono + rgbd pipelines) |
| `angles.py` + `test_angles.py` | joints -> RULA/REBA input angles |
| `rula_reba_score.py` | RULA/REBA scoring (X7 manipulation check) |
| `visualize_joints.py` | joints CSV -> 3D skeleton image/animation/overlay |
| `bag_to_video.py` | rosbag color/depth stream -> mp4 for review |
| `retarget_arm.py` | joints CSV -> 50 Hz H1-2 arm trajectory |
| `replay_arm.py` | trajectory -> rt/arm_sdk overlay player (unitree_sdk2py) |
| `sdk_replay_logger.py` | cmd/exe logger over raw DDS; primary log for X3 |
| `latency_logger.py` | skeleton for the not-yet-built real-time retarget node |

Canonical in `~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments/`:
`replay_logger.py` (ROS2 node, backup logging path), FK (`fk_h12.py`),
`success_check.py`.

For ZED captures see `../../icra2027/src/zed/` (Track C).

## Capture sources

`bag_to_joints.py` auto-detects the source (override with `--source`):

- **rgbd-dir** (primary for Block 1): iPhone 16 Pro LiDAR via Stray Scanner;
  a capture dir with `rgb.mp4`, `depth/*.png` (16-bit mm), `confidence/*.png`
  (0/1/2), `camera_matrix.csv`. Depth is sampled at native resolution.
- **rosbag**: D435i ROS2 bag (color + aligned depth + camera_info topics);
  used for the robot-mounted deployment camera.

## Depth sampling policy (FROZEN 2026-07-10)

Default `--depth-policy relaxed-limb`: LiDAR confidence >= 1 accepted at
elbows/wrists/knees/ankles, confidence == 2 required at shoulders/hips; 3x3
native-pixel patch, median of survivors, 150 mm spread reject, dt-aware jump
gate (`--max-joint-speed` 4 m/s x actual frame gap, with re-lock after 5
consecutive rejections). Chosen on M1_R_1 + test1 (~10x limb coverage for
~2 mm overall error). Every run prints `video_frames` / processed / fps / dt
and per-joint row counts; check `video_frames` matches the capture to catch
input-path mixups.

`--diagnose-depth` prints the gate-free per-joint confidence composition and
unenforced jump statistics when a capture needs investigating.

## Data contract: per-frame 3D joints

Canonical long CSV (`.csv`), or MediaPipe motion JSON (`.json`,
`json2pose.py` format) auto-detected:

```
frame,joint,x,y,z,conf
0,left_shoulder,0.121,-0.052,0.301,0.98
```

- `frame`: integer index.
- `x,y,z`: `--in-units` (default metres).
- `conf`: [0,1]; the mono file's conf drives the frame-exclusion rule
  (`--conf-threshold`, default 0.5).

Joint set: `--joints full` (default) exports 12 joints (L/R shoulder, elbow,
wrist, hip, knee, ankle) for RULA/REBA scoring via `angles.py`. Face
landmarks (nose, ears) are deliberately not exported; neck scoring uses a
neutral-neck assumption.

## Replay logging

Replay CSV (cmd/exe/err per joint at 50 Hz) is consumed by
`success_check.py` (FK from the URDF) in the h12_experiments package. Two
writers produce the identical format: the ROS2 `replay_logger` node and
`sdk_replay_logger.py` here (raw DDS via unitree_sdk2py, same stack as
`replay_arm.py`). Both are verified; the SDK logger is the primary (same
conda env and `--interface` flag as the replayer), and both abort early if
no lowstate arrives instead of logging NaN silently.

Lesson from the 2026-07-13 bring-up (two all-NaN trial logs): ROS2
prepends `rt/` to every topic name when mapping to DDS, so a ROS2 node
must subscribe to `lowstate` / `arm_sdk` to reach the robot's raw DDS
topics `rt/lowstate` / `rt/arm_sdk`; subscribing to `rt/lowstate` from
ROS2 listens on DDS `rt/rt/lowstate`, which nobody publishes. This is why
the official unitree_ros2 examples use `/lowstate`, `/arm_sdk`, `/lowcmd`.

`replay_arm.py --gravity-ff` (added 2026-07-13): the pilot M1 T1 failed
`success_check` at 12.35 cm mean EE, diagnosed as gravity sag (shoulder
error 97% one-signed, ~10 deg even at static holds, because the overlay
sent `tau=0` with `kp=80`). The flag adds a URDF-based gravity feedforward
`tau = dU/dq` (`GravityFF`, capped per joint, faded with the overlay
weight, `--gravity-gain` default 1.0). Sign and magnitude were validated
against the pilot log: model torque vs measured `kp*err` at static holds
correlate +0.95/+0.97 with the same sign, so the feedforward lifts rather
than doubles the sag. Predicted residual: gain 1.0 -> ~4 cm EE (should
pass 5 cm), gain ~1.4 -> ~1.5 cm. Start on the robot at gain 1.0.

Latency CSV (`t0..t3` per frame) comes from the retarget node once it
exists; `latency_logger.py` holds its skeleton.

`retarget_arm.py` input conditioning (added 2026-07-13 after the first
robot replay showed the raised-but-static left arm flapping): mono
MediaPipe depth (z) is the noisy axis, on M1_R_1 the static left wrist
measured 2.0 / 2.8 / 11.6 cm std in x / y / z with the z oscillation
phase-locked to the right-arm hammer strikes. Conditioning: median-5 +
zero-phase low-pass on every keypoint's z (`--z-cutoff` 0.8 Hz), a
dt-aware teleport gate on limb directions (`--max-dir-rate` 600 deg/s,
re-lock after `--relock` 5), a One-Euro filter on the directions
(`--euro-min-cutoff` 1.0, `--euro-beta` 0.5) and a zero-phase Butterworth
on the output joints (`--lp-cutoff` 3 Hz; direction-space spectrum shows
real motion incl. the hammer whip is < 3 Hz). M1_R_1 effect: left-arm
mean |vel| 53 -> 24 deg/s, velocity reversals 4.9 -> 1.5 /s, right-arm
strike rhythm preserved (peaks 8.3 -> 5.8 rad/s).

## Self-tests (no hardware, no capture)

```bash
/usr/bin/python3 bag_to_joints.py --self-test   # deproject, policies, jump gate
/usr/bin/python3 -m pytest test_angles.py -q
/usr/bin/python3 replay_arm.py TRAJ.csv --dry-run
# real DDS round-trip on loopback (conda env with unitree_sdk2py):
python3 sdk_replay_logger.py --self-test
```
