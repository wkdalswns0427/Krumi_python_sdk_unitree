# H1-2 experiment tooling (IROS 2026 workshop)

Offline analysis tools for the paper "From Worker to Humanoid: Markerless
Motion Retargeting and Ergonomic Assessment for Construction Tasks"
(IROS 2026 Workshop on Future of Construction). The experiment protocol and
schedule live in `~/mj_ws/h1-2_sensors/experiments/iros2026ws/EXPERIMENT_PLAN.md`
(data root, deviations log included).

All tools: system `/usr/bin/python3` (conda deactivated), paths as CLI args,
outputs default into a `b2g/` folder next to the input data.

## Tools in this directory

| Tool | Purpose |
|------|---------|
| `bag_to_joints.py` | capture -> per-frame 3D joints CSV (mono + rgbd pipelines) |
| `block1_compare.py` | mono vs rgbd per-joint 3D error (Block 1 metric) |
| `rula_reba_agreement.py` | RULA/REBA category agreement (Block 4 metric) |
| `angles.py` + `test_angles.py` | joints -> RULA/REBA input angles (scorer pending) |
| `visualize_joints.py` | joints CSV -> 3D skeleton image/animation/overlay |
| `bag_to_video.py` | rosbag color/depth stream -> mp4 for review |
| `diagnose_tail.py` | explain worst mono-vs-rgbd errors (depth bleed forensics) |
| `depth_policy_matrix.py` | one MediaPipe pass, all depth policies, coverage vs error |

Superseded here, canonical versions in `~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments/`:
`replay_logger.py` (ROS2 node), FK (`fk_h12.py`), `success_check.py`.
`latency_logger.py` remains the skeleton for the not-yet-built real-time
retarget node (Blocks 2/3).

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
consecutive rejections). Chosen from the `depth_policy_matrix.py` assessment
on M1_R_1 + test1 (~10x limb coverage for ~2 mm overall error; the `search`
fallback re-introduces background bleed and is rejected). Every run prints
`video_frames` / processed / fps / dt and per-joint row counts; check
`video_frames` matches the capture to catch input-path mixups.

`--diagnose-depth` prints the gate-free per-joint confidence composition and
unenforced jump statistics when a capture needs investigating.

## Block 1 evaluation workflow (per rep)

```bash
conda deactivate
cd ~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments
D=iphone_data/M1_R_1   # one capture dir per rep
/usr/bin/python3 bag_to_joints.py $D --mode mono --stride 2 --out $D/b2g/iph_mono.csv
/usr/bin/python3 bag_to_joints.py $D --mode rgbd --stride 2 --out $D/b2g/iph_rgbd.csv
/usr/bin/python3 block1_compare.py $D/b2g/iph_mono.csv $D/b2g/iph_rgbd.csv \
    --procrustes --out $D/b2g/block1_<tag>.csv --tag <tag>
```

Use the same `--stride` for mono and rgbd (frame indices are inner-joined).
Subject/device/capture metadata: `iphone_data/metadata.yaml` (capture `role:`
marks tuning vs evaluation clips).

### Alignment: mono and rgbd are in DIFFERENT frames

mono is MediaPipe's person-oriented world frame; rgbd is the camera optical
frame. A translation-only comparison reports that frame mismatch, not pose
error, so pick the metric deliberately:

| `block1_compare.py` flag | Removes | Use when |
|--------------------------|---------|----------|
| *(none)*                 | pelvis translation | both files already share a frame |
| `--procrustes`           | + rotation | **mono world vs rgbd camera (this study)** |
| `--procrustes-scale`     | + rotation + scale | PA-MPJPE, scale-invariant |

Both procrustes variants match the paper's claim language "relative 3D joint
error". Procrustes centres on the joint centroid (Kabsch), not the pelvis.

## Data contracts

### Per-frame 3D joints (Block 1)

Canonical long CSV (`.csv`), or MediaPipe motion JSON (`.json`,
`json2pose.py` format) auto-detected:

```
frame,joint,x,y,z,conf
0,left_shoulder,0.121,-0.052,0.301,0.98
```

- `frame`: integer index, inner-joined between files.
- `x,y,z`: `--in-units` (default metres).
- `conf`: [0,1]; only the mono file's conf drives the frame-exclusion rule
  (`--conf-threshold`, default 0.5; exclusion rate is reported).

Joint set: Block 1 scores 6 (L/R shoulder, elbow, wrist). Hips ride along so
the pelvis root can be derived. `--joints full` (bag_to_joints default) adds
knees + ankles for RULA/REBA scoring via `angles.py`; `block1_compare`
defaults to the 6-joint arm scope, so the wider export does not move Block 1
numbers. Face landmarks (nose, ears) are deliberately not exported; neck
scoring uses a neutral-neck assumption (like the wrist).

### Per-frame ergonomic scores (Block 4)

One CSV per pipeline, consumed by `rula_reba_agreement.py`:

```
frame,rula_score,reba_score
```

Grand scores are banded into risk categories with identical code for both
pipelines (or pass `rula_category`/`reba_category` directly). Bands, citable:
RULA 1-2 -> 1 Acceptable, 3-4 -> 2 Investigate, 5-6 -> 3 Change soon,
7 -> 4 Change now. REBA 1 -> 0 Negligible, 2-3 -> 1 Low, 4-7 -> 2 Medium,
8-10 -> 3 High, 11+ -> 4 Very high.

### Blocks 2/3

Replay CSV (cmd/exe/err per joint at 50 Hz) is written by the ROS2
`replay_logger` and consumed by `success_check.py` (FK from the URDF), both
in the h12_experiments package. Latency CSV (`t0..t3` per frame) comes from
the retarget node once it exists; `latency_logger.py` holds its skeleton.

## Self-tests (no hardware, no capture)

```bash
/usr/bin/python3 bag_to_joints.py --self-test   # deproject, policies, jump gate
/usr/bin/python3 block1_compare.py --demo
/usr/bin/python3 rula_reba_agreement.py --demo
/usr/bin/python3 -m pytest test_angles.py -q
```
