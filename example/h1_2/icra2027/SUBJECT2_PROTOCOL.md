# Capture protocol

S2 and S3. S1 is complete and needs nothing further.

---

## 1. Before anyone moves

### Measure each subject

Arm hanging relaxed, both sides, twice each. Take both measurements.

| symbol | measurement |
| --- | --- |
| **L_wrist** | shoulder joint centre to the **wrist crease** |
| L_hand | shoulder joint centre to the **back of the hand** |

`L_wrist` is what every later step divides by. Already measured:

| subject | L_wrist | L_hand |
| --- | --- | --- |
| S1 | 22.0 in, 0.5588 m | 24 in |
| S2 | 19.5 in, 0.4953 m | 21 in |
| S3 | 22.5 in, 0.5715 m | 25 in |

Record height, and whether the subject has construction work experience.

### Set the camera

- ZED at the robot neck, HD720, 60 fps, one SVO2 per capture.
- Robot standing still.
- **Subject facing the camera, square on.** This is the geometry S1 used and the only one in use.
- Same camera position for every block and every subject. Do not move it once shooting starts.
- `--min-conf 20` throughout.

### Place the pole

A pole standing at **chest height**, at these horizontal distances forward from the toes. The subject touches the pole, not the floor.

Distances are fractions of `L_wrist` so the levels match S1's in ratio terms.

| level | fraction | S2, L=19.5 | S3, L=22.5 | S1 was |
| --- | --- | --- | --- | --- |
| d1 | 0.55 L | 10.6 in | 12.3 in | 12 in |
| d2 | 0.73 L | 14.2 in | 16.4 in | 16 in |
| d3 | 0.83 L | 16.2 in | 18.7 in | 18.3 in |
| d4 | 0.91 L | 17.7 in | 20.5 in | 20 in |
| d5 | 1.00 L | 19.5 in | 22.5 in | 22 in |
| d6 | 1.09 L | 21.3 in | 24.5 in | 24 in |

---

## 2. Shot list

### Block A. Reach sweep. Do this first.

Six levels, three files per level, five reps inside each file. **18 files, 90 reaches per subject.**

```
RS_d1_1  RS_d1_2  RS_d1_3
RS_d2_1  RS_d2_2  RS_d2_3
RS_d3_1  RS_d3_2  RS_d3_3
RS_d4_1  RS_d4_2  RS_d4_3
RS_d5_1  RS_d5_2  RS_d5_3
RS_d6_1  RS_d6_2  RS_d6_3
```

**Keep the reps tight.** Dead time is what ruins these captures. Aim for under 60 percent of frames with the arm at rest, which means five reps back to back with minimal pause.

How to reach, every time:

- Forward at chest height, touch the mark, then return.
- Five reps per recording.
- Feet planted and torso upright. No leaning, no stepping, no shoulder travel.
- Touch the pole, not the floor. The reach is horizontal, not downward.

Watch the preview. If the working wrist drops tracking, redo the file.

### Block B. Motions.

- **M2 bricklaying**, 5 reps
- **M3 lifting**, 5 reps

Keep the placing hand visible to the camera.

### Block C. If time allows.

- **M1 overhead hammering**, 3 reps
- **M4 pick-and-stack**, 3 reps
- **M5 cross-body reach**, 3 reps

### If the day runs short

S2 is the prediction test and S3 is a same-length replication of S1. Finish S2 completely before starting S3, and stop at the end of a block rather than part way through one.

---

## 3. Commands

```bash
conda activate rical_unitree
ICRA=$HOME/mj_ws/Krumi_python_sdk_unitree/example/h1_2/icra2027
ZED=$HOME/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/zed_data

mkdir -p $ZED/S2_2026-09-02
mkdir -p $ZED/S3_2026-09-02
cd $ZED/S2_2026-09-02                      # or whichever subject

ZTJ="python3 $ICRA/src/zed/zed_to_joints.py"
$ZTJ --record RS_d1_1.svo2 --out RS_d1_1.csv --min-conf 20 --resolution HD720 --show
# ... through RS_d6_3, then the motions
```

Write a `subject.txt` in each folder:

```bash
cat > subject.txt <<'EOF'
subject_id:         S2
arm_wrist_m:        0.4953
arm_hand_m:         0.5334
height_m:           1.XX
construction_exp:   yes/no
camera_azimuth_deg: 0        # square on
EOF
```

---

## 4. Check after every two levels, before the pole moves on

```bash
python3 $ICRA/src/zed/check_sweep_span.py RS_d*_[123].csv
```

Two columns decide whether a level is usable.

| column | want | if not |
| --- | --- | --- |
| `%rest` | under 60 percent | reps are too far apart, redo the level tighter |
| `fwd cm` | rising down the column | the pole did not move as intended, check the distance |

The `ext` column should span at least 0.15 from d1 to d6. S1's reference span is 0.28.

### What S1 produced, for comparison

After conditioning, S1's six levels landed at these reconstructed extensions, with limit contact under direction matching in brackets:

| level | S1 extension | contact |
| --- | --- | --- |
| d1 | 90.7 % | 0/5 |
| d2 | 94.7 % | 1/5 |
| d4 | 97.6 % | 3/5 |
| d6 | 99.9 % | 5/5 |

**The prediction is that S2 and S3 reproduce this curve in extension terms while their pole distances differ in centimetres.** S2's poles sit about 2.5 in closer than S1's throughout, so if the threshold is governed by ratio rather than distance, S2 should show the same onset near 94 to 95 percent at those nearer marks.

Note that S1 never sampled below 90 percent extension even at the nearest pole. That is a property of the motion rather than the marks, since a person extends the arm substantially to touch anything in front of them. Do not chase lower extensions by moving the pole very close, which produced degenerate near-zero forward reach when it was tried.

---

## 5. After the session

1. Trim: `trim_joints.py <f>.csv --fps 60 --start 5 --end 2.5 --inplace`
2. Clean single-frame outliers at 400 mm.
3. Normalize to that subject's `arm_wrist_m`:
   ```bash
   python3 $ICRA/src/zed/normalize_skeleton.py RS_d1_1.csv --arm-length-m 0.4953 --inplace
   ```
4. Add the session folder to `SESSIONS` and `ARM_LENGTH_M` in `src/common/data_paths.py`.
5. Re-run the sweep and the motions with `ICRA_SESSION=S2`.
6. Compare the threshold in **extension ratio**, not in centimetres.

Plot violation rate against reconstructed extension ratio for all subjects on one axis. The prediction is that the curves lie on top of each other in ratio terms while sitting at different absolute distances. State plainly which outcome occurred.
