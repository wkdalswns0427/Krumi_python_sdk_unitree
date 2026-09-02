# Capture protocol

**Status: complete. S1 and S2 are captured and analysed. S3 was dropped.**

S2 provided Blocks A and B. Its motion captures replicate S1 and are reported. Its reach sweep saturated, reading above 99 percent reconstructed extension at every level, and is not used.

If a sweep is ever reshot, the one thing to change is pole placement. S2's two nearest poles produced 87 and 78 percent dead time and only 5 to 16 cm of forward reach, which is the same failure the S1 refilm showed. Push the near poles further out and keep the reps tight.

The procedure below is kept as the record of what was done.

---

## 1. Before anyone moves

### Measure each subject

Arm hanging relaxed, both sides, twice each. Take both measurements.

| symbol | measurement |
| --- | --- |
| **L_wrist** | shoulder joint centre to the **wrist crease** |
| L_hand | shoulder joint centre to the **back of the hand** |

`L_wrist` sets the pole distances and is what the skeleton normalization and every extension ratio divide by. A wrong value here corrupts the session silently. Already measured:

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

**The pole is touched by the palm, but the tracked keypoint is the wrist**, which sits one hand-length behind it. Every distance below accounts for that.

    pole distance  =  fraction * L_wrist  +  hand offset
    hand offset    =  L_hand - L_wrist

| subject | L_wrist | hand offset |
| --- | --- | --- |
| S1 | 22.0 in | 2.0 in |
| S2 | 19.5 in | 1.5 in |
| S3 | 22.5 in | 2.5 in |

The offset differs by a full inch across the three, so it cannot be folded into a single scale factor.

| level | wrist reach | S1 (already shot) | S2 | S3 |
| --- | --- | --- | --- | --- |
| d1 | 0.45 L | 12.0 in | **10.4 in** | **12.7 in** |
| d2 | 0.64 L | 16.0 in | **13.9 in** | **16.8 in** |
| d3 | 0.82 L | 20.0 in | **17.5 in** | **20.9 in** |
| d4 | 1.00 L | 24.0 in | **21.0 in** | **25.0 in** |

The fractions come from S1's own poles converted to wrist reach, so S1's column reproduces what was actually shot and the other two match it in wrist terms.

**Why fractions rather than fixed inches.** The sweep has to locate the onset inside each subject's own curve, so every subject's levels must bracket their own threshold. Reconstructing S1's geometry, the reach vector carries a large non-forward component of roughly 37 cm set by posture rather than by the pole. Holding the pole at a fixed distance would therefore push a shorter arm to a higher extension at every level. S2 would read about 0.90 where S1 reads 0.85, and would be at or past full extension by the middle of the sweep, returning 5 of 5 everywhere and locating nothing.

---

## 2. Shot list

### Block A. Reach sweep. Do this first.

Four levels, five files per level, five reps inside each file. **20 files, 100 reaches per subject.** This matches S1's structure of five files per level.

```
RS_d1_1  RS_d1_2  RS_d1_3  RS_d1_4  RS_d1_5
RS_d2_1  RS_d2_2  RS_d2_3  RS_d2_4  RS_d2_5
RS_d3_1  RS_d3_2  RS_d3_3  RS_d3_4  RS_d3_5
RS_d4_1  RS_d4_2  RS_d4_3  RS_d4_4  RS_d4_5
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
# ... through RS_d4_5, then the motions
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
python3 $ICRA/src/zed/check_sweep_span.py RS_d*_[1-5].csv
```

Two columns decide whether a level is usable.

| column | want | if not |
| --- | --- | --- |
| `%rest` | under 60 percent | reps are too far apart, redo the level tighter |
| `fwd cm` | rising down the column | the pole did not move as intended, check the distance |

The `ext` column should span at least 0.15 from d1 to d4. S1's reference span is 0.28.

### What S1 produced, for comparison

After conditioning, S1's six levels landed at these reconstructed extensions, with limit contact under direction matching in brackets:

| level | pole | S1 extension | contact |
| --- | --- | --- | --- |
| d1 | 12 in | 90.7 % | 0/5 |
| d2 | 16 in | 94.7 % | 1/5 |
| d3 | 20 in | 97.6 % | 3/5 |
| d4 | 24 in | 99.9 % | 5/5 |

**The prediction is that the onset appears near 94 to 95 percent extension for every subject, at pole distances that differ by inches.** S2's poles sit 1.6 to 3.0 in nearer than S1's, so if the onset lands at the same extension in both, the governing quantity is the ratio and not the distance. S3 is within half an inch of S1 in arm length but its poles run up to 1 in further out, because its hand is longer. If S3 reproduces S1's curve that is a replication, and it also confirms the hand-offset correction is doing its job.

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

Plot violation rate against reconstructed extension for all three subjects on one axis, and against pole distance on a second. The prediction is that the curves coincide on the extension axis and separate on the distance axis, with S2 shifted nearest. State plainly which outcome occurred, including the case where they fail to coincide.
