# Open questions (for advisor conversations)

Prioritized by how much the answer changes downstream work. The plan's
timeline puts the advisor conversation Aug 11-15; the first three should
happen there.

## 0. ZED capture pipeline details (blocks the first ZED session)

Sensor pivot confirmed 2026-08-25 (see `notes/decisions.md`). Remaining
technical decisions before capture:

- **Capture format:** SVO file (ZED SDK) or ROS2 bag (via
  `zed-ros2-wrapper`). SVO packages RGB + depth + IMU + timestamps in
  one file; ROS2 bag reuses `bag_to_joints.py`'s rosbag path with a
  topic rename. Pick one before capture starts.
- **ZED resolution / fps:** HD720@60 or HD1080@30 both work with
  MediaPipe. The IROS pipeline used 60 fps; matching that avoids
  re-tuning the temporal filter cutoffs.
- **Extrinsic calibration:** ZED-to-torso_link. Checkerboard + one
  session with the robot standing still gives it. Store in a fixed
  YAML in `data_pointers.md`.
- **Depth policy retune:** the iPhone `relaxed-limb` thresholds may not
  transfer. Run `../experiments/scripts/depth_policy_matrix.py` on the
  first ZED capture and either confirm or re-freeze.

**Ask advisors:** just confirm the capture format choice; the rest is
mechanical.

## 1. X1's efficiency-comparison anchor (blocks C2 reporting)

`PLAN.md` §Contributions says X1 should "report efficiency measures matching the IROS ones (execution time, trajectory length, two-arm mechanical work)". Two ways to close this:

- **(a) Run X1 on the IROS pick-and-stack sequence exactly** — use `../iros_pilot/s2_bricklay.py`'s task spec so the generated motion can be compared against the hand-designed native motion on the same axes. Preserves like-for-like.
- **(b) Report M1/M2/M3 efficiency independently** — the IROS comparison stays in the workshop paper, and ICRA argues its own numbers.

The preview run of X1 on the OLD captures produced `center_len ≈ 0.00-0.02 m` per rep, which is nonsense because the retargeter treats each hand independently and M1/M2/M3 do not have both hands on one object. This suggests (a) is the honest read — the two-arm mechanical work / center-length metrics assume a pick-and-stack-like task.

**Ask advisors:** which path? If (a), we also need to reproduce the IROS s2 comparison in this repo for consistency. If (b), replace `two_arm_center_length` with a per-arm metric (e.g., sum of wrist-tip trajectory length across both arms).

## 2. RA-L fallback decision (blocks the submission-venue commitment)

`PLAN.md` §Timeline: "RA-L takes rolling submissions, is a journal, and can carry a conference presentation slot. Given a refilm at the front of a five-week window, RA-L may be the better fit rather than the consolation. Decide with the advisors by Aug 20."

**Ask advisors:** ICRA-with-tight-timeline or RA-L-with-more-room? The scaffolding is neutral to venue.

## 3. Refilm session logistics (blocks everything downstream)

- iPhone tripod position, framing, three-quarter view — reuse the 0713 setup exactly? Or improve it?
- Reach-extent sweep: 4 or 5 target distances, where placed, how marked on the floor?
- Optional lateral / cross-body reach — worth including in the same session or defer?

**Ask advisors:** the plan says one session covers M1/M2/M3 + reach-extent; confirm the shot list before booking the space.

## 4. F-c reconciliation (blocks writing the intro)

IROS submission states "both conditions reached all four motion targets in all three trials." ICRA has to present the hand-separation / block-size evidence AND explain why that is not a contradiction with the IROS claim.

Draft framing: the IROS "targets reached" was a geometric gate (block-center visits within 8 cm with dwell) that both strategies pass. The physical stack requires two hands with a separation compatible with the block; the fidelity retargeting produces 51.8 cm hand separation around a 23.8 cm block, so a physical block would drop.

**Ask advisors:** is this framing safe, or does the plan want a stronger construction — e.g., re-running the IROS pilot with actual physical blocks and reporting the failure count?

## 5. B2 approach-direction error: contact-event vs rep-mean

The plan cites "~12 degrees approach-direction error at contact events" for B2. The preview code reports a rep-mean of ~40 degrees because it has no contact-event annotations. Two options:

- **Manually annotate contact events** in M1/M2/M3 (hammer strike frame for M1, block placement frame for M2, ground-contact frame for M3) and report B2 approach error over just those windows.
- **Automatically detect contact events** via forearm deceleration + wrist-y minimum. Simpler but harder to defend as ground truth.

Ping when the refilm is done and revisit; the answer depends on how clean the refilmed captures are.

## 6. Gravity-torque model

`x1_generated_across_motions.py` uses `sum |dq| * effort` as a mechanical-work proxy because a real gravity model needs link inertia parsing. See `notes/decisions.md` for revisit conditions.

**Deferred until:** X1 numbers become paper numbers (post-refilm).
