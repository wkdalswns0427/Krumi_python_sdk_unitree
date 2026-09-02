# Open questions (for advisor conversations)

Prioritized by how much the answer changes downstream work. The plan's
timeline puts the advisor conversation Aug 11-15; the first three should
happen there.

## 0. ZED capture pipeline details — RESOLVED 2026-08-26

Sensor pivot confirmed 2026-08-25; the two blocking decisions were settled
2026-08-26 (see `notes/decisions.md`).

- **Skeleton source: RESOLVED.** ZED SDK body tracking, BODY_38, via
  `src/zed/zed_to_joints.py`. Not MediaPipe. This also supersedes the
  2026-08-25 decision entry.
- **Capture format: RESOLVED.** SVO2, one file per capture, recorded while
  streaming (`zed_to_joints.py --record <capture>.svo2 --out <joints>.csv`).
- **Depth policy retune: VOID.** No per-keypoint depth sampling happens
  under SDK body tracking, so no policy applies. `depth_policy_matrix.py`
  also does not exist in `../experiments/scripts/`.
- **ZED resolution / fps: still open, but mechanical.** HD720@60 matches the
  IROS 60 fps and avoids re-tuning the temporal filter cutoffs. Set it at
  the start of the session and record it in `data_pointers.md`.
- **Extrinsic calibration: still required, Gate 2.** ZED-to-torso_link,
  checkerboard, robot standing still. Store the YAML path in
  `data_pointers.md`.

**Open sanity check for the first capture:** compare BODY_38 wrist-path
smoothness on one rep against the iPhone MediaPipe output at the same reach
before committing the whole session.

## 1. X1's efficiency-comparison anchor (blocks C2 reporting)

`PLAN_v2.md` §Contributions says X1 should "report efficiency measures matching the IROS ones (execution time, trajectory length, two-arm mechanical work)". Two ways to close this:

- **(a) Run X1 on the IROS pick-and-stack sequence exactly** — use `../iros_pilot/s2_bricklay.py`'s task spec so the generated motion can be compared against the hand-designed native motion on the same axes. Preserves like-for-like.
- **(b) Report M1/M2/M3 efficiency independently** — the IROS comparison stays in the workshop paper, and ICRA argues its own numbers.

The preview run of X1 on the OLD captures produced `center_len ≈ 0.00-0.02 m` per rep, which is nonsense because the retargeter treats each hand independently and M1/M2/M3 do not have both hands on one object. This suggests (a) is the honest read — the two-arm mechanical work / center-length metrics assume a pick-and-stack-like task.

**RESOLVED 2026-08-26.** Re-ran X1 on the fixed drivers. The anomaly was
entirely the frozen-target bug, not the midpoint metric:

| capture | centre length, broken | centre length, fixed |
| --- | --- | --- |
| M1_R_1..3 | 0.00-0.02 m | 16.74 / 17.73 / 17.63 m |
| M2_1..3 | 0.00-0.02 m | 5.87 / 6.31 / 6.01 m |
| M3_1..3 | 0.00-0.02 m | 7.17 / 7.24 / 6.63 m |

So `two_arm_center_length` is fine and does not need replacing. The argument
that "(a) is the honest read" rested on the metric looking broken, and that
reasoning is void. Decide (a) versus (b) on its merits instead.

Worth noting for the paper: M1 overhead hammering runs 16 to 18 m of centre
length against 6 to 7 m for M2 and M3, a real motion-class difference the
broken run hid completely.

**Approach-direction term: IMPLEMENTED 2026-08-26.** `solve_taskcentric` now
takes `target_forearm` + `w_approach` and augments the primary task on gated
frames only (`src/common/contact_events.py`, measured by
`src/x10_gated_approach.py`). Gating is the mechanism, not a convenience:
applied on every frame it puts 6 residuals on a 4-DOF arm, the null space
collapses, limit avoidance stands down, and the solve falls back onto the
human's posture. X10's always-on condition shows that boundary on purpose.

Still open: the gate currently comes from automatic wrist-speed-minimum
detection, which is defensible as a heuristic but not as ground truth. See
item 5.

**Ask advisors:** which path? If (a), we also need to reproduce the IROS s2 comparison in this repo for consistency. If (b), replace `two_arm_center_length` with a per-arm metric (e.g., sum of wrist-tip trajectory length across both arms).

## 2. RA-L fallback decision (blocks the submission-venue commitment)

`PLAN_v2.md` §Timeline: "RA-L takes rolling submissions, is a journal, and can carry a conference presentation slot. Given a refilm at the front of a five-week window, RA-L may be the better fit rather than the consolation. Decide with the advisors by Aug 20."

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

  **Updated 2026-08-26:** with the frozen-target bug fixed, B2's rep-mean
  approach error is 22.2 deg (median 18.2, and 17.5 excluding the M2_3
  outlier at 46.2), not the ~40 deg the broken preview reported. That is
  close enough to the plan's 12 deg that contact-event windows plausibly
  close the rest, which makes this annotation worth doing rather than
  worth abandoning.
- **Automatically detect contact events** via forearm deceleration + wrist-y minimum. Simpler but harder to defend as ground truth.

Ping when the refilm is done and revisit; the answer depends on how clean the refilmed captures are.

## 6. Gravity-torque model

`x1_generated_across_motions.py` uses `sum |dq| * effort` as a mechanical-work proxy because a real gravity model needs link inertia parsing. See `notes/decisions.md` for revisit conditions.

**Deferred until:** X1 numbers become paper numbers (post-refilm).
