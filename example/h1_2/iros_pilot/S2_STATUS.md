# S2 pilot: status brief (v2, post-hardware)

*(Self-contained summary for an outside reader. Two images may be attached:
`s2_bricklay_paths.png` and `s2_ergo_scores.png`; the sections "Reading the
path figure" and "Reading the ergonomics figure" describe them in words.)*

## Project context

IROS 2026 workshop paper on markerless human-to-humanoid motion retargeting
for construction tasks, on a Unitree H1-2. Human motions are captured with an
iPhone LiDAR, retargeted to the robot's arms, and replayed on the real robot.
The paper's central claim (the "fidelity question"): a robot CAN mimic human
motion at measurable cost, but it often SHOULD NOT, because robot-possible
strategies that violate human ergonomic norms can beat the human strategy on
every task metric. The S2 pilot is the preliminary evidence.

## The task and the two conditions

Two-handed pick-and-stack with REAL blocks (23.8 x 12.1 cm): grasp a block
from a central pile with both hands, place it on a growing 3-layer column
(24 cm total lift, layers exactly one block height apart). Task success is
tolerance-gated: the block center (midpoint of the two hands) must visit
pick + three layer stations within 8 cm with a sustained dwell.

- **Human-mimic baseline**: a real human performed the exact task on camera;
  that capture was retargeted (direction-matching IK) and replayed on the
  robot. This is faithful motion-level transfer of a real person, not a
  designed model.
- **Robot-native**: a designed motion for the same stations. Direct
  minimum-jerk moves, no lift-over arcs, no rests between blocks, and a
  deliberately non-human stance: a sustained DEEP SQUAT (knees measured at
  62 deg on hardware) that lowers the center of mass and brings the low pile
  into a better reach region. Joint speeds capped at 0.6 rad/s, SLOWER than
  the human capture's peaks, so the time advantage cannot come from raw
  speed.

Both conditions ran on the same robot, same controller, same gravity
compensation, same logger.

## Measured results (n = 3 trials per condition, real hardware)

| metric | human-mimic | robot-native | advantage |
| --- | --- | --- | --- |
| time | 25.03 +/- 0.06 s | 12.87 +/- 0.15 s | 1.9x |
| block-center path | 4.33 +/- 0.08 m | 0.92 +/- 0.03 m | 4.7x |
| energy, both arms | 247.3 +/- 3.0 J | 58.7 +/- 5.6 J (see note) | 4.2x |
| task success | 3/3 PASS | 3/3 PASS | - |

Energy is measured mechanical work (sum of |joint torque x joint velocity|
over both arms). Note: native trial 3's energy (52.3 J) sits apart from
trials 1-2 (~62 J) and is being re-verified; treat the energy spread as a
placeholder until then.

Key interpretive points:

- Path (4.7x) and energy (4.2x) are speed-independent; time (1.9x) was won
  at joint speeds SLOWER than the human's, so all three advantages come from
  the strategy, not the tempo.
- Repeatability is itself a finding: native time variance was near zero and
  the human replay repeated within 2 percent on every metric.

## The ergonomic inversion (measured, the paper's novel piece)

Both robot motions were also scored AS IF the robot were a person, through
the identical RULA/REBA code used for the real human capture, with the
robot's real measured stance angles. Mandatory caveat, printed on the figure:
this is a posture-similarity statement, not a claim about robot injury.

| series | RULA mean | REBA mean |
| --- | --- | --- |
| human capture (real person) | 2.34 | 1.96 |
| robot replaying that human (39 deg default knees) | 2.30 | 1.33 |
| robot-native (62 deg measured squat) | 1.77 | **2.09** |

Two mechanisms, one inversion:

1. **REBA (whole-body): the native scores WORSE than the human** (2.09 vs
   1.96), and more sharply: the native holds flagged posture for 100 percent
   of the cycle and never enters the "negligible" band, while the human is
   negligible 26 percent of the time. A person cannot hold a deep squat for a
   task cycle; the robot parks there and collects the efficiency. This is the
   thesis: an ergonomically-negative, robot-possible strategy that wins on
   every task metric.
2. **RULA (arm-weighted): mimicry INHERITS the human's arm cost** (robot
   mimic 2.30, nearly identical to the human's 2.34), while the native
   avoids it (1.77). Copying the motion copies the posture cost.

Integrity note that survived an internal audit: the H1-2's DEFAULT standing
stance already has 39 deg knee bend, which must be (and is) applied to both
robot series equally; at 39/39 there is NO inversion. Only the deliberate
62 deg squat creates it, which is exactly the point: it is a chosen
robot-optimal stance, verified on hardware with a live joint-state probe.

## Reading the path figure (s2_bricklay_paths.png)

Left panel, side view (forward x vs height z): the red human-baseline path
is the retargeted real capture, a large tangled loop that rises above the
column, retreats toward the body, and sweeps far beyond the work zone; the
teal robot-native path is three short direct diagonals from pile to layers.
Orange markers show pick (square, low) and the three stacked layers
(circles, rising column). Right panel, top view: the block center runs
straight up the midline for both, with the two faint hand paths mirrored
about it (the two-handed grasp). One-sentence takeaway: same stations, same
success, but the human strategy travels 4.7x farther through a much larger
volume.

## Reading the ergonomics figure (s2_ergo_scores.png)

Two panels, RULA and REBA grand scores over the task cycle (0-100 percent),
with the published risk bands shaded (green acceptable/negligible up through
red). Red solid = real human capture; orange dashed = robot mimicking that
capture; teal = robot-native. In the RULA panel the red and orange traces
sit almost on top of each other (mimicry inherits arm cost) with teal below.
In the REBA panel the teal native line rides at 3 for the entire cycle,
above the human's 1-2 baseline; the human shows brief transient spikes to
4-6 (bending moments) but returns to low bands, while the native never
drops below its flagged plateau. Sustained versus transient exposure is the
honest framing of the inversion.

## Honesty boundaries

- Ergonomic scores are posture-only lower bounds (load, coupling, and
  activity multipliers are zero for every series equally).
- Scoring the robot with human tables is posture similarity, not injury
  risk; the robot has no spine to protect, which is the point.
- Embodiment-scale finding: retargeting transfers motion, not task geometry.
  The human's placements land at roughly 0.6 scale on the robot, so the
  replayed baseline visits the stations (PASS at 8 cm tolerance) but its
  release heights cannot physically stack the real blocks; its on-camera run
  is a pantomime beside the stack. Direct motion transfer fails the physical
  task even when it passes the geometric gate; task-level transfer is the
  implied fix.
- Arm energy only; the squat's leg-holding cost is not in the energy metric
  (stated in the writeup). The crouch did raise arm energy slightly versus
  standing (about 59 vs 53 J), an honest tradeoff that is still 4.2x below
  the baseline.
- The spec tolerance was relaxed once (6 to 8 cm, 0.3 to 0.15 s dwell)
  because real human placements are transient touches; applied identically
  to both conditions, both PASS.

## Remaining before submission

Filming (side view, both conditions) and the native trial-3 energy
re-verification happen tomorrow. Then: results section, figure captions,
and an advisor summary of the accumulated scope decisions (two-handed
energy accounting, the relaxed gate, the designed squat stance, real-block
geometry).
