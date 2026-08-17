# X2 cost-of-fidelity result (v1)

Run 2026-08-05 via `x2_costoffidelity.py` on M2/M3 reps 1-3, right arm.
Fidelity config = clean B1 retarget (yaw hack ON, flip-free). Task-centric =
X6 causal solve on the same EE target. Task achievement held fixed.

Mean over reps:

| metric | fidelity | task-centric | cost of fidelity |
| --- | --- | --- | --- |
| manipulability (mean Yoshikawa) | 0.0247 | 0.0248 | +1% (negligible) |
| gravity load (mean, sum abs tau) | 10.7 Nm | 10.7 Nm | ~0 Nm |
| joint-limit margin (min, rad)    | 0.061   | 0.088   | 31% closer; fidelity crosses the limit in M2_3 (-0.001) and M3_1 (-0.004) |

## Interpretation (honest)

The manipulability/gravity cost is essentially ZERO at these weights, because
the two configurations are nearly identical (see x2_config_figure): the task-
centric objective matches the approach direction EVERY frame, which over-
constrains the arm and leaves almost no null space for the robot-native terms
to exploit. The task-centric solve therefore only fixed the branch/limit
problem. So the MEASURED cost of fidelity is FEASIBILITY (joint-limit margin
and violations), tied to the X6 flip story (the mirror branch parks at the
limit), NOT energy or dexterity.

The plan's target sentence ("X% manipulability, Y Nm additional gravity") does
NOT hold at flip-fixing weights. Do not claim it without the robot-optimal
variant below.

## Fork for the headline (decide before the advisor meeting)

a) ROBOT-OPTIMAL variant: enforce approach direction only at contact EVENTS
   (low EE speed), freeing the swivel null space between events, and let
   R_robot actively maximize manipulability / minimize gravity there. Re-
   measure. Decisive test for whether a manipulability/gravity cost exists at
   all. Either outcome is publishable.
b) REFRAME C3 around feasibility: the limit-margin + violation + flip data
   already supports "fidelity drives the arm to the edge of its workspace."

Recommendation: (a) - it determines the headline and prevents overclaiming a
manipulability cost the data may not support.

## Robot-optimal variant result (2026-08-06, x2_robot_optimal.py)

Fork (a) built and run. Approach direction enforced at contact events only
(low EE speed), swivel null space freed, candidates ranked by robot health
(manip up, gravity down, limit margin up). M2_1 + M3_1 cross-motion:

| metric | fidelity | robot-optimal | cost of fidelity |
| --- | --- | --- | --- |
| manipulability (mean) | 0.0249 | 0.0257 | +3% (small, and weight-dependent) |
| gravity load (mean) | 10.7 Nm | 11.5 Nm | -0.8 Nm (fidelity is LOWER; NOT a cost) |
| joint-limit margin (min) | 0.043 rad | 0.107 rad | ~60% less margin; fidelity CROSSES the limit on M3_1 |

The configs now genuinely differ (x2_robotopt_figure: fidelity pulls the elbow
in, robot-optimal keeps it out).

## VERDICT (settles the headline)

Even with the null space freed and the solver optimizing robot health, the
manipulability cost of fidelity is only ~3-5% and the gravity "cost" reverses
sign. The ONLY large, robust, mechanistically-grounded cost of fidelity is
JOINT-LIMIT MARGIN: reproducing the human configuration runs the arm at roughly
one third to one half the joint-limit margin of a task-optimal configuration for
the same task, and drives it into limit violations. This is the same mechanism
as the X6 flips (the mirror branch parks at the limit).

Lead C3 / X2 on FEASIBILITY (limit margin + violations + flips). Report
manipulability as a small secondary effect. DROP the gravity claim entirely
(the plan's "Y Nm additional gravity load" is not supported; fidelity is if
anything slightly lower). Leading on limit margin is also safer because it is
weight-robust, whereas the manipulability/gravity numbers depend on the R_robot
weighting, a modeling choice a reviewer can poke.

Honest target sentence: "reproducing the worker's configuration leaves the arm
at ~40-50% of the joint-limit margin available to a task-optimal configuration
for the identical task, and drives it into limit violations, for no task
benefit; this is the same ill-conditioning that produces the branch flips."

## 6-rep firm-up (2026-08-07, full M2/M3 x reps 1-3)

| rep | manip mean f/ro | gravity mean f/ro (Nm) | limit margin min f/ro (rad) |
| --- | --- | --- | --- |
| M2_1 | 0.0269/0.0280 | 11.9/12.6 | 0.089/0.179 |
| M2_2 | 0.0248/0.0258 | 12.3/12.4 | 0.023/0.162 |
| M2_3 | 0.0259/0.0273 | 12.2/12.7 | -0.001/0.154 |
| M3_1 | 0.0230/0.0234 | 9.5/10.3 | -0.004/0.035 |
| M3_2 | 0.0234/0.0237 | 9.1/10.0 | 0.083/0.183 |
| M3_3 | 0.0242/0.0242 | 9.0/10.0 | 0.173/0.093 |
| mean | 0.0247/0.0254 (+3%) | 10.7/11.3 (-0.7) | 0.061/0.134 (+55%) |

Confirms the verdict at 6 reps: manipulability cost of fidelity ~3% (small, weight-dependent), gravity is not a cost (fidelity ~0.7 Nm lower), joint-limit margin is the large effect (mean +55%). NUANCE for honesty: (1) fidelity VIOLATES a joint limit on 2 of 6 reps (M2_3 -0.001, M3_1 -0.004); the task-optimal config violates on 0 of 6. This binary is the most robust claim. (2) The mean margin gap is not monotonic: on the easy rep M3_3, where fidelity already had ample margin (0.173), the robot-optimal solve traded some margin back for manipulability/gravity (0.093), so it is slightly worse there. So lead on the VIOLATION COUNT (2/6 vs 0/6) plus the mean margin, not a per-rep margin guarantee. Revised target sentence: "reproducing the worker's configuration drives the arm into joint-limit violations on 2 of 6 captures and leaves it, on average, at ~45% of the joint-limit margin of a task-optimal configuration for the identical task, for no task benefit; this is the same ill-conditioning that produces the branch flips."
