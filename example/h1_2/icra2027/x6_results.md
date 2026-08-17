# X6 result (Phase-0 critical ablation)

Run 2026-08-05 via `x6_taskcentric.py` on M2 (bricklaying) and M3 (lifting),
reps 1-3, right shoulder-yaw. Flip = single-frame |dyaw|/dt > 6 rad/s.

Three conditions, TASK held identical (same EE position + approach target from
the clean B1 retarget), only redundancy resolution differs:
- fidelity : B1 direction-matching, greedy, yaw hack OFF (the failure mode).
- TC-greedy: task-centric objective (EE pos + approach + R_robot: limit margin,
  manipulability-ranked, continuity), CAUSAL single warm-started solve, no
  global fallback, no learning.
- TC-multi : same objective, multi-start global solver.

| rep  | fidelity flips (pk rad/s) | TC-greedy flips | TC-multi flips | TC task err |
| ---- | ------------------------- | --------------- | -------------- | ----------- |
| M2_1 | 10 (16.4)                 | 0               | 0              | 0.17 cm     |
| M2_2 | 10 (9.5)                  | 0               | 0              | 0.17 cm     |
| M2_3 | 12 (11.7)                 | 0               | 0              | 0.18 cm     |
| M3_1 | 0  (4.2)                  | 0               | 0              | 0.12 cm     |
| M3_2 | 7  (9.5)                  | 0               | 0              | 0.12 cm     |
| M3_3 | 8  (9.3)                  | 0               | 0              | 0.12 cm     |

47 fidelity flips across the reps -> 0 under both task-centric conditions,
with sub-2 mm EE tracking (so "no flips" is not won by dropping the task).

## Interpretation: OUTCOME 3 (leaning)

The flips are caused by the OBJECTIVE (matching the human configuration), not
by a weak solver. TC-greedy == a plain causal single solve == already 0 flips,
so the fix is causal and cheap; no global search and no learning are needed for
branch consistency. This is the plan's Outcome 3: be willing to publish a
method-and-analysis paper with no learning.

CAVEAT (must resolve before claiming "cheap online"): TC-greedy is an
unoptimized scipy L-BFGS-B with numerical gradients, ~30 s for ~800 frames
(~37 ms/frame). That is not yet 50 Hz. X5 (latency, with an analytic Jacobian)
must confirm the causal solve hits deployment rate. If it cannot, learning
re-enters ONLY as distillation-for-speed (Outcome 2). Either way the OBJECTIVE
is the fix and learning is not needed for correctness.

## Consequences for the paper

- C1 (posture fidelity is the wrong objective): STRONGLY supported, hardware-
  derived. Lead with this. Survives regardless of learning.
- Learning (C2's learned instantiation): DEMOTED. Justifiable now only via
  latency (needs X5) or noise robustness (X4). Do not build the teacher-student
  pipeline yet.
- B2 THREAT RISES: TC-greedy is essentially task-space IK with null-space
  resolution (= B2). If generic B2 also kills the flips (likely, given this),
  then the novel-formulation framing of C2 is thin. The contribution is then
  C1 (argument) + C3 (cost-of-fidelity) + C4 (replica vs replacement), not a new
  solver. Run B2 next to measure any delta vs TC.
- X2 / C3 (cost of fidelity) becomes the headline: both fidelity and task-
  centric configs for the same task now exist, so quantify the robot-side cost
  (manipulability, gravity load, joint-limit margin) the fidelity config pays.

## Next

1. Advisor conversation (Francis + Cho): present this result before committing
   six weeks. The objective flip fixes the flips without learning - reshapes the
   paper toward argument + analysis.
2. X5 latency (analytic Jacobian) to settle Outcome 2 vs 3.
3. B2 control (generic task-space IK) to settle whether C2 is a contribution.
4. X2 cost-of-fidelity (C3) as the new headline.
