# B2 control result (is C2 a method?)

Run 2026-08-07 via `b2_control.py` on M2/M3 reps 1-3, right arm. B2 = textbook task-space DLS position IK + null-space joint-limit avoidance (Liegeois), causal warm-start, no fidelity term, no R_robot, no yaw hack, no learning.

| rep | fidelity flips (yaw hack off) | B2 flips | B2 pos err | B2 approach-dir err at events |
| --- | --- | --- | --- | --- |
| M2_1 | 10 | 1 | 0.21 cm | 15.7 deg |
| M2_2 | 10 | 2 | 0.20 cm | 6.3 deg |
| M2_3 | 12 | 2 | 0.21 cm | 10.2 deg |
| M3_1 | 0 | 2 | 0.20 cm | 12.8 deg |
| M3_2 | 7 | 2 | 0.20 cm | 12.9 deg |
| M3_3 | 8 | 2 | 0.21 cm | 13.2 deg |

## Verdict

1. Generic task-space IK LARGELY removes the flips: 10-12 per rep under fidelity drops to 1-2 under B2. The flip fix is the OBJECTIVE (position target + null-space robot-native resolution instead of human-direction matching), a known idea. So C2 as a novel flip-fixing SOLVER does not survive: a textbook baseline mostly does it. (Note B2 leaves 1-2 residual flips and even adds ~2 on the clean M3_1, from null-space/warm-start swivel jumps; our task-centric solve was exactly 0, marginally cleaner, but that gap is small and not a headline.)

2. B2 sacrifices TASK-RELEVANT ORIENTATION: mean ~12 deg approach-direction error at contact events, because position-only IK does not constrain the strike/place direction. Our task-centric formulation enforces orientation at events (and frees it between), keeping it small while still avoiding flips. THAT is C2's surviving contribution: task-relevant tolerance, not a new solver.

## Consequence for the paper

The paper is C1 (the argument that fidelity is the wrong objective) + C3 (the feasibility cost of fidelity) + C4 (replica vs replacement). C2 is reframed honestly as a task-tolerance-aware instantiation of task-space IK, not a novel method; do not oversell a new IK. Learning stays out unless X5 latency forces it. This is exactly the plan's anticipated outcome for "generic task-space IK also removes the flips."
