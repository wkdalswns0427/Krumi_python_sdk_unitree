# Contribution → experiment → evidence traceability

Cross-reference for reviewers and for keeping the paper honest during
writing. Keep in sync with `PLAN.md` §Contributions and §Traceability.

## C2 — Automated Robot-Specific Motion generation

**Delta vs IROS:** IROS hand-designed one motion for one sequence and said so in limitations. ICRA generates it automatically across M1/M2/M3 using task achievement + robot-native regularizers, no posture-fidelity term.

**Evidence:**
- `src/x1_generated_across_motions.py` — runs the generation across all three motions and reports IROS-comparable efficiency measures.
- `src/b2_control.py` — establishes that a generic DLS position IK (no novelty) also removes flips, so the C2 claim is "automated + across motions + carries approach-direction tolerance", not "novel solver".
- `src/common/ik_taskcentric.py` — the objective. Documents that dropping the posture-fidelity term is what fixes the flips.

**What breaks the claim:**
- If X1 does not produce feasible motion on one of M1/M2/M3.
- If the "task-relevant tolerance" (approach direction at contact events) claim cannot be quantified because we do not have contact-event annotations for the refilmed captures.

## C3 — The feasibility mechanism

**Claim (weight-robust, lead with this):** Fidelity retargeting drives the arm into a joint-limit violation on N of 6 captures against 0 of 6 for task-optimal. Secondary (report second): mean limit margin ~45% of task-optimal.

**Evidence:**
- `src/x2_costoffidelity.py` — per-frame fidelity vs task-optimal on the same target.
- `src/x6_taskcentric.py` — flip counts (the same ill-conditioning surfaces as branch flips in the yaw redundancy).
- `src/reach_extent_sweep.py` — curve version of the claim (turns "sometimes violates" into "rate rises with reach extent"). Requires refilm.

**Caveats acknowledged in the plan:**
- Manipulability is NOT a headline; the 3-5% number is within weight sensitivity.
- The per-rep margin gap is not monotonic; on M3_3 the task-optimal solve traded margin for manipulability. Say so.

## C4 — Replica vs replacement

**Claim:** Fidelity is a design choice, not a default. Naming the two modes (replica vs replacement) turns the ICRA finding from a criticism of prior work into guidance for practitioners.

**Evidence:**
- `src/x7_dual_mode.py` — same motion, two IK objectives, RULA-scored side by side. Demonstrates that the choice is discrete (posture bands separate) rather than a spectrum.
- IROS §II.A RULA finding for the replica-side anchor.

**Failure mode:** If both modes land in overlapping posture bands, C4 has to be argued instead of demonstrated. Pressure valve per plan.

## C1 — Task/execution decomposition (DEMOTED)

Made in IROS §V ("what should be retained from human demonstrations, and what aspects of execution can be left open"). Restated in one paragraph of the ICRA paper, with a citation to the workshop paper. Do not lead with this.

## F-a / F-b / F-c: the three failures, one cause

| Failure | Script | Result target | Reviewer risk |
| --- | --- | --- | --- |
| F-a. Branch flips | `x6_taskcentric.py` | 47 → 0 (plan value), n=6 reps | Reviewer will ask about the naive-baseline strawman argument; response in `PLAN.md` §Notation |
| F-b. Joint-limit violations | `x2_costoffidelity.py` | 2/6 vs 0/6 | Weight-robust; safe to lead |
| F-c. Physical task failure | `../iros_pilot/s2_pickstack_handsep.csv` + s2_bricklay.py | Hand separation 51.8 cm vs 23.8 cm block | Must reconcile with the IROS submission's "all four targets reached" — see plan §What the IROS submission changed |

## Anti-claims (do NOT resurrect)

- Novel IK solver claim — killed by B2.
- Learning contribution — killed by X6 (causal single solve suffices).
- Gravity-cost claim — killed by X2 (fidelity gravity load if anything lower).
- Manipulability as a headline — 3-5% is within weight sensitivity.
- Branch-consistency as justification for a learned student.
