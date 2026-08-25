# ICRA 2027 plan (canonical, verbatim from user 2026-08-25)

The text below is the plan as delivered by the user on 2026-08-25 after the
IROS 2026 workshop submission (see `../iros_pilot/S2_STATUS.md` and the
attached IROS_WS_FoC.pdf). It is copied here without edits so future Claude
sessions can read the source, not a summary. Numeric claims cited as
"done" (X6/X2/B2 results, RULA numbers, IROS efficiency numbers) come from
work outside this repo — the icra2027 folder was cleared for a fresh start
before this plan was written.

If this file and README.md disagree, the plan is authoritative — **except**
for the addendum below, which the user provided after the plan text and
which supersedes any conflicting section.

## Addendum: 2026-08-25 sensor pivot

Human capture switches from iPhone LiDAR on tripod (IROS) to a ZED depth
camera mounted at the H1-2 neck (ICRA and onward). User confirmations:

- **Skeleton source:** ZED RGB + depth → MediaPipe (reuse the existing
  `bag_to_joints.py` pipeline; only the input reader changes).
- **Camera motion:** robot standing still during capture; ZED-to-base
  extrinsic is fixed for the session.
- **Pick-and-stack (F-c evidence):** re-capture with the ZED so the
  paper does not mix sensors.

Implications:
- The plan's "Sensing discipline: iPhone = human capture, ZED = robot
  perception" is superseded. ICRA numbers are ZED-only.
- Any cross-reference against the IROS pick-and-stack must either
  acknowledge the sensor mismatch or reproduce the IROS motion on the ZED.
- Track C's D1 (ZED-to-base extrinsic) is no longer optional; it is a
  prerequisite for interpreting keypoint depth in the base frame.
- The `--depth-policy relaxed-limb` thresholds were tuned on iPhone
  LiDAR confidence and must be re-evaluated on the first ZED capture.

Repo scaffolding accepts this by routing all drivers through
`src/common/data_paths.py::ACTIVE_DATA_ROOT` (single edit to switch).

---

## What the IROS submission changed

The submitted paper is the pick-and-stack comparison alone, framed on productivity. The error budget is gone: no depth anisotropy, no gravity sag, no branch flips, no screening study. RULA survives only as a manipulation check confirming the two strategies differ posturally as intended. That is a cleaner paper, but ICRA now inherits something different from what the old plan assumed.

**C1 is now largely made in IROS.** The submitted abstract already argues human-derived transfer "may not fully define how a humanoid should ultimately execute the transferred motion," and the discussion already asks what should be retained from a demonstration and what can be left open. That was the planned ICRA headline. It is now the workshop paper's conclusion, so ICRA cannot lead on it.

**F-a is unpublished, which helps.** The branch flips never appeared in IROS. ICRA can present them fresh rather than self-citing.

**F-c has a hole.** The embodiment-scale finding was cut, and the submitted paper now states both conditions "reached all four motion targets in all three trials." Nothing in it says the retargeted motion could not physically stack. ICRA must present the hand-separation and scale evidence from scratch and make the geometric-gate versus physical-stacking distinction explicit, or a reader who has seen both papers will read them as contradicting each other.

**The submitted limitations section writes the ICRA paper.** It says Robot-Specific Motion "was hand-designed rather than generated through optimization, so the current comparison does not establish the range or magnitude of efficiency gains achievable." That is the delta: automated generation replacing hand design, evaluated across motions rather than one sequence, with a mechanism explaining why the inherited posture costs something.

---

## Thesis

IROS asked whether departing from the demonstration helps and answered it with one hand-designed motion on one sequence. ICRA answers it systematically. Robot-Specific Motion is generated rather than hand-designed, evaluated across the three captured construction motions, and the efficiency difference is traced to a mechanism: the inherited posture leaves the arm near its own joint limits, which costs margin, produces solver failures, and in the limit produces motion the robot cannot execute. Fidelity is a mode you select, not a default.

**Guardrail.** Do not claim existing retargeting ignores the robot. It does not. The claim concerns which term is the objective and which is the constraint. The field optimizes fidelity subject to robot feasibility; we optimize task achievement subject to robot health. The flip is licensed by the application.

---

## Contributions

**C2 (primary, rehabilitated). Automated Robot-Specific Motion generation.** IROS hand-designed one motion for one sequence and said so in its limitations. ICRA generates it: task achievement subject to robot-native regularizers, no posture-fidelity term, applied to M1, M2, and M3. B2 established that the flip fix itself is generic, so the claim is not a novel solver. The claim is that the generation is automatic, that it applies across motions, and that it carries task-relevant tolerance where generic position-only IK loses about 12 degrees of approach direction at contact events.

**C3 (primary). The mechanism.** Reproducing the worker's configuration leaves the arm at roughly 45 percent of the joint-limit margin available to a task-optimal configuration for the identical task, and drives it into limit violations on 2 of 6 captures against 0 of 6. The same ill-conditioning produces the branch flips. This explains the IROS efficiency result rather than merely repeating it. Lead on the violation count, which is binary and weight-robust; report mean margin second. The per-rep margin gap is not monotonic (on M3_3 the task-optimal solve traded margin for manipulability), and say so.

**C4. Replica versus replacement.** Fidelity is the goal in replica mode (training, hazard demonstration, legibility) and a cost in replacement mode. Naming the modes turns the argument from criticism into guidance.

**C1, demoted.** The argument itself is IROS's. ICRA restates it in one paragraph, cites the workshop paper, and moves on.

**Dropped:** the gravity cost claim, manipulability as a headline (3 to 5 percent is within weight sensitivity), the learned retargeter, and the branch-consistency argument for learning.

---

## Status: three results, three deletions

X6, X2, and B2 are done, and each narrowed the paper.

**X6 killed the learning contribution.** The task-centric objective removes all 47 branch flips with a plain causal single solve, matching the expensive multi-start. There is no non-causal global solution that only distillation could deliver, so branch consistency no longer justifies a learned student.

**X2 killed the energy claim.** Fidelity gives up about 3 percent manipulability and its gravity load is if anything lower. The cost is feasibility.

**B2 killed the novel-solver claim.** Generic textbook DLS position IK with null-space limit avoidance also removes most flips (10-12 per rep down to 1-2), though at about 12 degrees of approach-direction error at contact events.

**Consequence: this is an analysis paper with an automated-generation component, not a method paper.** No learning, no new solver.

---

## Notation

**X = experiments.** Numbering has gaps because X4 was dropped and X5 demoted. Do not renumber; scripts and result files use these names. **B = baselines.** **C = contributions.** **F = the three failures** that form the mechanism argument.

### The three failures, one cause

| Failure | Evidence | Source | IROS status |
| --- | --- | --- | --- |
| F-a. Branch flips | 47 flips across 6 reps under fidelity, 0 under task-centric | X6 | unpublished, claim fresh |
| F-b. Joint-limit violations | 2 of 6 captures violate a limit under fidelity, 0 of 6 task-optimal | X2 | unpublished, claim fresh |
| F-c. Physical task failure | Hand separation 51.8 cm around a 23.8 cm block; the geometric gate passes, the physical stack does not | IROS pilot data, cut from the paper | **must present from scratch and reconcile with the submitted "all four targets reached"** |

The reviewer objection to expect: you compared naive direction-matching against task-space IK with null-space optimization and found the latter has better limit margin, which is what null-space optimization is for. The defense is that B1 is not a strawman but the field's default, since H2O, HumanPlus, OmniH2O, OKAMI, and Liu and Liu all track human configuration or keypoints. The version that clears the bar is not "task-space IK is better" but "the inherited posture is what makes the retargeted motion inefficient, and in the limit infeasible, and here is the mechanism."

---

## Formulation

Baseline (IROS Human-Derived Motion, B1):

```
q_t = argmin_q || d_robot(q) - d_human(p_t) ||^2 + smoothness
      subject to joint limits
```

Matches configuration. The yaw residual gradient is shallow in forward-reaching poses, which is where the flips originate.

Proposed (generated Robot-Specific Motion):

```
q_t = argmin_q  L_task( FK(q), x_t ) + lambda * R_robot(q, q_{t-1})
```

No posture-fidelity term. L_task carries task-relevant tolerance including approach direction at contact events. R_robot references no human: joint-limit avoidance, manipulability, gravity load, temporal continuity, velocity limits. Report weight sensitivity honestly, since only limit margin is weight-robust.

---

## Motion data

**The captures are being refilmed.** The Human-Derived Motion condition needs a clean redo. Treat this as the schedule's first hard dependency, since X1, X3, and X7 all consume it.

| Dataset | Content | Feeds | Status |
| --- | --- | --- | --- |
| M1 overhead hammering | 3 reps, iPhone tripod | The control. Zero flips, task-dependence evidence | **refilm** |
| M2 bricklaying | 3 reps, iPhone tripod | Primary flip case. X1, X3 | **refilm** |
| M3 lifting | 3 reps, iPhone tripod | Second flip case. X1, X3 | **refilm** |
| Reach-extent sweep | Same forward reach at 4-5 target distances, 3 reps each | C3 as a curve rather than a count | **capture in the same session** |
| Pick-and-stack | Human capture plus hand-designed robot-native, 3 hardware trials each | IROS result. F-c evidence lives in this data | done, published |
| Replay logs | 42 logs including pilots and failures | Hardware reference for X3 | done |
| Lateral / cross-body reach | 3 reps | Is the failure axis forward reach or arm extension generally? | optional |

The refilm and the reach-extent sweep should happen in one session, since the setup is identical: iPhone on tripod, three-quarter view, feet planted, torso upright, no leaning, since the retargeter transfers arms only. Reprocessing after the refilm means X6 and X2 must be re-run on the new captures before their numbers can be reported. Budget that.

C3 currently rests on 2 of 6 captures, which a reviewer will call thin. The reach-extent sweep turns "fidelity sometimes violates joint limits" into "violation rate rises with reach extent, and here is the curve," which is a mechanism rather than an anecdote.

**What not to capture.** More construction task variety. A fourth representative task adds breadth without adding evidence. ICRA should vary the quantity that causes the failure.

---

## Experiments

| ID | Question | Supports | Status |
| --- | --- | --- | --- |
| X6 | Does the objective flip fix the flips without learning? | F-a | done Aug 5, **re-run on new captures** |
| X2 | What does fidelity cost, task held fixed? | C3, F-b | done Aug 7, **re-run on new captures** |
| B2 | Is the formulation a method, or does generic IK suffice? | C2 scope | done Aug 7 |
| X1 | Does generated Robot-Specific Motion work across all three motions? | **C2** | remaining, now primary |
| X3 | Does it hold on the real robot? | C2, C3 | remaining, **non-negotiable** |
| X7 | Can both modes be demonstrated on one motion? | C4 | remaining |
| X5 | Can the causal solve hold 50 Hz deployed? | deployment note | demoted, not a gate |
| X4 | Noise robustness | (was training aug.) | dropped |

**X6 result** (to be reproduced on the refilmed captures). M2 and M3, reps 1-3, right shoulder-yaw. Task held identical, only redundancy resolution changed. Flip defined as a single-frame yaw jump above 6 rad/s. 47 flips to 0 with sub-2 mm tracking, so "no flips" is not won by dropping the task. The causal single solve matched the expensive multi-start, so it is the objective and not the solver that fixes them. Solver unoptimized at about 37 ms per frame. `x6_taskcentric.py`, `x6_results.md`.

**X2 result** (6 reps, to be reproduced). Mean manipulability +3 percent, gravity -0.7 Nm (not a cost), limit margin +55 percent with fidelity at about 45 percent of task-optimal, and fidelity violates a joint limit on 2 of 6 captures against 0 of 6. The side-by-side configuration figure shows fidelity pulling the elbow in while robot-optimal keeps it out at the same wrist target. `x2_costoffidelity.py`, `x2_robot_optimal.py`, `x2_results.md`.

**B2 result.** Generic DLS position IK plus null-space limit avoidance drops flips from 10-12/rep to 1-2/rep but costs about 12 degrees of approach-direction error at contact events. `b2_control.py`, `b2_results.md`.

### Why each remaining experiment exists

**X1 is now the primary experiment, not cheap insurance.** It generates Robot-Specific Motion automatically for M1, M2, and M3 and shows it works across all of them. That is C2, and it is what IROS explicitly did not do. Report efficiency measures matching the IROS ones (execution time, trajectory length, two-arm mechanical work) so the generated motion can be compared against the hand-designed baseline the workshop paper reported.

**X3 cannot be cut.** Everything else is simulation or offline analysis. It is not a repeat of IROS, which compared only on pick-and-stack; X3 runs M2 and M3 on hardware. The specific thing to check: do the X2 limit violations actually manifest during replay, or does the controller clamp them silently? Either answer matters and the second needs reporting.

**X7 makes C4 concrete.** Without it, replica versus replacement is argued rather than demonstrated. Same motion, both modes: fidelity lands in the human's posture band, generated motion does the work from a configuration no worker could adopt.

**X5 demoted.** No longer a decision gate, since learning is out. The causal solve at 37 ms per frame is 27 Hz, and reaching 50 Hz is solver optimization rather than a research question. Report as a deployment note.

---

## Baselines

**B1** the IROS Human-Derived Motion retargeter, hardware-measured and the field's default. **B2** generic task-space IK with null-space limit avoidance, done. **B3** an external published method adapted to H1-2, stretch goal. **B4** learned retargeter, dropped.

---

## Traceability

| Claim | Evidence | Gap |
| --- | --- | --- |
| C2 automated generation across motions | X1, plus B2 for the tolerance argument | **X1 not yet run, and it needs the refilmed captures** |
| C3 the feasibility mechanism | X2, X6 | Thin at n=6. The reach-extent sweep fixes this |
| C4 replica vs replacement | Argued from the IROS RULA finding | Needs X7 to be demonstrated |
| F-c physical task failure | Pick-and-stack hand-separation data, unpublished | Must reconcile with the submitted paper's "all four targets reached" |

---

## Related work

T1. Liu and Liu, CACIE 2026 (already cited as [9] in the IROS submission). Humanoid learning construction tasks from worker demonstrations, teacher-student whole-body policy, 82.45 mm MPJPE. **Read before writing related work.** They retarget for fidelity and train a policy to track the retargeted reference, which is exactly what this paper argues against.

T2. The residual-learning cluster (ASAP, RobotDancing, SENTINEL). Not a threat to the thesis, but a hard boundary on what may be claimed.

The gap: ergonomics-aware robotics (DULA/DEBA, NeuroErgo, REBA-based planning) keeps the human in the loop while the robot adapts to reduce the human's load. Nobody uses posture in a setting where the robot replaces the human. That is what C4 occupies.

---

## Track C: the ZED node, off the critical path

With learning out and X5 demoted, the real-time node gates nothing in this paper. It becomes journal and GR00T infrastructure. The one piece worth doing while the hardware is out is D1, the camera-to-base extrinsic, since a hidden extrinsic error masquerades as retargeting error and it is cheap to measure now.

Sensing discipline: the **iPhone on a tripod is human capture**, and the **ZED at neck level is robot perception**, egocentric and moving with the torso. Never place a number from one against a number from the other.

---

## Timeline

Five weeks to Sep 15, with a refilm and full reprocessing at the front. This is tighter than the previous version because the captures are being redone.

- **Aug 11-15.** Advisor conversation. Refilm M1, M2, M3 plus the reach-extent sweep in one session. Begin reprocessing.
- **Aug 15-22.** Reprocess captures. Re-run X6 and X2 on the new data and confirm the numbers hold. Build X1 (generated motion for all three).
- **Aug 22-29.** X1 complete. X3 hardware sessions.
- **Aug 29 to Sep 5.** X3 completion, X7, figures.
- **Sep 5-15.** Write and submit.

**The refilm is the critical path.** If it slips past Aug 15, everything downstream compresses against a fixed deadline, and the pressure valves apply immediately.

**Pressure valves, in order:** drop B3; drop the reach-extent sweep and report C3 at n=6 with the thinness stated; drop X7 and make C4 argued rather than demonstrated; report single subject as a stated limitation. **Do not drop X3.**

**Fallback.** RA-L takes rolling submissions, is a journal, and can carry a conference presentation slot. Given a refilm at the front of a five-week window, RA-L may be the better fit rather than the consolation. Decide with the advisors by Aug 20.
