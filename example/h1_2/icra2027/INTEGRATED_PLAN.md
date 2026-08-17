# Integrated Research Program Plan

Written 2026-08-02, updated 2026-08-06. Supersedes the separate IROS closeout, ICRA working plan, and ZED node notes. Those remain valid as detail documents; this is the single view.

Three tracks, one system:

- IROS 2026 workshop, deadline Aug 24. Submitting.
- ICRA 2027, deadline expected mid-September. VERIFY when the CFP posts.
- Journal (JCCE if the workshop invitation comes, otherwise the GR00T merge).

## Status snapshot (2026-08-07)

X6, X2, and the B2 control are done. B2 (2026-08-07): generic textbook task-space IK already largely removes the flips (10-12/rep to 1-2/rep), so the flip fix is generic and C2 is NOT a novel solver; C2's surviving contribution is task-relevant tolerance (B2 loses ~12 deg approach direction at contact events, ours keeps it). So the paper is C1 (argument) + C3 (feasibility cost) + C4 (replica/replacement), with C2 a task-tolerance-aware instantiation; learning stays out unless X5 latency forces it.

X6 and X2 are done. X6 (go/no-go): the objective flip repairs the branch flips with a plain causal solver, no learning required for correctness (the plan's third outcome). X2 (cost of fidelity): the cost is FEASIBILITY, not energy. Even with the null space freed and the solver optimizing robot health, fidelity gives up only ~3-5% manipulability and its gravity load is if anything slightly lower; the one large, robust, mechanistic cost is JOINT-LIMIT MARGIN (the arm runs at ~40-50% of a task-optimal configuration's margin and hits limit violations, the same ill-conditioning that produces the flips). Consequences: lead the paper on the objective argument (C1) and the FEASIBILITY cost (C3, reframed off manipulability/gravity onto limit margin); treat learning as contingent on the latency result (X5); DROP the gravity-cost claim. Do not build the teacher-student pipeline. Next actions: advisor conversation with X6 + X2, then B2 control; X5 waits on the ZED node.

---

# 1. Hardware and sensing: two cameras

This distinction governs everything below and must not blur.

**iPhone 16 Pro (Stray Scanner), static tripod. HUMAN CAPTURE.** All IROS results are measured with it: the per-axis anisotropy, the screening study, the retargeted baselines. Nothing about this changes. It is also the deployability claim: a phone, no suits, no site instrumentation.

**ZED 2i, forward-facing at neck level on the H1-2. ROBOT PERCEPTION.** Egocentric, moves with the torso, and therefore not comparable to a tripod-mounted camera. Never place the two in the same comparison.

What the ZED unblocks:

- The real-time node, cut from IROS. Live camera to arm_sdk at 50 Hz.
- Task targets grounded in observed scene geometry instead of a scaled human trajectory. This addresses the IROS embodiment-scale finding directly.
- A deployed pipeline to measure end to end, making the latency experiment a measurement rather than a synthetic solve-time comparison. This is now load-bearing: X6 left "cheap online" unproven, and X5 on the deployed pipeline is what settles it.

**Scope discipline.** Scene perception is a second research thread and ICRA is six weeks. The objective flip is the paper; onboard perception is infrastructure. Do not claim "perception-driven task specification" as an ICRA contribution. That belongs in the journal version with GR00T.

---

# 2. Track B: ICRA 2027

## Thesis

Posture fidelity is the wrong retargeting objective when the application is hazard removal rather than teleoperation. Transfer the task, not the configuration: optimize task achievement subject to robot-native regularizers (manipulability, joint-limit margin, gravity load), with no posture-fidelity term.

**The delta from IROS is THE OBJECTIVE, not the learning.** X6 confirms this was the right bet: swapping the objective fixes the failure by itself. Lead with the objective flip in every conversation.

**Guardrail:** do not claim existing retargeting ignores the robot. It does not. The claim is about which term is the objective and which is the constraint. The field optimizes fidelity subject to robot feasibility; we optimize task achievement subject to robot health. The flip is licensed by the application, not by anyone's oversight.

## How IROS feeds ICRA

| IROS finding | ICRA role |
| --- | --- |
| E1 depth anisotropy (1.5x / 2.4x / 3.6x by motion) | Training noise model, measured on the deployment camera rather than guessed |
| E2 validated URDF gravity model | Robot-native cost term AND a metric in the headline experiment |
| E3 yaw branch flips | The motivating negative result the new objective repairs. X6 confirms it repairs |
| The pick-and-stack pilot | Preliminary evidence motivating the systematic study |
| The IROS pipeline | Baseline B1, hardware-validated |

IROS asks how well the method works. ICRA asks whether it was optimizing the right thing. A workshop paper whose own negative result motivates its extension is the ideal structure; say so in the introduction.

## Formulation

Baseline (IROS, fidelity retargeting):

```
q_t = argmin_q || d_robot(q) - d_human(p_t) ||^2 + smoothness
      subject to joint limits
```

Matches CONFIGURATION. This is where the branch flips originate: the residual gradient with respect to shoulder yaw is shallow in forward-reaching poses.

Proposed (task-centric):

```
q_t = argmin_q  L_task( FK(q), x_t ) + lambda * R_robot(q, q_{t-1})
```

with no posture-fidelity term. L_task is task achievement against the extracted specification, with task-relevant tolerance. R_robot references no human: manipulability, joint-limit avoidance, gravity torque load (the IROS-validated model), temporal continuity and velocity limits.

## Why learned: revised after X6

The original argument was: a warm-started local solver flips because it is greedy and the gradient is shallow; a global optimizer would not flip but is slow and non-causal; learning distills the global solution into a fast causal map.

X6 breaks the middle link. A causal, warm-started, single solve of the task-centric objective already produces zero flips (see result below). There is no non-causal global solution that only learning can deliver; the causal solver is already flip-free. So branch consistency no longer justifies learning.

Learning's surviving justifications are two, both to be measured, not assumed:

1. Latency. If the causal per-frame solve cannot hold 50 Hz on the deployed pipeline (X5), a distilled single-forward-pass student earns its place purely on speed.
2. Noise robustness (X4), trained against the E1 anisotropy.

If X5 shows the causal solve meets rate, the paper has no learning component and becomes method plus analysis, which this plan accepts. Residual plant correction stays a cited implementation detail (ASAP, RobotDancing, SENTINEL); never a contribution.

## Contributions

- C1. The argument, with hardware evidence, that posture fidelity is the wrong objective for hazard removal. Confirmed by X6. This is the paper, and it stands independent of learning.
- C3. The cost-of-fidelity study, now the quantitative headline and settled by X2: for the identical task, reproducing the human configuration is a FEASIBILITY cost, not an energy one. It runs the arm at ~40-50% of the joint-limit margin of a task-optimal configuration and into limit violations, the same ill-conditioning that produces the flips. Manipulability is a small (~3-5%) secondary effect; gravity is NOT a cost (fidelity is if anything lower), so the gravity claim is dropped. Lead on limit margin: it is weight-robust and mechanistic, whereas manipulability/gravity depend on the R_robot weights.
- C4. Replica vs replacement mode, a distinction the field conflates.
- C2. The task-centric formulation, now settled by B2 and reframed: generic textbook task-space IK (B2) already largely removes the flips (10-12 per rep down to 1-2), so C2 is NOT a novel flip-fixing solver. Its surviving, honest, modest contribution is task-relevant tolerance: B2 sacrifices ~12 deg of approach-direction at contact events (position-only), whereas our formulation keeps orientation where the task needs it while still avoiding flips. Present C2 as a task-tolerance-aware instantiation of task-space IK, not a new method. The learned variant enters only if X5 or X4 justify it.

## Baselines

- B1 the IROS geometric retargeter (ours, hardware-measured).
- B2 task-space IK with generic null-space resolution. The "is learning needed / is C2 a method" control. DONE (2026-08-07): generic textbook DLS position IK + null-space joint-limit avoidance drops flips from 10-12/rep to 1-2/rep, so the flip fix is generic and C2 is not a novel solver; but B2 costs ~12 deg approach-direction error at contact events, which our formulation avoids (C2's surviving contribution = task-relevant tolerance). Detail: icra2027/b2_control.py, b2_results.md.
- B3 an external published method adapted to H1-2. Stretch goal.
- B4 the proposed learned task-centric retargeter. Build only if X5 or X4 justify it.

## Experiment matrix

**X6. The critical ablation. DONE (2026-08-05).** Solve the task-centric objective on M2 and M3, count branch flips. Three outcomes were: flips persist (learning justified); flips vanish (learning drops to latency and robustness); flips vanish and the objective solves cheaply online (learning may be unnecessary).

**RESULT.** M2 and M3, reps 1 to 3, right shoulder-yaw. Task held identical, only the redundancy resolution changed. Flip defined as a single-frame yaw jump above 6 rad/s.

| rep | fidelity, yaw-hack off (flips) | task-centric, causal single solve | task-centric, multi-start | EE task error |
| --- | --- | --- | --- | --- |
| M2_1 | 10 | 0 | 0 | 0.17 cm |
| M2_2 | 10 | 0 | 0 | 0.17 cm |
| M2_3 | 12 | 0 | 0 | 0.18 cm |
| M3_1 | 0 | 0 | 0 | 0.12 cm |
| M3_2 | 7 | 0 | 0 | 0.12 cm |
| M3_3 | 8 | 0 | 0 | 0.12 cm |

47 fidelity flips across the reps go to 0 under the task-centric objective, with sub-2 mm end-effector tracking, so "no flips" is not won by dropping the task. The causal single solve matches the expensive multi-start, so it is the objective, not the solver, that fixes the flips. This is the third outcome. Caveat: the causal solver used here is unoptimized (about 37 ms per frame), not yet proven at 50 Hz; X5 on the deployed pipeline settles whether learning re-enters for latency. Script and detail: icra2027/x6_taskcentric.py, icra2027/x6_results.md.

X1. Branch-flip elimination across tasks and baselines. Flip count, peak yaw velocity, percent of frames at a joint limit. Sim plus hardware. Extends X6 to the full baseline set including B2.

**X2. THE COST OF FIDELITY. The headline. DONE (2026-08-06).** Per motion, the fidelity configuration vs a robot-optimal configuration for the same task (task achievement held fixed), on manipulability, gravity load, joint-limit margin, and violations. RESULT (v1 constrained + robot-optimal variant with the swivel null space freed): the cost is FEASIBILITY, not energy. Fidelity gives up only ~3-5% manipulability, and its gravity load is if anything slightly lower; the one large, robust, mechanistic cost is joint-limit margin, ~40-50% of a task-optimal configuration's margin, with limit violations. Revised target sentence: "reproducing the worker's configuration leaves the arm at ~40-50% of the joint-limit margin available to a task-optimal configuration for the identical task, and drives it into limit violations, for no task benefit; this is the same ill-conditioning that produces the branch flips." The side-by-side configuration figure (x2_robotopt_figure) shows fidelity pulling the elbow in vs robot-optimal keeping it out at the same wrist target. Detail: icra2027/x2_results.md, x2_costoffidelity.py, x2_robot_optimal.py. Confirmed on the full 6 reps (2026-08-07): mean manipulability +3%, gravity -0.7 Nm (not a cost), limit margin +55% (fidelity ~45% of the task-optimal margin), and fidelity VIOLATES a joint limit on 2 of 6 captures vs 0 of 6 for the task-optimal config. Lead on the violation count (binary, robust) plus mean margin; the per-rep margin gap is not monotonic (on the easy rep M3_3 the task-optimal solve traded margin back for manip/gravity), and manipulability/gravity are R_robot-weight-dependent while limit margin is not.

X3. Hardware replay, both methods. Non-negotiable for ICRA. Where the ZED is available, ground task stations in observed scene geometry rather than the scaled human trajectory, and report whether that alone closes the physical-task gap the IROS pilot exposed.

X4. Perception noise robustness under the anisotropy measured in E1. With the ZED in the loop this can run against live onboard sensing as well as injected noise.

X5. Latency end to end on the deployed pipeline: ZED frame arrival, pose, solve, publish. Median and p95 per stage. Now pivotal: it decides whether the causal task-centric solve is deployable as is (no learning) or whether learning is warranted for speed.

X7. Replica vs replacement mode on the same motion.

## Related work: two threats

T1. Liu and Liu, CACIE 2026. Humanoid learning construction tasks from worker demonstrations, teacher-student whole-body policy, 82.45 mm MPJPE. READ BEFORE WRITING RELATED WORK. Differentiation: they retarget for fidelity and train a policy to track the retargeted reference; we argue tracking the human's configuration is the wrong target.

T2. The residual-learning cluster. Not a threat to the thesis, but a hard boundary on what may be claimed.

Also position against ergonomics-aware robotics (DULA/DEBA, NeuroErgo, REBA-based planning). All of it keeps the human IN the loop while the robot adapts to reduce the human's load. Nobody uses ergonomic risk where the robot REPLACES the human. That is the gap.

## Immediate next actions (post-X6, post-X2)

- [ ] Advisor conversation with Francis and Cho, presenting X6 + X2, before committing six weeks. Frame: objective flip fixes the failure without learning (X6); the cost of fidelity is feasibility, quantified (X2); paper is argument plus cost-analysis.
- [x] B2 control: generic IK also removes the flips (C2 is an instantiation, not a novel solver); C2's contribution is task-relevant tolerance (B2 loses orientation at events).
- [x] X2 full 6-rep run: means confirmed (manip +3%, gravity not a cost, limit margin +55%, fidelity violates a limit on 2/6 captures vs 0/6 task-optimal).
- [ ] IRB: single-subject for ICRA (stated limitation), start multi-subject now for the journal. Confirm existing captures are covered if the subject is a labmate.
- [x] X2 cost-of-fidelity (verdict: feasibility, not energy).
- [x] X6 ablation.
- [x] CFP deadline verified.
- [x] training_data consolidated (42 replay logs).

---

# 3. Track C: infrastructure (the ZED real-time node)

Built in parallel, gating X5 and enabling X3's grounded-station variant. Full spec in claude_code_realtime_zed_node_task.md. Summary of deliverables:

- D0. ZED stack verified: CUDA, ROS2 wrapper, coexistence with CycloneDDS, fps and depth validity at 0.4 to 1.5 m (near the ZED minimum range; confirm it resolves there). Gate: do not proceed until this passes.
- D1. Camera-to-base extrinsic, measured not guessed, verified at three or more workspace locations with residuals reported. Load-bearing: a systematic extrinsic error will look like a retargeting error and be chased in the wrong subsystem.
- D2. Real-time node: ZED RGB and depth, MediaPipe, confidence-gated patch-median depth sampling (reuse the offline sampler), causal depth conditioning, existing solver with yaw penalty, publish at 50 Hz. Note: the offline zero-phase Butterworth is non-causal and cannot run live; the causal substitute introduces phase lag that must be reported as part of the latency budget. Graceful degradation on pose or depth loss. Hard enable gate.
- D3. Latency instrumentation, four timestamps per frame, plus latency_report.py.
- D4. Fiducial-based task stations (AprilTag on pick surface and stack location), emitted in base frame. Deliberately not a perception contribution.
- D5. Comparison harness: same task with stations from the scaled human trajectory vs observed scene geometry. This turns the IROS "task-level transfer is the implied fix" into a measurement.

Safety: arms-only, locomotion disabled, e-stop held, first runs with the command path disconnected, then 50 percent speed scaling.

---

## Pressure valves, in order

1. Drop the multi-subject expansion; report single subject as a limitation.
2. Drop B3, lean on B1 and B2.
3. Drop D4 and D5 (the grounded-station work); X5 still runs on D2 alone.

Do NOT drop hardware validation (X3).

## Fallback

RA-L takes rolling submissions, is a journal, and can carry a conference presentation slot. Decide by mid-August, not September.
