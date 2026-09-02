# Keep the Direction, Reset the Reach

Working title: **Keep the Direction, Reset the Reach: Task-Defined Retargeting for Construction and Manufacturing Humanoids**

Scope is deliberately bounded to manual work in construction and manufacturing. The mechanism is general to retargeting, but claiming only these two domains keeps the evidence and the claim the same size, which is the safer position under review.

Target venue: ICRA 2027. RA-L is not being pursued. Robot: Unitree H1-2, arm-only protocol.

**In one line:** when a humanoid *replaces* a human worker, retargeting should transfer the action, not the arm. We show that copying the worker's reach extension, even implicitly, is what drives the robot into its joint limits, and that defining the target by the action removes the problem.

Status as of 2026-09-01. The ZED refilm is captured and the first hardware replay is done. Every offline experiment has been re-run for **both arms** on all twenty-five motion captures and twenty reach-sweep repetitions, after correcting the missing human-side conditioning. Single subject.

---

## Thesis

Retargeting inherits the demonstrator's configuration through two channels, and dropping the posture term closes only one of them.

A retargeter with no posture term still needs a wrist target, and the natural construction of that target, along the demonstrator's limb directions at the robot's own link lengths, gives it a magnitude set by the demonstrator's elbow angle. A worker reaching at full stretch therefore asks the robot for full stretch. Above roughly 90% extension the swivel circle has contracted and the arm sits on its limits. Extension is a property of the configuration, not of the task. Keeping the reach direction and re-setting the magnitude removes the limit contact entirely on reaching motions, and does nothing for cross-body reach, which becomes infeasible through a different mechanism.

Posture fidelity also reduces clearance directly, by 0.149 of margin on the reaching arm with no limit avoidance active. That is a cost rather than a blocking mechanism, and it is specific to the arm doing the reaching.

## Purpose

Markerless human-to-humanoid retargeting for manual work in construction and manufacturing, where the robot replaces the worker rather than being teleoperated. Both settings are dimensioned for the human body and change faster than fixed automation can be re-tooled, which is what makes a humanoid form attractive and a human demonstration the natural input. Much of the field retargets to match human configuration. We ask whether that is the right objective for replacement work, and we measure what the choice yields and what it removes.

Three guardrails keep the claim measured:
1. We do not claim existing retargeting ignores the robot. The distinction is which term is the objective and which is the constraint.
2. We do not claim a better solver. Generic IK matches ours numerically.
3. We do not claim the robot reproduces the human's hand path. It deliberately does not, and we say so.

## Contributions

| | Claim | Evidence |
| --- | --- | --- |
| **C1** (primary) | Implicit extension fidelity. A target rebuilt from human arm directions carries the demonstrator's reach extension, and above roughly 90% extension that is what drives limit contact. Re-setting the magnitude while keeping the direction removes it. | X9, X6 |
| **C1b** | The remedy has a stated scope. Cross-body reach stays infeasible through a direction-driven mechanism that extension re-setting does not touch. | X6 on M5 |
| **C2** (support) | Posture fidelity reduces clearance to the joint limits directly. With no limit avoidance it holds 0.149 less margin than a posture-free objective on the reaching arm. | X8 |
| **C3** (support) | Published objectives sit on one monotonic axis of posture weight, and matching approach direction reduces clearance sharply on the reaching arm. | B3, X10 |
| **C4** | Replica versus replacement. Fidelity is the goal in replica mode and a cost in replacement mode. | X7 |

Side is a factor throughout, and it matters. Every effect below is strong on the reaching arm and weak or absent on the supporting arm.

## Results

All from the 2026-08-29 ZED session after skeleton normalization and human-side conditioning, both arms, twenty-five captures. Single subject.

### C1. Extension is what drives limit contact

Four conditions on one solver. Fidelity is direction matching, fidelity-reg adds the shipped yaw regularizer, task-centric chases a target rebuilt from the demonstrator's directions, and task-defined keeps that reach direction while re-setting the magnitude.

Reps showing limit contact, out of five per motion:

| motion | fidelity | fid-reg | task-centric | **task-defined** |
| --- | --- | --- | --- | --- |
| M1 overhead hammering (control) | 0/5 | 0/5 | 0/5 | **0/5** |
| M2 bricklaying | 5/5 | 5/5 | 3/5 | **0/5** |
| M3 lifting | 5/5 | 5/5 | 0/5 | **0/5** |
| M4 pick-and-stack | 5/5 | 4/5 | 1/5 | 1/5 |
| M5 cross-body reach | 5/5 | 5/5 | 5/5 | 5/5 |
| **total, reaching arm** | **20/25** | 19/25 | 9/25 | **6/25** |
| total, supporting arm | 9/25 | 9/25 | 9/25 | 5/25 |

Branch flips fall 1930 to 678 to 411 on the reaching arm. The control motion never approaches a limit under any condition, so the effect is task-dependent rather than a property of the robot.

### The reach sweep is the cleanest evidence

Same forward reach at four marked distances, five repetitions each, reaching arm:

| target | robot extension | fidelity | task-centric | **task-defined** | fid margin | tdef margin |
| --- | --- | --- | --- | --- | --- | --- |
| 12 in | 83.7% | 0/5 | 0/5 | **0/5** | 0.327 | 0.416 |
| 16 in | 87.1% | 0/5 | 0/5 | **0/5** | 0.353 | 0.395 |
| 20 in | 94.3% | 4/5 | 5/5 | **0/5** | 0.343 | 0.412 |
| 24 in | 99.8% | 5/5 | 5/5 | **0/5** | 0.292 | 0.378 |

A sharp threshold near 90% extension, nothing below it and almost everything above it. Task-centric fails exactly as fidelity does at the far distances, because its target inherited the demonstrator's extension. Task-defined holds at zero across the whole sweep while tracking its own target to 0.1 mm. This is the result that should lead the paper.

The supporting arm reaches only 83 to 87% during these trials and shows almost no contact, which is consistent with the threshold rather than an independent test of it.

### C1b. Where the remedy stops working

On cross-body reach, task-centric and task-defined are indistinguishable at 5/5 apiece, so re-setting extension is inert there. Crossing the midline drives shoulder yaw and roll toward their limits irrespective of how far the hand travels. There are two mechanisms, and this paper measures a remedy for one of them.

Pick-and-stack is the other boundary. On the supporting arm every condition produces exactly 4/5, including task-defined, which indicates the workspace rather than the objective is binding. That motion should not be cited as evidence either way.

### C2. Posture fidelity reduces clearance

Posture weight crossed with limit avoidance, reaching arm. Cells give mean limit margin and reps with contact out of twenty-five.

| | limits off | w=0.05 | w=0.20 | w=0.50 |
| --- | --- | --- | --- | --- |
| posture pinned (OKAMI) | 0.198, 23/25 | 0.332, 18/25 | 0.497, 6/25 | 0.597, 0/25 |
| posture free | **0.347, 8/25** | 0.415, 12/25 | 0.500, 5/25 | 0.599, 0/25 |

With no limit avoidance, posture fidelity holds **0.149 less margin** and shows contact on 23 of 25 captures against 8. That direct cost is the claim to make.

Do not claim more than that. Limit avoidance at a task-preserving weight helps the pinned condition (23 to 18) and makes the free condition worse (8 to 12), so the earlier framing that limit avoidance cannot rescue a posture-matched arm is not supported. On the supporting arm the effect does not appear at all: posture-pinned holds slightly more margin than posture-free, 0.267 against 0.252. C2 is a cost on the reaching arm, not a general blocking mechanism.

The high limit-avoidance weights remain outside the useful range. At w=0.20 tip error is 48.7 mm and at w=0.50 it is 165.9 mm, so those columns are the boundary rather than results.

### C3. The posture-weight axis

Published objectives swept over posture weight, reaching arm:

| condition | posture weight | margin | contact | tip | approach |
| --- | --- | --- | --- | --- | --- |
| OKAMI gain 1.0 | 0.04 / 0.08 | 0.198 | 23/25 | 16.6 mm | 11.2 deg |
| H2O keypoint | hard | 0.228 | 20/25 | 27.6 mm | 8.9 deg |
| our retargeter, naive | hard | 0.232 | 20/25 | 16.8 mm | 4.0 deg |
| OKAMI gain 0.1 | 0.004 / 0.008 | 0.272 | 21/25 | 2.4 mm | 19.5 deg |
| our retargeter, regularized | hard | 0.321 | 19/25 | 23.2 mm | 5.3 deg |
| OKAMI gain 0.01 | 4e-4 | 0.340 | 9/25 | 4.2 mm | 28.4 deg |
| OKAMI gain 0.001 | 4e-5 | 0.347 | 8/25 | 4.2 mm | 29.4 deg |
| **task-centric** | none | **0.417** | **9/25** | 1.5 mm | 38.1 deg |

The OKAMI sweep is monotonic: margin rises from 0.198 to 0.347 as posture weight falls three decades, and contact falls from 23 to 8. Zero posture weight reaches 0.417. Approach-direction error grows across the same span, which is the trade.

On the supporting arm the axis is flat, with every condition between 0.252 and 0.341, so this too is a reaching-arm effect.

**Matching approach direction reduces clearance sharply.** Reaching arm:

| | margin | contact | tip | approach at contact |
| --- | --- | --- | --- | --- |
| no approach term | **0.417** | **9/25** | 1.5 mm | not enforced |
| gated to contact frames | 0.300 | 23/25 | 11.0 mm | 3.3 deg |
| enforced throughout | 0.206 | 23/25 | 32.7 mm | 9.1 deg |

Gating to 36.6% of frames still takes contact from 9 to 23 captures and degrades tracking from 1.5 to 11.0 mm. On the supporting arm the same term is nearly free, 0.355 to 0.336 with contact unchanged at 9/25, so the cost is specific to the arm doing the reaching.

### X3. Hardware replay, 2026-08-31

The only non-simulation result. M3_1 lifting, reaching arm, both conditions replayed under an identical 2 Hz filter and speed scale, with gravity feed-forward and an e-stop operator present.

| | frames commanded onto a limit | executed outside limits | tracking error |
| --- | --- | --- | --- |
| fidelity, regularized | 100 of 6037 | **0** | 3.113 deg at limit frames, 1.180 deg elsewhere |
| task-defined | **0** | 0 | 1.138 deg, max 4.52 deg |

**Outcome B.** The onboard controller held every joint inside its limits and the cost appeared as tracking error elevated by a factor of 2.6, precisely at the frames the offline analysis predicted. Peak error was 9.53 deg under fidelity against 4.52 deg for task-defined. This settles the definitional caveat: the clamp is real, it never exceeds a limit, and it is measurable.

Caveats. The contrast is 100 frames on one capture, and the replay drove the reaching arm only while the motion is bimanual. A capture with more contact should be replayed.

### Does the task still succeed?

The task-defined target does not put the hand where the demonstrator's hand was, so the obvious objection is that limit contact was avoided by not performing the task. Measuring hand-position error against the demonstrator cannot answer that, because it measures fidelity, which is the thing being argued against.

We ask the deployment question instead. A worker who cannot reach a point steps closer rather than over-extending, and a legged robot can do the same. For each capture we find the single horizontal base translation that brings the whole trajectory inside the usable radius, one translation for the capture rather than one per frame, because a stance shift is a step and a per-frame shift is not.

| | mean step | fully reachable after one step | mean residual | worst residual | within the 8 cm tolerance |
| --- | --- | --- | --- | --- | --- |
| reaching arm | 14.0 cm | 12/25 | 1.61 cm | 6.51 cm | **25/25** |
| supporting arm | 11.9 cm | 16/25 | 0.74 cm | 6.24 cm | **25/25** |

A single step of about 14 cm brings every capture within 6.5 cm of every point the task requires, and **all fifty arm-captures fall inside the 8 cm attainment tolerance our own prior work used to define task success**. The re-set reach therefore costs a stance adjustment, not the task.

Two things to say plainly. Roughly half the captures need more than one step to be exactly reachable, because a motion that sweeps a wide volume cannot be covered by a single fixed base position, which is why a real deployment would step during the task as a bricklayer does. And this is a reachability argument computed offline, not a demonstration of a completed physical task, which remains a limitation.

### Sensitivity of the extension ceiling

The ceiling of 0.88 is not a tuned value on a knife edge. Swept over the reaching arm it shows a flat minimum of 6 captures in contact across ceilings 0.88, 0.90 and 0.94.

| ceiling | 0.78 | 0.82 | 0.86 | 0.88 | 0.90 | 0.94 | 0.98 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contact out of 25 | 8 | 8 | 7 | **6** | **6** | **6** | 8 | 11 |
| mean margin | 0.354 | 0.382 | 0.405 | 0.415 | 0.423 | 0.437 | 0.410 | 0.391 |
| tip error mm | 0.66 | 0.63 | 0.61 | 0.60 | 0.60 | 0.61 | 6.32 | 10.74 |
 Above it the solver can no longer reach even its own target, tip error rising from 0.60 mm to 10.74 mm at a ceiling of 1.00, which is the near-singular regime appearing again. The supporting arm prefers a slightly lower ceiling, bottoming out at 4 of 25 between 0.78 and 0.86, so 0.88 is reported as a two-arm compromise rather than an optimum.

### Supporting

- **X2:** at matched task accuracy, fidelity shows contact on 20 of 25 captures against 9 for task-centric on the reaching arm, and 9 against 9 on the supporting arm.
- **B2:** 693 flips and 9/25 contact, against task-centric's 678 and 9/25. The same method numerically, so no solver is claimed.
- **Flips are a symptom.** The shipped yaw regularizer changes flips (1930 to 1961 on the reaching arm) without changing contact meaningfully (20/25 to 19/25). Reducing flips does not buy feasibility.

## Method

Retargeting operates on human arm directions in the subject's own torso frame, built from shoulders and hips, so it does not depend on the camera-to-robot extrinsic.

Baseline family, parameterized by posture weight `w_p`:

```
q = argmin_q  L_task(FK(q), x)  +  w_p * P_human(q, p)  +  w_l * R_limit(q)
```

H2O and our direction matcher sit at `w_p` effectively infinite. OKAMI sits near 0.04 to 0.08. The proposed method sits at `w_p = 0` with `w_l` about 0.05.

The target `x` is where the contribution lives. Rebuilding it as `shoulder + l_up * u + l_fo * f` uses the robot's limb lengths but keeps the human's elbow angle, so its magnitude is the human's extension. The task-defined target keeps the reach direction and re-sets the magnitude:

```
ext_robot = min(ext_human * ext_scale, ext_cap)
```

`ext_cap` is 0.88, just inside the measured violation threshold near 90%. `ext_scale` maps a capture's whole reach profile into the usable band, so the relative shape of the motion survives rather than being clipped at the top.

## Data

- ZED SDK body tracking (BODY_38), neck-mounted. HD720 at 60 fps, SVO2 per capture.
- Session `experiments/zed_data/2026-08-29/`. Motions at 5 reps each: M1 hammering, M2 bricklaying, M3 lifting, M5 cross-body reach. M4 two-handed pick-and-stack at 3 clean reps. Reach sweep at 4 distances from 12 to 24 inches, 5 reps each.
- All captures trimmed to the task window and cleaned of single-frame tracking outliers.

### Skeleton normalization, and why it is required

BODY_38 degrades when the subject reaches toward the camera. The arm points along the camera axis, is foreshortened, and the wrist is pushed out in depth. In this session the measured shoulder-to-wrist distance reached 72 to 79 cm at the far reach distances while the upper arm plus forearm summed to 52 to 55 cm. A rigid arm cannot do that. Confidence does not catch it, because the keypoints are confident and wrong, and an outlier filter does not catch it, because the error is sustained rather than a single-frame spike.

Every capture is therefore normalized to the subject's measured arm length of 24 inches, or 0.6096 m. The shoulder and the reach direction are kept, the reach distance is clamped to the true arm length, and the elbow is re-placed by two-link inverse kinematics on the side the measured elbow was on. This preserves the action and discards the magnitudes the sensor got wrong.

This step is not cosmetic. Before it, the robot target sat at 97 to 100% extension at every sweep distance, violations saturated at 9 or 10 out of 10 for every method, and the X8 contrast inverted. Any paper using this pipeline has to report the normalization.

## Corrections applied, and what they moved

Two faults were found on 31 August while preparing the hardware replay. Both are fixed and every number above is post-fix. Recording them because they changed conclusions.

### Missing human-side conditioning

The IROS retargeter applies two stages to the limb directions before solving, inside its main loop rather than in a reusable function, so our reader reproduced the torso frame and direction extraction but not these. Every solver received raw per-frame directions.

- A **teleport gate** dropping frames whose limb direction rotates faster than a person can move, 220 frames on one capture.
- A **One-Euro filter** on the direction components, renormalized, with elbow flexion recomputed from the filtered directions.

Without them the shoulder-yaw branch is ambiguous frame to frame. Against the retargeter itself on one capture, our fidelity condition produced yaw spanning -2.90 to +2.72 rad where the tool produced -0.51 to +0.95. After the fix ours spans -0.21 to +0.78. Pitch and elbow always agreed.

Three conclusions changed, and two of them were ours to lose.

- **C2 weakened.** The margin separation at a task-preserving limit-avoidance weight fell from 0.193 to 0.083, and the violation counts stopped behaving monotonically. C2 is now a direct cost, measured with limit avoidance off, not a blocking mechanism.
- **The non-monotonic zone on the posture axis disappeared.** The dip at OKAMI gain 0.1 was a conditioning artifact. The axis is monotonic.
- **A claim reversed.** Our own shipped regularizer went from margin 0.231 to 0.321, better than naive rather than worse. The claim that patching flips costs feasibility is withdrawn.

C1 and the reach-sweep threshold were essentially unaffected, which is the reason to lead with them.

### Right arm only

Every experiment analysed the right arm, inherited from the IROS work where the flips were documented there, and never justified for the bimanual motions. Wrist travel shows the supporting arm doing 77% of the reaching arm's work in bricklaying, 54% in lifting and 75% in pick-and-stack, against 8% in cross-body reach.

Both arms are now analysed, and the asymmetry is itself a finding. Fidelity shows contact on 20 of 25 captures on the reaching arm and 9 of 25 on the supporting arm. Approach-direction matching reduces clearance sharply on the reaching arm and is nearly free on the supporting one. The mechanism lives where the reaching happens, and the paper should say so rather than averaging the two.

## Baselines

- **B1:** our direction-matching retargeter, reported both naive and regularized.
- **B2:** generic task-space damped-least-squares IK with null-space limit avoidance.
- **B3:** H2O-style hard keypoint matching and OKAMI-style soft posture weighting.

## Remaining work

Done and not repeated below: the capture session, skeleton normalization, the D1 extrinsic, and every offline experiment across all twenty-five captures with X9 covering the twenty sweep repetitions.

### 1. X3 hardware, revised design

The first replay used one arm at speed scale 0.2. Both choices should change, and the numbers say so.

**Run both arms.** The motions are bimanual and the supporting arm performs 54 to 77 percent of the reaching arm's wrist travel, so a single-arm replay leaves a stated limitation in the paper. Side asymmetry is a headline finding, and hardware evidence for it is far stronger than offline evidence alone. Minimum wrist-to-wrist separation is 22.1 cm on M2, 22.6 cm on M3 and 19.0 cm on M4, so self-collision is not a concern on these captures. M5 was not checked and is not needed. The prep function already writes a full fifteen-joint vector with both side column groups, filling one from the solve and the other from rest, so solving both sides is a contained change.

**Run a speed ladder, not a single faster speed.** The `--vel-warn` threshold of 4.0 rad/s is a hand-set lab-safety margin, not a hardware limit. URDF limits are 9.0 rad/s at shoulder pitch and roll and 20.0 rad/s at yaw and elbow. At 0.5 the worst joint on M3_1 fidelity is the elbow at 6.11 rad/s, which is 31 percent of its limit, and no joint exceeds 36 percent. So 0.5 is well inside hardware capability and the only real constraint is operator safety.

Keep 0.2 and add 0.5 rather than replacing one with the other. Speed then becomes a reported factor. This preserves continuity with the result already written up, removes the quasi-static limitation, and answers step 2 of the endurance concept in the same session. If the 2.6-fold tracking-error result survives at 0.5, it is considerably more convincing.

| condition | peak joint velocity, raw | at 0.2 | at 0.5 |
| --- | --- | --- | --- |
| fidelity, regularized | 12.22 rad/s (elbow) | 2.44 | 6.11 |
| task-defined | 5.22 rad/s (elbow) | 1.04 | 2.61 |

**A finding sitting in that table.** Fidelity demands 2.3 times the peak joint velocity of task-defined on the same motion. This is an unreported cost of fidelity, it is free to add, and it matters for the endurance concept, since fidelity moves faster while task-defined pushes harder. Those are different wear mechanisms.

Captures to run: M2 bricklaying and M3 lifting, the strongest pair at 5/5 contact under fidelity against 0/5 for task-defined. Four trajectories times two speeds is eight runs.

### 2. Blocking a submission

- **X3, both arms, speeds 0.2 and 0.5.** The first replay gave Outcome B on only 100 contact frames of one capture. Its trajectories were also built on the superseded arm-length constant, so it must be redone regardless.
- **Figures 1 to 4 and Tables 1 to 3.** The outline is cut to this set and budgeted at 7.94 pages. Figure 1 needs panel (b) added to the existing system diagram. Figure 4 has real hardware data behind it.
- **Resolve every VERIFY marker in `references.bib`.** Eleven entries are unconfirmed, including the closest related work.
- **Decide C4.** It is now carried by one discussion paragraph and has no dedicated experiment, since X7 dual mode is still a skeleton. Either build X7 or demote C4 to a discussion point.

### 3. Corrected on 2026-09-01: the arm-length constant, and what it fixed

**The constant was wrong.** The demonstrator measures 22 in shoulder to wrist, 0.5588 m. Every analysis before 2026-09-01 divided by 0.6096 m, which was a shoulder-to-**hand** measurement. The tracker puts its wrist keypoint at the wrist, so the conditioning constant was 9.1 percent too large.

**What that broke.** Extension ratios were understated by 9.1 percent, and the same constant set the skeleton normalization clamp, so the conditioned inputs themselves were wrong. The apparent saturation at the far sweep levels was an artifact of it.

**What the correction bought.** Re-normalizing at 0.5588 and re-running the sweep turns a step function into a clean dose-response.

| pole | ext % | fidelity | task-centric | task-defined |
| --- | --- | --- | --- | --- |
| 12 in | 90.7 | **0/5** | 0/5 | 0/5 |
| 16 in | 94.7 | **1/5** | 0/5 | 0/5 |
| 18.3 in | 99.6 | 2/3 | 2/3 | 0/3 |
| 20 in | 97.6 | **3/5** | 5/5 | 0/5 |
| 22 in | 95.1 | 2/3 | 3/3 | 3/3 |
| 24 in | 99.9 | **5/5** | 5/5 | 0/5 |

Twenty-six repetitions across six pole distances, all frontal, both sessions combined. The earlier report of 1/5 then 0/5 then 5/5 then 5/5 was non-monotonic. **The onset is between 94 and 95 percent, not 90.** A graded rise is harder to dismiss as an artifact than a cliff, so this is the stronger result.

One loose end: at 22 in the task-defined target violates 3/3, breaking an otherwise clean record. Per-capture rates are 0.5 to 1.0 percent of frames, so these are marginal touches. Check before it goes in a table.

### 3c. The extension ceiling is a trade-off, and both ends are reported

Swept on 25 captures per arm at the corrected 0.5588 m arm length. Contact is the joint minimum over both arms, attainment is captures within the 8 cm tolerance after one base step.

| ext_cap | contact right | contact left | combined | attainment right | attainment left |
| --- | --- | --- | --- | --- | --- |
| 0.78 | 8/25 | 4/25 | 12 | | |
| 0.82 | 6/25 | 4/25 | **10** | | |
| **0.84** | | | **10** | **22/25** | **21/25** |
| 0.86 | 6/25 | 4/25 | **10** | | |
| **0.88** | 6/25 | 5/25 | 11 | **25/25** | **25/25** |
| 0.94 | 6/25 | 9/25 | 15 | | |
| 1.00 | 10/25 | 15/25 | 25 | | |

**0.88 is the operating point, and 0.84 is reported alongside it because the sweep is what justifies the choice.**

Re-running X6 on all twenty-five captures at 0.84 returns results identical to 0.88 on both arms, at 20, 17, 8 and 6 of 25 on the reaching arm and 13, 14, 14 and 5 on the supporting arm. The ceiling therefore has no effect on the limit-contact result over the range that matters. Its only measurable effect is on task attainment, which falls from 50 of 50 at 0.88 to 43 of 50 at 0.84.

So 0.88 is not a compromise between two costs. It matches the lower ceiling everywhere the paper makes a claim and loses nothing on reach, which is the straightforward reason to prefer it. The sweep is still worth reporting, because a parameter that turns out not to matter over its plausible range is a stronger position than one tuned to a single value.

Reporting both is also the more honest presentation. The ceiling is a parameter that trades joint clearance against reach, not a value that was discovered, and the sweep shows the shape of that trade rather than a single tuned number.

The reach-extent sweep is unaffected by the choice. Task-defined produces 0 out of 5 contact at every level under both ceilings, so the dose-response result does not depend on it.

### 3b. Two recommendations that did not survive contact with the data

Recorded so they are not repeated.

**The 45 degree camera angle was wrong.** It was meant to fix saturation, but saturation was never real, only the constant was. At 45 degrees the shoulder line lies partly along the depth axis, so the torso frame gets noisy and everything derived in it inherits that. At matched pole distance the 45 degree block reads extension 0.626 where frontal reads 0.893. Every frontal capture from both sessions lies on one monotonic curve. The 45 degree points do not. **Frontal is the standard geometry.**

**Rescaling the pole distances as fractions of arm length was wrong.** It put the near levels at 6.6 to 11 in, where 79 to 90 percent of frames are the arm at rest and forward reach is 4 to 11 cm. The original absolute distances of 12 to 24 in were better calibrated. Two new tools guard this: `check_sweep_span.py` reports rest fraction and forward-reach monotonicity per level, and `check_saturation.py` reports physically impossible frames.

**A third finding worth carrying into the paper.** A hanging arm is nearly straight, scoring 0.90 to 0.97 on any extension measure. Captures that are mostly rest therefore report the resting posture rather than the reach. Extension must be measured on reach frames, meaning the top quartile of forward reach within each capture, not over whole captures.

### 4. Changes a claim if it comes out badly

- **Verify the redundancy accounting numerically** against the task Jacobian rank at real capture configurations. Paragraph 10 currently presents the swivel argument as a mechanism when it is a hypothesis, and a reviewer can attack that. Compute only, no capture, and cheap.
- **Diagnose the M5 cross-body failure.** Direction-driven rather than extension-driven, so it needs its own account. The candidate is shoulder yaw and roll near their limits during midline crossing. If a second remedy exists it belongs in the paper, and if not the scope statement stands as the honest result.

### 5. Completeness, low risk

- **Annotate contact events.** The X10 gate is a wrist-speed heuristic rather than ground truth.
- **Explain the 31 versus 12 percent torque gap** between the M3_1 hardware log and the gravity model. One script, and it decides whether the endurance concept is about posture or about controller effort.

### 6. Needs a person, not compute

- **S2 and S3.** Protocol in `SUBJECT2_PROTOCOL.md`. S2 at 19.5 in is the prediction test, sitting 2.5 in below S1. S3 at 22.5 in is a same-length replication of S1, which tests whether two people at the same arm length share a threshold. Six pole distances, frontal, chest height, matched to S1's levels in ratio terms. S1 needs nothing further.

### 7. Advisors

- Approval of the pick-and-stack paragraph before it is load-bearing.
- Whether C1 or C2 leads.
- Whether C4 survives without X7.

## Limitations to state plainly

- Single subject, one robot, arm-only protocol.
- **The analysis covers one arm of motions that are bimanual.** The left arm performs 54 to 77 percent of the right's wrist travel in bricklaying, lifting and pick-and-stack. The re-analysis covers both arms and side should be reported as a factor.
- **The task-defined target does not reproduce the human's hand position**, deliberately. Reachability after a single base step is reported instead, and all fifty arm-captures fall inside the 8 cm attainment tolerance our prior work used. That is an offline reachability argument, not a demonstration of a completed physical task, and the paper should say so.
- The ZED skeleton required normalization, and the raw sensor is unreliable for reaches toward the camera. Cross-paper comparison must name both the camera and the skeleton algorithm.
- `ext_cap` at 0.88 sits in a flat minimum from 0.88 to 0.94 on the reaching arm, so it is not a knife-edge value, but the plateau was located on one subject.
- Approach-direction error is a per-rep mean rather than contact-only.
- **"Violation" means the solver drove a joint onto its limit and clamped there**, not that the commanded angle exceeded the limit. Minimum margins sit at about 1e-10 in the affected reps, which is the clamp. This is still a real failure, since the arm saturates and loses a degree of freedom, but the paper must say it precisely rather than implying the command left the feasible set. X3 on hardware is what turns this into a physical claim.

## Open items for the advisors

1. Does C1 lead, or C2? Our leaning is now C1, the extension finding, because it is the cleanest categorical result and it is actionable.
2. How to frame the fact that the robot does not reproduce the human's hand path. We think it is a feature of replacement-mode retargeting, but it needs to survive a reviewer who reads it as tracking failure.
3. Whether the skeleton-normalization step needs its own validation section, given that it changed the headline result.

Venue is settled: ICRA 2027, not RA-L.
