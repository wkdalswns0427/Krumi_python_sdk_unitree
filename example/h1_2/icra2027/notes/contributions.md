# Contribution to experiment to evidence traceability

Cross-reference for reviewers and for keeping the paper honest during writing.
Keep in sync with `PLAN_v2.md` §Contributions and §Traceability.

**Rewritten 2026-08-26** for the new C1 to C4 structure. The old numbering is
mapped below so nothing is lost; superseded reasoning stays in the append-only
`notes/decisions.md`.

| New | Content | Was |
| --- | --- | --- |
| C1 | The competition mechanism | new; replaces old C3's causal claim |
| C2 | The retargeting axis with published anchors | new; absorbs old B3 |
| C3 | The trade-off frontier and operating point | reframed from old C2 |
| C4 | Replica versus replacement | old C4, unchanged in spirit |
| dropped | Automated generation as a headline | old C2 |
| dropped | Task/execution decomposition | old C1, made in IROS |

---

## C1 (primary). Posture fidelity blocks the standard feasibility fix

**Claim, lead with the binary form:** null-space limit avoidance takes joint
limit violations from 4/6 to 1/6 when posture is free, and only from 4/6 to
3/6 when posture is pinned. Secondary, report second: posture fidelity
captures 31% of the available margin gain and absorbs the other 69%
(+0.028 against +0.090).

**What this replaced.** The old claim was that inherited posture *causes*
violations. It does not. Without limit avoidance both conditions violate 4/6
regardless of posture weight. What posture does is consume the redundancy the
remedy needs.

**Evidence:**
- `results/b3/factorial_2x2.py` (X8). The core figure. One factor changes per
  cell inside a single solver family, so posture and limit avoidance are not
  confounded.
- `src/x2_costoffidelity.py` (X2). Per-frame fidelity against task-optimal.
- `src/b3_published_baselines.py`. Shows posture weight alone is a weak lever:
  three decades of it buys 0.020 of margin.
- `src/reach_extent_sweep.py` (X9). The curve version. Needs the sweep capture.

**What breaks the claim:**
- If the ZED re-run does not reproduce the interaction.
- If X3 shows the controller silently clamps the violations, making them
  invisible on hardware. That would not refute C1 but it changes what the
  violation count means, and it has to be reported either way.

**Caveats that must appear:**
- Only limit-avoidance weights up to about 0.05 are valid operating points. At
  w=0.20 tip error is 32 mm and at w=0.50 it is 155 mm; those win margin by
  abandoning the task.
- The posture-ON w=0.05 cell has slightly worse tip error (3.05 against
  2.23 mm), so the two are not perfectly matched on task performance.
- n=6, single subject, one robot, arm-only protocol.

---

## C2 (primary). The retargeting axis, anchored by published methods

**Claim:** retargeting objectives are not a bloc. They differ in how hard they
pin arm posture, that difference is measurable on one axis, and the proposed
objective is the zero-posture endpoint of that axis rather than a different
category of method.

**Evidence:**
- `src/common/ik_keypoint.py`. H2O-style and OKAMI-style objectives.
- `src/b3_published_baselines.py`. Eight conditions on one axis, six reps.

**Correction this contribution carries.** The old plan wrote that H2O,
HumanPlus, OmniH2O, OKAMI and Liu and Liu "all track human configuration or
keypoints." OKAMI does not, really: its posture terms sit 12 to 25 times below
its wrist-position term and exist to "maintain natural postures." Citing it as
a configuration tracker overstates the case.

**What breaks the claim:**
- If the adaptation boundary is judged unfaithful. Mitigation is to state it
  explicitly: reproduced is each method's retargeting objective; not
  reproduced is H2O's feasibility filter, imitation policy and sim-to-real,
  and OKAMI's video reconstruction, object warping and finger retargeting.
- OKAMI's weights were tuned for Pink's residual scaling and do not transfer
  verbatim, which is why the posture weight is swept rather than reported at
  one point.

**Objection to pre-empt:** H2O's feasibility filter is H2O's own answer to
infeasibility. Reply: filtering discards the motion whereas we re-solve it,
and the filter operates on whole sequences rather than the per-frame
redundancy choice.

---

## C3 (supporting). The trade-off frontier and where to operate

**Claim:** posture matching is not simply worse. It buys approach direction
almost for free (0.1 deg against 22.2) and costs feasibility. For a replica
task that is the right trade; for a replacement task it is not. Recommended
operating point for replacement: zero posture weight with limit avoidance
around w=0.05, reaching margin 0.351 at 1/6 violations and 2.2 mm tip error.

**Evidence:**
- `src/b3_published_baselines.py`, approach-error column.
- `src/x10_gated_approach.py`. Whether gating approach direction to contact
  frames buys back tolerance without paying posture's cost everywhere.
- `src/common/contact_events.py`. The gate.

**What breaks the claim:**
- If the gated approach term cannot recover approach accuracy at contacts
  without collapsing the null space. The always-on condition in X10 exists to
  show that boundary explicitly.
- Approach error is currently a rep-mean, not contact-only. Manual annotation
  of the ZED captures closes this; see `open_questions.md` item 5.

---

## C4. Replica versus replacement

**Claim:** fidelity is a design choice, not a default. Naming the two modes
turns the finding from a criticism of prior work into guidance. C2's axis
gives the modes a coordinate instead of a label, and C3 gives each mode a
recommended operating point.

**Evidence:**
- `src/x7_dual_mode.py`. Same motion, two objectives, RULA-scored side by side.
- IROS §II.A RULA finding for the replica-side anchor.

**Failure mode:** if both modes land in overlapping posture bands, C4 has to be
argued instead of demonstrated. X7 is pressure valve #2, so this may happen by
choice rather than by result.

---

## F-a, F-b, F-c after the 26 Aug measurements

| Failure | Script | Status |
| --- | --- | --- |
| F-a. Branch flips | `x6_taskcentric.py`, `results/strawman_check/` | **Demoted to a reported symptom.** A scalar yaw regularizer takes the baseline 90 to 15 to 1. Our own IROS paper shipped the regularized version. Report it for the contrast (flips are patchable, margin is not), never lead with it |
| F-b. Joint-limit violations | `x2_costoffidelity.py`, X8 | **Reframed into C1.** Not caused by posture; posture blocks the fix |
| F-c. Physical task failure | `../iros_pilot/s2_pickstack_handsep.csv` | Unchanged and still owed. 51.8 cm hand separation around a 23.8 cm block. Must reconcile with the IROS submission's "all four targets reached" |

---

## Anti-claims. Do NOT resurrect

- **Novel IK solver.** Killed by B2: generic DLS matches `solve_taskcentric`
  on limit margin to 1e-4 across all six reps.
- **Learning contribution.** Killed by X6: a causal single solve suffices.
- **Gravity cost.** Killed by X2: fidelity's gravity load is if anything lower.
- **Manipulability as a headline.** 5.6% is within weight sensitivity.
- **Branch consistency as justification for a learned student.**
- **Automated generation as a headline.** "We generate it with task-space IK"
  is not a contribution when generic DLS is numerically identical.
- **Inherited posture causes joint-limit violations.** Measured false. The
  true claim is C1.
