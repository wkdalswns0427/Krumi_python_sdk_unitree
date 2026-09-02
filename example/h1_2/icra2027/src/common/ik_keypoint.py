#!/usr/bin/env python3
"""B3: published-method baselines, adapted to the H1-2 4-DOF arm.

The plan's defence against the strawman objection is that B1 is "the field's
default, since H2O, HumanPlus, OmniH2O, OKAMI and Liu and Liu all track human
configuration or keypoints". This module implements two of those methods
directly so the claim is measured rather than asserted.

Both are adaptations, not reimplementations, and the paper must say so. What
is reproduced is each method's RETARGETING OBJECTIVE, which is the part this
paper argues about. What is not reproduced is everything downstream: H2O's
privileged imitation policy and sim-to-data filtering, OKAMI's object-aware
trajectory warping and dex-retargeting for fingers. Those are separable from
the retargeting stage and orthogonal to the feasibility question.

H2O (He et al., IROS 2024, arXiv 2403.04436)
--------------------------------------------
Retargets by "minimizing the 12 joint position differences using Adam", after
a shape-fitting step that optimises SMPL shape parameters so the human's bone
lengths match the robot's. The matched set explicitly includes elbows and
wrists. It matches keypoint POSITIONS, not local joint angles, because
"direct copying the local joint angles from human to humanoid ... can lead to
large differences in end-effector positions". No explicit redundancy
resolution is described.

Adaptation here: equal-weight position matching on the two arm keypoints
inside our scope (elbow, wrist). The shape-fitting step is reproduced by
construction, since targets are built with the ROBOT's own limb lengths
(`human_directions.robot_scaled_arm_targets`), which is what shape fitting
achieves. Damped Gauss-Newton stands in for Adam; both are local descent on
the same objective, and the objective is what is being compared.

OKAMI (Li et al., CoRL 2024, arXiv 2410.11792)
-----------------------------------------------
Weighted multi-objective IK (via the Pink library) on shoulder orientation,
elbow orientation, wrist orientation and wrist position, with weights
0.04, 0.04, 0.08 and 1.0 respectively, chosen to "maintain natural postures".
So OKAMI is wrist-position dominant with posture entering as a weak
regulariser, roughly 25x below the position term.

Adaptation here: wrist ORIENTATION is dropped, because the protocol inherited
from IROS fixes the three wrist joints at zero, so there is no DOF that could
serve it. Shoulder and elbow orientation become upper-arm and forearm
direction matching, which is what those link orientations reduce to for a
4-DOF chain. Weights keep OKAMI's published ratio.

Scale caveat, and it must be reported: OKAMI's weights were tuned for Pink's
own residual scaling. Here the position residual is in metres (order 0.5) and
the direction residuals are unit-vector differences (order 0 to 2), so the
published ratio does not transfer verbatim. `solve_okami_style` therefore
takes `pos_scale` to put the terms on a common footing, and the driver sweeps
the posture weight so the result is reported as a dose-response curve rather
than as one arbitrary point.
"""

import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    return None if n < 1e-9 else v / n


def _damped_gn_step(J, r, damping, n=4):
    """One damped Gauss-Newton step for residual r with Jacobian J."""
    JtJ = J.T @ J + damping * np.eye(n)
    return np.linalg.solve(JtJ, -J.T @ r)


def _solve(residual, q_prev, limits4, max_iter=30, damping=1e-2, eps=1e-4,
           tol=1e-6):
    """Generic damped Gauss-Newton over the 4-DOF arm with limit clamping.

    `residual(q)` returns a 1-D array (already weighted) or None if the pose
    is degenerate. Returns (q, final_residual_norm, iters).
    """
    q = np.array(q_prev, dtype=float)
    lo = np.asarray(limits4["lo"])
    hi = np.asarray(limits4["hi"])
    it = 0
    for it in range(max_iter):
        r = residual(q)
        if r is None:
            break
        J = np.zeros((len(r), 4))
        for k in range(4):
            qp = q.copy()
            qp[k] += eps
            rp = residual(qp)
            if rp is None:
                continue
            J[:, k] = (rp - r) / eps
        dq = _damped_gn_step(J, r, damping)
        q = np.clip(q + dq, lo, hi)
        if np.linalg.norm(dq) < tol:
            break
    r = residual(q)
    res = float(np.linalg.norm(r)) if r is not None else float("nan")
    return q, res, it + 1


def solve_h2o_style(fk, elbow_target, wrist_target, q_prev, limits4,
                    w_elbow=1.0, w_wrist=1.0, max_iter=30, damping=1e-2):
    """H2O-style keypoint position matching on (elbow, wrist).

    Equal weights by default, matching H2O's unweighted sum over its 12
    matched joint positions. Returns (q4, residual_m, iters).
    """
    et = np.asarray(elbow_target, dtype=float)
    wt = np.asarray(wrist_target, dtype=float)

    def residual(qv):
        _a, e, w = fk.positions(qv)
        return np.concatenate([w_elbow * (e - et), w_wrist * (w - wt)])

    return _solve(residual, q_prev, limits4, max_iter=max_iter,
                  damping=damping)


def solve_okami_style(fk, wrist_target, u_hat, f_hat, q_prev, limits4,
                      w_pos=1.0, w_shoulder=0.04, w_elbow=0.08,
                      pos_scale=1.0, w_limit=0.0, max_iter=30, damping=1e-2):
    """OKAMI-style weighted IK: wrist position dominant, posture as a weak term.

    Published weights are shoulder orientation 0.04, elbow orientation 0.04,
    wrist orientation 0.08, wrist position 1.0. Wrist orientation is dropped
    here (wrist joints fixed at zero by the protocol), so its 0.08 is carried
    onto the forearm-direction term, which is the closest surviving proxy for
    distal orientation.

    `pos_scale` multiplies the position residual so metre-scale and
    unit-vector-scale terms are commensurate. `pos_scale=1.0` reproduces the
    published ratio literally; the driver sweeps it.

    Returns (q4, residual, iters).
    """
    wt = np.asarray(wrist_target, dtype=float)
    u = np.asarray(u_hat, dtype=float)
    f = np.asarray(f_hat, dtype=float)

    lo = np.asarray(limits4["lo"])
    hi = np.asarray(limits4["hi"])
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) + 1e-9

    def residual(qv):
        a, e, w = fk.positions(qv)
        ru = _unit(e - a)
        rf = _unit(w - e)
        if ru is None or rf is None:
            return None
        terms = [
            w_pos * pos_scale * (w - wt),
            w_shoulder * (ru - u),
            w_elbow * (rf - f),
        ]
        if w_limit > 0:
            # Joint-centering term. This is Pink's PostureTask toward
            # mid-range, i.e. the standard way limit avoidance is configured
            # in exactly the solver OKAMI uses. Included so posture fidelity
            # and limit avoidance can be varied INDEPENDENTLY: without it,
            # any comparison against a null-space-avoiding solver confounds
            # the two, which is the reviewer objection PLAN_v2.md anticipates.
            terms.append(w_limit * (qv - mid) / half)
        return np.concatenate(terms)

    return _solve(residual, q_prev, limits4, max_iter=max_iter,
                  damping=damping)
