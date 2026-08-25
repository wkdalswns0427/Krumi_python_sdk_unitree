#!/usr/bin/env python3
"""X6 / Robot-Specific: task-centric IK.

Objective:
    L_task(FK(q), x_target) + lambda * R_robot(q, q_prev)

where L_task is tip-position error (extended in x1_generated to include
approach direction at contact events) and R_robot is a robot-only regularizer
combining joint-limit avoidance, temporal continuity, and a manipulability
push. No posture-fidelity term.

Notes:
- The same tip-position IK exists in iros_pilot/s2_bricklay.py::ArmIK; this
  version adds an explicit limit-avoidance gradient (null-space push away
  from limits) and exposes weights that will be swept in reports.
- The claim is not a novel solver (B2 proved plain DLS + null-space avoidance
  also fixes most flips). The claim is that dropping the posture-fidelity
  term is what removes the ill-conditioning.
"""

import numpy as np


def limit_avoidance_gradient(q, lo, hi):
    """Gradient of a smooth (mid-range) potential; drives q toward midpoints.

    U(q) = sum_i (q_i - mid_i)^2 / (range_i)^2 ; grad w.r.t. q_i.
    Weight applied by caller.
    """
    mid = 0.5 * (lo + hi)
    rng = 0.5 * (hi - lo) + 1e-6
    return 2.0 * (q - mid) / (rng ** 2)


def manipulability(J):
    """Yoshikawa manipulability sqrt(det(J J^T)) for a 3xN Jacobian."""
    A = J @ J.T
    d = np.linalg.det(A)
    return float(np.sqrt(max(d, 0.0)))


def solve_taskcentric(fk, target_tip, q_prev, limits4,
                      lambda_limit=0.05, lambda_cont=0.02, lambda_manip=0.0,
                      max_iter=100, damping=1e-2, tol=1e-4):
    """Tip-position IK + null-space robot regularizers, no posture term.

    Returns (q4, residual_m, iters).
    """
    q = np.array(q_prev, dtype=float)
    lo = np.asarray(limits4["lo"])
    hi = np.asarray(limits4["hi"])
    target = np.asarray(target_tip, dtype=float)

    for it in range(max_iter):
        tip = fk.tip(q)
        r = target - tip
        if np.linalg.norm(r) < tol:
            return q, float(np.linalg.norm(r)), it
        J = fk.tip_jacobian(q)

        # primary task step: damped least squares toward target
        JtJ = J.T @ J + damping * np.eye(4)
        dq_task = np.linalg.solve(JtJ, J.T @ r)

        # null-space push: gradient of robot regularizers projected into null
        N = np.eye(4) - np.linalg.pinv(J) @ J
        g_robot = lambda_limit * limit_avoidance_gradient(q, lo, hi)
        g_robot += lambda_cont * (q - np.asarray(q_prev))
        if lambda_manip > 0:
            # numerical d(manip)/dq
            m0 = manipulability(J)
            for k in range(4):
                qp = q.copy(); qp[k] += 1e-4
                Jp = fk.tip_jacobian(qp)
                g_robot[k] -= lambda_manip * (manipulability(Jp) - m0) / 1e-4
        dq_null = -N @ g_robot

        q = np.clip(q + dq_task + dq_null, lo, hi)

    return q, float(np.linalg.norm(target - fk.tip(q))), max_iter
