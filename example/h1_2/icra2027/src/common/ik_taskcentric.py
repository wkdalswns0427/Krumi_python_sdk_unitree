#!/usr/bin/env python3
"""X6 / Robot-Specific: task-centric IK.

Objective:
    L_task(FK(q), x_target) + lambda * R_robot(q, q_prev)

where L_task is tip-position error PLUS, on contact-event frames only, the
forearm approach direction (gate from `contact_events.py`), and R_robot is a
robot-only regularizer
combining joint-limit avoidance, temporal continuity, and a manipulability
push. No posture-fidelity term.

Notes:
- The same tip-position IK exists in iros_pilot/s2_bricklay.py::ArmIK; this
  version adds an explicit limit-avoidance gradient (null-space push away
  from limits) and exposes weights that will be swept in reports.
- The claim is not a novel solver (B2 proved plain DLS + null-space avoidance
  also fixes most flips). The claim is that dropping the posture-fidelity
  term is what removes the ill-conditioning.
- The approach-direction term is what separates this from B2's plain DLS, and
  it is GATED to contact frames on purpose. Applied on every frame it puts 6
  residuals on a 4-DOF arm, leaving no null space for limit avoidance, and the
  solve collapses back onto the human's posture. That is measurably the h2o
  row in results/b3/. Task-relevant tolerance means constraining approach
  where the task needs it and leaving the arm free everywhere else.
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


def forearm_direction(fk, q4):
    """Unit vector along the forearm (elbow to tip) for pose q4."""
    _a, e, w = fk.positions(q4)
    v = w - e
    n = np.linalg.norm(v)
    return None if n < 1e-9 else v / n


def forearm_jacobian(fk, q4, eps=1e-4):
    """3x4 Jacobian of the forearm unit direction. Finite differences."""
    base = forearm_direction(fk, q4)
    if base is None:
        return np.zeros((3, 4))
    J = np.zeros((3, 4))
    for k in range(4):
        qp = np.asarray(q4, dtype=float).copy()
        qp[k] += eps
        d = forearm_direction(fk, qp)
        if d is not None:
            J[:, k] = (d - base) / eps
    return J


def solve_taskcentric(fk, target_tip, q_prev, limits4,
                      lambda_limit=0.05, lambda_cont=0.02, lambda_manip=0.0,
                      target_forearm=None, w_approach=0.0,
                      max_iter=100, damping=1e-2, tol=1e-4):
    """Tip-position IK + null-space robot regularizers, no posture term.

    `target_forearm` with `w_approach > 0` augments the PRIMARY task with the
    forearm approach direction. Pass it only on contact frames; leave it None
    elsewhere. The caller owns the gate (`contact_events.gate_for_capture`).

    Returns (q4, tip_residual_m, iters). The residual is always tip position
    in metres so it stays comparable across gated and free frames.
    """
    use_approach = (target_forearm is not None and w_approach > 0)
    f_target = np.asarray(target_forearm, dtype=float) if use_approach else None
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
        r_task = r

        if use_approach:
            # Augment the primary task with approach direction. On these
            # frames the arm carries 6 residuals over 4 DOF, so the null space
            # collapses and limit avoidance stands down. That is the intended
            # trade: the task is pickier here, so the robot gets less freedom.
            f_now = forearm_direction(fk, q)
            if f_now is not None:
                J = np.vstack([J, w_approach * forearm_jacobian(fk, q)])
                r_task = np.concatenate([r, w_approach * (f_target - f_now)])

        # primary task step: damped least squares toward target
        JtJ = J.T @ J + damping * np.eye(4)
        dq_task = np.linalg.solve(JtJ, J.T @ r_task)

        # null-space push: gradient of robot regularizers projected into null
        N = np.eye(4) - np.linalg.pinv(J) @ J
        g_robot = lambda_limit * limit_avoidance_gradient(q, lo, hi)
        g_robot += lambda_cont * (q - np.asarray(q_prev))
        if lambda_manip > 0:
            # numerical d(manip)/dq
            m0 = manipulability(fk.tip_jacobian(q))
            for k in range(4):
                qp = q.copy(); qp[k] += 1e-4
                Jp = fk.tip_jacobian(qp)
                g_robot[k] -= lambda_manip * (manipulability(Jp) - m0) / 1e-4
        dq_null = -N @ g_robot

        q = np.clip(q + dq_task + dq_null, lo, hi)

    return q, float(np.linalg.norm(target - fk.tip(q))), max_iter
