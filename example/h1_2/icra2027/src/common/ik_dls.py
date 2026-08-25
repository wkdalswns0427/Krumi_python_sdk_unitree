#!/usr/bin/env python3
"""B2: textbook DLS position IK with null-space limit avoidance.

This is the control that B2 was intended to establish: standard damped
least squares for tip-position, with a null-space clamped-joint-limit
avoidance term (Chiaverini/Siciliano-style, no manipulability, no continuity
weight). The purpose is to show that a generic method also removes most flips
but at the cost of ~12 deg of approach-direction error at contact events
because it optimizes only the wrist position and ignores forearm direction.

Interface is identical to ik_taskcentric.solve_taskcentric but with the
options fixed to the "textbook" version.
"""

import numpy as np

from .ik_taskcentric import limit_avoidance_gradient


def solve_dls(fk, target_tip, q_prev, limits4,
              lambda_limit=0.05, max_iter=100, damping=1e-2, tol=1e-4):
    """Plain DLS position IK + null-space limit avoidance. Returns (q, res, it)."""
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
        JtJ = J.T @ J + damping * np.eye(4)
        dq_task = np.linalg.solve(JtJ, J.T @ r)
        N = np.eye(4) - np.linalg.pinv(J) @ J
        dq_null = -N @ (lambda_limit * limit_avoidance_gradient(q, lo, hi))
        q = np.clip(q + dq_task + dq_null, lo, hi)

    return q, float(np.linalg.norm(target - fk.tip(q))), max_iter
