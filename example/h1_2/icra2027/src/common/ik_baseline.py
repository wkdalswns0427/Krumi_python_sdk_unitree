#!/usr/bin/env python3
"""B1: fidelity IK (direction-matching), reproducing the IROS retargeter.

Given target unit vectors for upper-arm and forearm directions in the torso
frame, solve for [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow] by
damped Gauss-Newton. This is the field-default (H2O, HumanPlus, OmniH2O,
OKAMI, Liu&Liu all match human configuration or keypoints), and it is
exactly the objective that produces the branch flips this paper analyzes.

The default here mirrors experiments/scripts/retarget_arm.py::solve_arm with
yaw_reg=0 and yaw_neutral=0 (the "before regularizers" version), so the flip
counts we compare against task-centric IK are measured on the naive baseline
rather than on the already-patched one. The regularizers are exposed for
optional comparisons.
"""

import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    return None if n < 1e-9 else v / n


def solve_fidelity(fk, u_hat, f_hat, q_prev, limits4,
                   suppress_yaw=False, yaw_reg=0.0, yaw_neutral=0.0,
                   max_iter=10, damping=1e-2):
    """Direction-matching IK. Returns (q4, residual_deg, clamped_flags)."""
    q = np.array(q_prev, dtype=float)
    yaw_ref = float(q_prev[2])
    lo = np.asarray(limits4["lo"])
    hi = np.asarray(limits4["hi"])
    clamped = np.zeros(4, dtype=bool)

    def residual(qv):
        a, e, w = fk.positions(qv)
        r1 = _unit(e - a)
        r2 = _unit(w - e)
        if r1 is None or r2 is None:
            return None
        return np.concatenate([r1 - u_hat, r2 - f_hat])

    for _ in range(max_iter):
        r = residual(q)
        if r is None:
            break
        J = np.zeros((6, 4))
        eps = 1e-4
        for k in range(4):
            if suppress_yaw and k == 2:
                continue
            qp = q.copy(); qp[k] += eps
            rp = residual(qp)
            if rp is None:
                continue
            J[:, k] = (rp - r) / eps
        JtJ = J.T @ J + damping * np.eye(4)
        rhs = -J.T @ r
        if not suppress_yaw:
            if yaw_reg > 0:
                JtJ[2, 2] += yaw_reg
                rhs[2] -= yaw_reg * (q[2] - yaw_ref)
            if yaw_neutral > 0:
                JtJ[2, 2] += yaw_neutral
                rhs[2] -= yaw_neutral * q[2]
        dq = np.linalg.solve(JtJ, rhs)
        if suppress_yaw:
            dq[2] = 0.0
        q_new = np.clip(q + dq, lo, hi)
        clamped |= (np.abs(q_new - (q + dq)) > 1e-9)
        q = q_new
        if np.linalg.norm(dq) < 1e-4:
            break

    r = residual(q)
    res_deg = float("nan")
    if r is not None:
        a1 = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(r[:3]) / 2, 0, 1)))
        a2 = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(r[3:]) / 2, 0, 1)))
        res_deg = max(a1, a2)
    return q, res_deg, clamped
