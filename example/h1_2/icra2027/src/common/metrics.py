#!/usr/bin/env python3
"""Metrics used across X6/X2/B2/X1.

- limit_margin: normalized distance to the nearest joint limit, per config
- normalized_manipulability: sqrt(det(J J^T)) / arm_length^3
- gravity_torque: static tau_g = dU/dq via URDF-derived potential (fk-only)
- flip_count: single-frame yaw jumps whose speed exceeds a threshold
- approach_direction_error_deg: angle between commanded and target forearm dir

All operate on the 4-DOF arm chain (shoulder p/r/y + elbow), matching fk.py.
"""

import numpy as np

G = 9.81


def limit_margin(q, lo, hi):
    """min_i min(q_i - lo_i, hi_i - q_i) / (0.5 * range_i); in [0, 1].

    1.0 = at the midpoint; 0.0 = at a limit. Weight-robust because it is a
    single-joint statistic; the paper reports mean of this across a rep.
    """
    q = np.asarray(q); lo = np.asarray(lo); hi = np.asarray(hi)
    rng = 0.5 * (hi - lo) + 1e-9
    mid = 0.5 * (lo + hi)
    dist = (rng - np.abs(q - mid)) / rng
    return float(np.clip(dist.min(), 0.0, 1.0))


def limit_violation(q, lo, hi, tol=1e-6):
    """True if any joint is within `tol` of, or beyond, a limit."""
    q = np.asarray(q); lo = np.asarray(lo); hi = np.asarray(hi)
    return bool(np.any((q <= lo + tol) | (q >= hi - tol)))


def normalized_manipulability(J, arm_length=0.5):
    """sqrt(det(J J^T)) / arm_length^3, so the number is dimensionless."""
    A = J @ J.T
    d = np.linalg.det(A)
    return float(np.sqrt(max(d, 0.0))) / (arm_length ** 3 + 1e-9)


def flip_count(q_seq, dt, threshold_rad_per_s=6.0, joint_index=2):
    """Count single-frame jumps in q[joint_index] with speed above threshold.

    q_seq is (T, N). Returns integer count. joint_index defaults to shoulder
    yaw (index 2 in the 4-DOF chain), where flips originate.
    """
    q = np.asarray(q_seq)
    if q.ndim != 2 or q.shape[0] < 2:
        return 0
    dq = np.diff(q[:, joint_index]) / dt
    return int(np.sum(np.abs(dq) > threshold_rad_per_s))


def approach_direction_error_deg(q, fk, target_forearm_unit):
    """Angle between the achieved forearm direction and the target unit vector."""
    a, e, w = fk.positions(q)
    v = w - e
    n = np.linalg.norm(v)
    if n < 1e-9:
        return float("nan")
    v = v / n
    return float(np.degrees(np.arccos(np.clip(np.dot(v, target_forearm_unit),
                                              -1.0, 1.0))))


def per_frame_stats(q_series, fk, limits4, dt, target_forearms=None):
    """Aggregate over a full solved trajectory. Returns a dict.

    q_series : (T, 4) joint angles for one arm.
    target_forearms : optional (T, 3) unit vectors for approach-direction error.
    """
    q = np.asarray(q_series)
    lo, hi = limits4["lo"], limits4["hi"]
    margins = [limit_margin(qi, lo, hi) for qi in q]
    manips = []
    for qi in q:
        J = fk.tip_jacobian(qi)
        manips.append(normalized_manipulability(J))
    viol = [limit_violation(qi, lo, hi) for qi in q]
    out = dict(
        n_frames=len(q),
        mean_margin=float(np.mean(margins)),
        min_margin=float(np.min(margins)),
        mean_manip=float(np.mean(manips)),
        any_violation=bool(np.any(viol)),
        violation_frames=int(np.sum(viol)),
        flips_yaw=flip_count(q, dt, joint_index=2),
    )
    if target_forearms is not None:
        errs = [approach_direction_error_deg(qi, fk, fv)
                for qi, fv in zip(q, target_forearms)]
        out["mean_approach_err_deg"] = float(np.nanmean(errs))
        out["p95_approach_err_deg"] = float(np.nanpercentile(errs, 95))
    return out
