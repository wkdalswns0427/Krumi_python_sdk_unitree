#!/usr/bin/env python3
"""Bar-aware self-collision clearance between the two arms.

The H1-2 in this setup carries a flat bar hand roughly 0.20 m long mounted at
the wrist and pointing along the forearm axis. The wrist frame is therefore not
the end of the arm, and clearance measured between wrist frames understates the
real gap by up to 0.40 m when both forearms point inward.

That understatement is not academic. On the lifting capture the wrist frames
stay 22 cm apart while the bars pass within 1 mm of each other, because the
motion brings both forearms toward the midline.

Everything here treats limbs as line segments and returns the minimum distance
between the two arms across a trajectory. A segment model is optimistic, since
it gives the bar no width and the links no radius, so a margin has to be added
on top rather than trusting the number at face value.

The task-defined target makes this worse rather than better. Pulling each wrist
in toward its own shoulder swings the hands toward the midline, so the
intervention that buys joint clearance costs hand clearance. That is worth
reporting rather than only guarding against.
"""

import numpy as np

HAND_LEN_M = 0.20          # flat bar from the wrist, along the forearm axis
SAFE_MARGIN_M = 0.08       # refuse trajectories closer than this


def _seg_seg(p1, p2, q1, q2):
    """Minimum distance between segment p1p2 and segment q1q2."""
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a, b, c = u @ u, u @ v, v @ v
    d, e = u @ w, v @ w
    D = a * c - b * b
    if D < 1e-12:                       # near parallel
        sc = 0.0
        tc = e / c if c > 1e-12 else 0.0
    else:
        sc = (b * e - c * d) / D
        tc = (a * e - b * d) / D
    sc = min(max(sc, 0.0), 1.0)
    tc = min(max(tc, 0.0), 1.0)
    return float(np.linalg.norm(w + sc * u - tc * v))


def arm_segments(fk, q, hand_len=HAND_LEN_M):
    """[(name, p, q)] for upper arm, forearm and hand bar at configuration q."""
    sh, el, wr = fk.positions(q)
    d = wr - el
    n = np.linalg.norm(d)
    tip = wr + hand_len * (d / n) if n > 1e-9 else wr
    return [("upper", sh, el), ("fore", el, wr), ("bar", wr, tip)]


def clearance(fkR, fkL, qR, qL, hand_len=HAND_LEN_M):
    """Minimum distance between any right-arm and any left-arm segment, and
    which pair produced it."""
    R = arm_segments(fkR, qR, hand_len)
    L = arm_segments(fkL, qL, hand_len)
    best, who = float("inf"), None
    for rn, r1, r2 in R:
        for ln, l1, l2 in L:
            d = _seg_seg(r1, r2, l1, l2)
            if d < best:
                best, who = d, f"R.{rn}-L.{ln}"
    return best, who


def scan(fkR, fkL, QR, QL, hand_len=HAND_LEN_M, margin=SAFE_MARGIN_M):
    """Clearance over a whole trajectory.

    Returns a dict with the minimum, the frame it occurs on, which segment pair
    is responsible, and how many frames fall inside the margin."""
    n = min(len(QR), len(QL))
    d = np.empty(n)
    pair = [None] * n
    for i in range(n):
        d[i], pair[i] = clearance(fkR, fkL, QR[i], QL[i], hand_len)
    i = int(d.argmin())
    return dict(min_m=float(d[i]), argmin_frame=i, pair=pair[i],
                p05_m=float(np.percentile(d, 5)),
                median_m=float(np.median(d)),
                frames_under_margin=int((d < margin).sum()),
                frames_under_02=int((d < 0.02).sum()),
                n_frames=n, safe=bool(d.min() >= margin), per_frame=d)


def verdict(s, margin=SAFE_MARGIN_M):
    if s["min_m"] < 0.02:
        return "COLLISION RISK, do not run"
    if s["min_m"] < margin:
        return f"UNSAFE, closest {s['min_m']*100:.1f} cm is inside the {margin*100:.0f} cm margin"
    return "ok"


# ── torso ────────────────────────────────────────────────────────────────────
# The H1-2 trunk, taken from torso_link.STL: half-width 0.159 m in the shoulder
# band, spanning roughly z 0.05 to 0.62 in the pelvis frame. Modelled as an
# upright capsule, which is generous at the shoulders and tight at the waist.
TORSO_R = 0.159
TORSO_Z = (0.05, 0.62)


def torso_clearance_point(p, r=TORSO_R, zspan=TORSO_Z):
    """Signed distance from a point to the torso capsule. Negative is inside."""
    z = min(max(p[2], zspan[0]), zspan[1])
    return float(np.linalg.norm(p - np.array([0.0, 0.0, z]))) - r


def torso_clearance(fk, q, r=TORSO_R, zspan=TORSO_Z, n=24):
    """Worst signed distance from the upper arm and forearm to the torso."""
    sh, el, wr = fk.positions(q)
    worst = float("inf")
    for a, b in ((sh, el), (el, wr)):
        for t in np.linspace(0.0, 1.0, n):
            worst = min(worst, torso_clearance_point(a + (b - a) * t, r, zspan))
    return worst


def repair_torso(fk, qs, side, margin=0.02, max_deg=25.0, smooth=9):
    """Abduct the shoulder just enough to lift the arm off the torso.

    Only shoulder roll is touched, and only outward. Roll is the joint that
    swings the arm away from the trunk, and the axes are not mirrored on this
    robot, so outward is negative on the right and positive on the left.

    The correction is found per frame by bisection, then smoothed with a moving
    average so the replay does not acquire a velocity step at each repair. The
    same procedure is applied to every condition, so it cannot favour one, and
    the magnitude applied is reported.
    """
    qs = np.asarray(qs, dtype=float).copy()
    sgn = -1.0 if side == "right" else 1.0
    need = np.zeros(len(qs))
    for i, q in enumerate(qs):
        if torso_clearance(fk, q) >= margin:
            continue
        lo, hi = 0.0, np.radians(max_deg)
        for _ in range(24):                      # bisect on the added abduction
            mid = 0.5 * (lo + hi)
            t = q.copy(); t[1] += sgn * mid
            if torso_clearance(fk, t) >= margin:
                hi = mid
            else:
                lo = mid
        need[i] = hi
    if smooth > 1 and need.any():
        # Smoothing spreads each correction over its neighbours so the replay
        # does not acquire a velocity step, but it also pulls the correction
        # down at the frames that needed it most. Take the larger of the two so
        # smoothing can only ever add abduction, never remove what was required.
        k = np.ones(smooth) / smooth
        need = np.maximum(need, np.convolve(need, k, mode="same"))
    qs[:, 1] += sgn * need
    return qs, need
