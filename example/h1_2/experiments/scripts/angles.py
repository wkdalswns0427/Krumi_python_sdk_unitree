#!/usr/bin/env python3
"""angles.py - derived body angles for RULA/REBA scoring.

Pure functions that turn one frame of the full joint set (bag_to_joints
--joints full: canonical names, metric xyz) into the angles the RULA/REBA
scorer consumes. No file IO here; the scorer owns thresholds and scoring.

Conventions
-----------
- Angles in degrees.
- Input frames are y-DOWN in both pipelines (MediaPipe world and the camera
  optical frame), so the default up axis is (0, -1, 0). In rgbd mode "up"
  being the negated camera y assumes a level camera; the capture metadata
  records camera height and leveling. Pass a custom up vector otherwise.
- Every angle returns None when any landmark it needs is missing or has
  confidence below conf_min (default 0.5). The scorer decides how to handle
  gaps; nothing is interpolated here.
- Wrist angles are NOT computable reliably from MediaPipe Pose (no stable
  hand reference), so wrist_left/wrist_right are always None and the scorer
  applies the neutral-wrist assumption (RULA wrist score 1).
- Face landmarks (nose, ears) are EXCLUDED from the pipeline export (user
  decision 2026-07-09), so on pipeline CSVs neck_flexion returns None and the
  scorer applies a neutral-neck assumption, exactly like the wrist. The
  function itself stays implemented (and unit-tested) for synthetic input or
  future data that does carry head landmarks.

Angle definitions
-----------------
upper_arm_flexion : angle between the upper arm (shoulder -> elbow) and the
    trunk-down direction (mid_shoulder -> mid_hip). 0 = arm hanging along the
    trunk, 90 = horizontal, 180 = full overhead reach.
lower_arm_flexion : elbow flexion = 180 minus the interior angle at the elbow
    between (elbow -> shoulder) and (elbow -> wrist). 0 = straight arm.
trunk_flexion : angle of the trunk vector (mid_hip -> mid_shoulder) from the
    up axis. 0 = upright, 90 = horizontal bow.
trunk_side_bend : best-effort lateral lean: arcsin of the trunk vector
    component along the hip line (left_hip -> right_hip direction).
    Approximation: treats the hip line as the body's lateral axis.
trunk_twist : best-effort axial twist: angle between the shoulder line and the
    hip line after projecting both onto the plane perpendicular to the trunk
    vector. Approximation: shoulder motion (e.g. one-arm reach) can register
    as twist.
neck_flexion : angle between the head vector (mid_shoulder -> head point) and
    the trunk vector. Head point is mid-ear when both ears pass conf_min,
    else the nose (documented best effort; the nose sits forward of the head
    centre so nose-based values read a few degrees flexed).
knee_flexion : interior hip-knee-ankle angle per side. ~180 = straight leg,
    smaller = more flexion (deep squat ~60-110).

legs_support_info() returns stance width, hip width, and both knee angles so
the scorer can apply the REBA leg rules (bilateral support, knee bend bands).
"""

import numpy as np

UP_DEFAULT = (0.0, -1.0, 0.0)   # y-down frames: up is negative y


# ── small vector helpers ─────────────────────────────────────────────────────
def _unit(v):
    n = np.linalg.norm(v)
    return None if n < 1e-9 else v / n


def _angle_deg(u, v):
    """Angle between two vectors in degrees, or None on degenerate input."""
    uu, vv = _unit(np.asarray(u, float)), _unit(np.asarray(v, float))
    if uu is None or vv is None:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(uu, vv), -1.0, 1.0))))


def _get(joints, conf, conf_min, *names):
    """Return [np.array(xyz), ...] for names, or None if any is missing or
    below the confidence threshold."""
    out = []
    for n in names:
        if n not in joints:
            return None
        if conf is not None and conf.get(n, 1.0) < conf_min:
            return None
        out.append(np.asarray(joints[n], dtype=float))
    return out


def _mid(a, b):
    return (a + b) / 2.0


# ── per-angle functions ──────────────────────────────────────────────────────
def trunk_vector(joints, conf=None, conf_min=0.5):
    """mid_hip -> mid_shoulder, or None."""
    g = _get(joints, conf, conf_min,
             "left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if g is None:
        return None
    ls, rs, lh, rh = g
    return _mid(ls, rs) - _mid(lh, rh)


def upper_arm_flexion(joints, side, conf=None, conf_min=0.5):
    g = _get(joints, conf, conf_min, f"{side}_shoulder", f"{side}_elbow")
    t = trunk_vector(joints, conf, conf_min)
    if g is None or t is None:
        return None
    shoulder, elbow = g
    return _angle_deg(elbow - shoulder, -t)          # -t = trunk-down


def lower_arm_flexion(joints, side, conf=None, conf_min=0.5):
    g = _get(joints, conf, conf_min,
             f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
    if g is None:
        return None
    shoulder, elbow, wrist = g
    interior = _angle_deg(shoulder - elbow, wrist - elbow)
    return None if interior is None else 180.0 - interior


def trunk_flexion(joints, conf=None, conf_min=0.5, up=UP_DEFAULT):
    t = trunk_vector(joints, conf, conf_min)
    return None if t is None else _angle_deg(t, up)


def trunk_side_bend(joints, conf=None, conf_min=0.5):
    t = trunk_vector(joints, conf, conf_min)
    g = _get(joints, conf, conf_min, "left_hip", "right_hip")
    if t is None or g is None:
        return None
    lh, rh = g
    lat = _unit(rh - lh)
    tu = _unit(t)
    if lat is None or tu is None:
        return None
    return float(np.degrees(np.arcsin(np.clip(np.dot(tu, lat), -1.0, 1.0))))


def trunk_twist(joints, conf=None, conf_min=0.5):
    t = trunk_vector(joints, conf, conf_min)
    g = _get(joints, conf, conf_min,
             "left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if t is None or g is None:
        return None
    ls, rs, lh, rh = g
    tu = _unit(t)
    if tu is None:
        return None

    def _proj(v):
        return v - np.dot(v, tu) * tu               # onto plane perp. to trunk

    return _angle_deg(_proj(rs - ls), _proj(rh - lh))


def neck_flexion(joints, conf=None, conf_min=0.5):
    t = trunk_vector(joints, conf, conf_min)
    if t is None:
        return None
    sh = _get(joints, conf, conf_min, "left_shoulder", "right_shoulder")
    if sh is None:
        return None
    ears = _get(joints, conf, conf_min, "left_ear", "right_ear")
    if ears is not None:
        head = _mid(*ears)
    else:
        nose = _get(joints, conf, conf_min, "nose")
        if nose is None:
            return None
        head = nose[0]
    return _angle_deg(head - _mid(*sh), t)


def knee_flexion(joints, side, conf=None, conf_min=0.5):
    g = _get(joints, conf, conf_min,
             f"{side}_hip", f"{side}_knee", f"{side}_ankle")
    if g is None:
        return None
    hip, knee, ankle = g
    return _angle_deg(hip - knee, ankle - knee)      # interior angle


def wrist_flexion(joints, side, conf=None, conf_min=0.5):
    """Always None: MediaPipe Pose has no reliable hand direction. The scorer
    applies the neutral-wrist assumption (documented in the module header)."""
    return None


def legs_support_info(joints, conf=None, conf_min=0.5, up=UP_DEFAULT):
    """Stance geometry for the REBA leg rules. Returns a dict with
    stance_width_m (horizontal ankle separation), hip_width_m, and per-side
    knee flexion; entries are None when landmarks are missing/low-conf."""
    upv = _unit(np.asarray(up, float))
    ank = _get(joints, conf, conf_min, "left_ankle", "right_ankle")
    stance = None
    if ank is not None and upv is not None:
        d = ank[1] - ank[0]
        stance = float(np.linalg.norm(d - np.dot(d, upv) * upv))
    hips = _get(joints, conf, conf_min, "left_hip", "right_hip")
    hip_w = None if hips is None else float(np.linalg.norm(hips[1] - hips[0]))
    return dict(
        stance_width_m=stance,
        hip_width_m=hip_w,
        knee_flexion_left=knee_flexion(joints, "left", conf, conf_min),
        knee_flexion_right=knee_flexion(joints, "right", conf, conf_min),
    )


# ── one-call frame summary ───────────────────────────────────────────────────
def compute_frame_angles(joints, conf=None, conf_min=0.5, up=UP_DEFAULT):
    """All RULA/REBA input angles for one frame.

    joints: {name: (x, y, z)} metric positions (canonical names).
    conf:   {name: confidence} optional; below-threshold landmarks make the
            angles that need them return None.
    """
    return {
        "upper_arm_flexion_left": upper_arm_flexion(joints, "left", conf, conf_min),
        "upper_arm_flexion_right": upper_arm_flexion(joints, "right", conf, conf_min),
        "lower_arm_flexion_left": lower_arm_flexion(joints, "left", conf, conf_min),
        "lower_arm_flexion_right": lower_arm_flexion(joints, "right", conf, conf_min),
        "wrist_left": wrist_flexion(joints, "left", conf, conf_min),
        "wrist_right": wrist_flexion(joints, "right", conf, conf_min),
        "trunk_flexion": trunk_flexion(joints, conf, conf_min, up),
        "trunk_side_bend": trunk_side_bend(joints, conf, conf_min),
        "trunk_twist": trunk_twist(joints, conf, conf_min),
        "neck_flexion": neck_flexion(joints, conf, conf_min),
        "knee_flexion_left": knee_flexion(joints, "left", conf, conf_min),
        "knee_flexion_right": knee_flexion(joints, "right", conf, conf_min),
    }
