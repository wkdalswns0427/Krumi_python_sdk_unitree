#!/usr/bin/env python3
"""Unit tests for angles.py on synthetic poses.

Poses are built in a y-DOWN frame (like both pipelines), up = (0, -1, 0).
Run:  /usr/bin/python3 -m pytest test_angles.py -q
      /usr/bin/python3 test_angles.py
"""

import sys

import pytest

from angles import compute_frame_angles, legs_support_info


def _standing():
    """Upright, arms hanging, legs straight. All angles near neutral."""
    return {
        "nose": (0.0, -0.72, 2.42),
        "left_ear": (0.08, -0.75, 2.5), "right_ear": (-0.08, -0.75, 2.5),
        "left_shoulder": (0.18, -0.55, 2.5), "right_shoulder": (-0.18, -0.55, 2.5),
        "left_elbow": (0.18, -0.25, 2.5), "right_elbow": (-0.18, -0.25, 2.5),
        "left_wrist": (0.18, 0.05, 2.5), "right_wrist": (-0.18, 0.05, 2.5),
        "left_hip": (0.15, 0.0, 2.5), "right_hip": (-0.15, 0.0, 2.5),
        "left_knee": (0.15, 0.45, 2.5), "right_knee": (-0.15, 0.45, 2.5),
        "left_ankle": (0.15, 0.90, 2.5), "right_ankle": (-0.15, 0.90, 2.5),
    }


def test_standing_neutral():
    a = compute_frame_angles(_standing())
    assert abs(a["trunk_flexion"]) < 3
    assert abs(a["trunk_side_bend"]) < 3
    assert abs(a["trunk_twist"]) < 3
    assert abs(a["neck_flexion"]) < 3
    for side in ("left", "right"):
        assert abs(a[f"upper_arm_flexion_{side}"]) < 3
        assert abs(a[f"lower_arm_flexion_{side}"]) < 3
        assert a[f"knee_flexion_{side}"] > 177          # ~180 = straight leg
        assert a[f"wrist_{side}"] is None               # neutral-wrist rule


def test_trunk_bow_90():
    j = _standing()
    # Bend forward: shoulders and head swing to hip height, toward the camera.
    j["left_shoulder"] = (0.18, 0.0, 1.95)
    j["right_shoulder"] = (-0.18, 0.0, 1.95)
    j["left_ear"] = (0.08, 0.0, 1.75)
    j["right_ear"] = (-0.08, 0.0, 1.75)
    j["nose"] = (0.0, 0.03, 1.72)
    a = compute_frame_angles(j)
    assert abs(a["trunk_flexion"] - 90) < 4
    assert abs(a["neck_flexion"]) < 4                   # head follows the trunk


def test_overhead_reach():
    j = _standing()
    for side, sx in (("left", 0.18), ("right", -0.18)):
        j[f"{side}_elbow"] = (sx, -0.85, 2.5)           # elbow above shoulder
        j[f"{side}_wrist"] = (sx, -1.15, 2.5)           # wrist above elbow
    a = compute_frame_angles(j)
    for side in ("left", "right"):
        assert abs(a[f"upper_arm_flexion_{side}"] - 180) < 4
        assert abs(a[f"lower_arm_flexion_{side}"]) < 4  # arm straight overhead


def test_deep_squat():
    j = _standing()
    j["left_hip"] = (0.15, 0.35, 2.7)
    j["right_hip"] = (-0.15, 0.35, 2.7)                 # hips drop and shift back
    j["left_knee"] = (0.15, 0.45, 2.45)
    j["right_knee"] = (-0.15, 0.45, 2.45)               # knees forward
    j["left_shoulder"] = (0.18, -0.15, 2.55)
    j["right_shoulder"] = (-0.18, -0.15, 2.55)
    a = compute_frame_angles(j)
    for side in ("left", "right"):
        assert a[f"knee_flexion_{side}"] < 120          # strongly bent
        assert abs(a[f"knee_flexion_{side}"] - 105) < 8
    assert 8 < a["trunk_flexion"] < 30                  # forward lean in a squat


def test_confidence_gating_returns_none():
    j = _standing()
    conf = {name: 0.99 for name in j}
    conf["left_elbow"] = 0.2                            # below the 0.5 default
    a = compute_frame_angles(j, conf=conf)
    assert a["upper_arm_flexion_left"] is None
    assert a["lower_arm_flexion_left"] is None
    assert a["upper_arm_flexion_right"] is not None     # other side unaffected
    a2 = compute_frame_angles({k: v for k, v in j.items() if k != "right_hip"})
    assert a2["trunk_flexion"] is None                  # missing landmark -> None


def test_legs_support_info():
    info = legs_support_info(_standing())
    assert abs(info["stance_width_m"] - 0.30) < 0.02
    assert abs(info["hip_width_m"] - 0.30) < 0.02
    assert info["knee_flexion_left"] > 177


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
