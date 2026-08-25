#!/usr/bin/env python3
"""Load a per-frame human joints CSV and produce per-frame arm-direction inputs.

This is the "reader" the retargeter needs. Reuses the torso-frame and
arm-directions logic from experiments/scripts/retarget_arm.py by importing
the same functions, so the human side is bit-identical across the two repos.
Wrist positions in the torso frame are also returned (needed for X2/X6 to
give the task-centric IK a target).
"""

import csv
import os
import sys

import numpy as np

EXP = os.path.expanduser(
    "~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/scripts")
if EXP not in sys.path:
    sys.path.insert(0, EXP)
from retarget_arm import (  # noqa: E402
    load_joints_csv, torso_frame, arm_directions, filter_depth_axis)


def load_frames(joints_csv, conf_min=0.3, fps=60.0, z_cutoff=0.8):
    """{frame: {joint: xyz}} with the same conditioning as the IROS retargeter."""
    frames = load_joints_csv(joints_csv, conf_min)
    if z_cutoff > 0:
        filter_depth_axis(frames, fps, z_cutoff)
    return frames


def per_frame_inputs(frames, side, fps=60.0):
    """Yield (t_s, u_hat, f_hat, wrist_torso_xyz) per usable frame.

    u_hat : upper-arm unit vector in the subject's torso frame
    f_hat : forearm unit vector in the subject's torso frame
    wrist_torso_xyz : wrist position in the subject's torso frame, ORIGIN AT
                     the shoulder. This is the target the task-centric IK
                     will chase, but note: subject/robot limb lengths differ,
                     so callers should either (a) scale to the robot's arm
                     length or (b) reconstruct the "robot-scaled" wrist from
                     u_hat, f_hat and the robot's own limb lengths. The
                     retargeter uses (b); we do the same in x6/x2 so the
                     task target is bit-consistent with the IROS pipeline.
    """
    for fi in sorted(frames):
        j = frames[fi]
        R = torso_frame(j)
        if R is None:
            continue
        d = arm_directions(j, R, side)
        if d is None:
            continue
        u_hat, f_hat, _flex = d
        shoulder = R.T @ (j[f"{side}_shoulder"]
                          - 0.5 * (j["left_hip"] + j["right_hip"]))
        wrist = R.T @ (j[f"{side}_wrist"]
                       - 0.5 * (j["left_hip"] + j["right_hip"]))
        wrist_from_shoulder = wrist - shoulder
        yield float(fi) / fps, u_hat, f_hat, wrist_from_shoulder


def robot_scaled_wrist_target(fk, u_hat, f_hat):
    """Reconstruct the wrist target using the ROBOT's own limb lengths.

    Same construction as retarget_arm.py::main:
        target_tip = anchor + l_up * u_hat + l_forearm * f_hat
    using the current FK rest pose to measure l_up and l_forearm. The IK is
    called with q_prev, and it is q_prev that fixes the anchor position; we
    return a callable so the caller can rebuild the target inside its loop.
    """
    def build(q_prev):
        a, e, w = fk.positions(q_prev)
        l_up = float(np.linalg.norm(e - a))
        l_fo = float(np.linalg.norm(w - e))
        return a + l_up * u_hat + l_fo * f_hat
    return build
