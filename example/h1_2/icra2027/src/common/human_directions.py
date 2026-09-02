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
    load_joints_csv, torso_frame, arm_directions, filter_depth_axis, OneEuro)

# Human-side conditioning frozen in the IROS retargeter's main loop. These are
# NOT optional. Without them the solver receives raw per-frame directions, the
# shoulder-yaw branch becomes ambiguous frame to frame, and yaw swings through
# the full joint range instead of the roughly +/-0.5 rad the tool produces.
MAX_DIR_RATE_DEG = 600.0   # teleport gate on limb-direction rotation
RELOCK = 5                 # accept a fast frame after this many rejections
EURO_MIN_CUTOFF = 1.0
EURO_BETA = 0.5


def _dir_rate_deg(a, b, dt):
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return np.degrees(np.arccos(c)) / max(dt, 1e-6)


def load_frames(joints_csv, conf_min=0.3, fps=60.0, z_cutoff=0.0):
    """{frame: {joint: xyz}} with the conditioning the retargeter expects.

    z_cutoff defaults to 0 (off) for the ZED era. ZED SDK body tracking is
    3D-native with no noisy per-keypoint depth. Its z axis is up, not depth, so
    the iPhone depth low-pass would smooth real vertical motion. The iPhone
    preview used z_cutoff=0.8. Pass it explicitly to reproduce that."""
    frames = load_joints_csv(joints_csv, conf_min)
    if z_cutoff > 0:
        filter_depth_axis(frames, fps, z_cutoff)
    return frames


def per_frame_inputs(frames, side, fps=60.0, with_flex=False):
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
    euro = OneEuro(EURO_MIN_CUTOFF, EURO_BETA) if EURO_MIN_CUTOFF > 0 else None
    last_dir = None
    rejects = 0
    for fi in sorted(frames):
        j = frames[fi]
        R = torso_frame(j)
        if R is None:
            continue
        d = arm_directions(j, R, side)
        if d is None:
            continue
        u_hat, f_hat, _flex_raw = d
        t_now = float(fi) / fps

        # Teleport gate against the last ACCEPTED directions. A skeleton glitch
        # that rotates a limb faster than a person can move is dropped, not
        # solved, so it cannot seed the next frame with a bad branch.
        if last_dir is not None:
            t_ref, u_ref, f_ref = last_dir
            dt = max(t_now - t_ref, 1e-6)
            rate = max(_dir_rate_deg(u_hat, u_ref, dt),
                       _dir_rate_deg(f_hat, f_ref, dt))
            if rate > MAX_DIR_RATE_DEG and rejects < RELOCK:
                rejects += 1
                continue
        rejects = 0
        last_dir = (t_now, u_hat, f_hat)

        # One-Euro on the direction components, then renormalize.
        if euro is not None:
            filt = euro(np.concatenate([u_hat, f_hat]), t_now)
            u_f = filt[:3] / np.linalg.norm(filt[:3]) if np.linalg.norm(filt[:3]) > 1e-9 else None
            f_f = filt[3:] / np.linalg.norm(filt[3:]) if np.linalg.norm(filt[3:]) > 1e-9 else None
            if u_f is not None and f_f is not None:
                u_hat, f_hat = u_f, f_f

        # flex from the FILTERED directions, as the tool does
        flex_deg = float(np.degrees(np.arccos(
            np.clip(np.dot(u_hat, f_hat), -1.0, 1.0))))

        shoulder = R.T @ (j[f"{side}_shoulder"]
                          - 0.5 * (j["left_hip"] + j["right_hip"]))
        wrist = R.T @ (j[f"{side}_wrist"]
                       - 0.5 * (j["left_hip"] + j["right_hip"]))
        wrist_from_shoulder = wrist - shoulder
        if with_flex:
            yield t_now, u_hat, f_hat, wrist_from_shoulder, flex_deg
        else:
            yield t_now, u_hat, f_hat, wrist_from_shoulder


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


# ext_cap is a trade-off parameter, not a tuned constant, and the paper
# reports both ends of it. Swept on 25 captures per arm at the corrected
# 0.5588 m arm length:
#
#   0.88  primary. Limit contact 6/25 right and 5/25 left, 11/50 combined.
#         Task attainment after one base step: 25/25 and 25/25.
#   0.84  contact-minimising. 10/50 combined, the joint minimum for both
#         arms, but attainment falls to 22/25 and 21/25.
#
# One capture of joint clearance against seven of task attainment, so 0.88
# is the operating point and 0.84 is reported alongside it.
def task_defined_wrist_target(fk, u_hat, f_hat, ext_scale=1.0, ext_cap=0.88):
    """Wrist target defined by the ACTION, not by the human's arm extension.

    robot_scaled_wrist_target uses the robot's limb lengths but keeps the human's
    arm directions, so the magnitude of the resulting target carries the human's
    elbow angle. A human reaching at 99% extension therefore asks the robot for
    99% extension, which is near-singular and against the joint limits. That is
    posture fidelity arriving through the back door: extension is a property of
    the configuration, not of the task.

    Here the reach DIRECTION is kept, because that is where the action happens,
    and the reach MAGNITUDE is re-set into a workspace the robot can serve well:

        ext_robot = min(ext_human * ext_scale, ext_cap)

    ext_scale rescales a capture's whole reach profile, so the relative shape of
    the motion, reaching further at one moment than another, is preserved while
    the range is moved into the robot's well-conditioned band. ext_cap is the
    hard ceiling. The measured sweep puts the violation threshold near 90%
    extension, so 0.88 keeps the arm just inside it.
    """
    def build(q_prev):
        a, e, w = fk.positions(q_prev)
        l_up = float(np.linalg.norm(e - a))
        l_fo = float(np.linalg.norm(w - e))
        arm = l_up + l_fo
        v = l_up * u_hat + l_fo * f_hat
        L = float(np.linalg.norm(v))
        if L < 1e-9 or arm < 1e-9:
            return a + v
        d_hat = v / L
        ext = min((L / arm) * ext_scale, ext_cap)
        return a + d_hat * (ext * arm)
    return build


def human_extension_ratio(fk, rows, q_rest):
    """p95 of the human-derived extension ratio over a capture's frames.

    Used to compute the per-capture ext_scale so a whole motion is mapped into
    the robot's usable range rather than clipped at the top."""
    a, e, w = fk.positions(q_rest)
    l_up = float(np.linalg.norm(e - a))
    l_fo = float(np.linalg.norm(w - e))
    arm = l_up + l_fo
    exts = []
    for row in rows:
        u_hat, f_hat = row[1], row[2]
        v = l_up * u_hat + l_fo * f_hat
        exts.append(float(np.linalg.norm(v)) / arm if arm > 1e-9 else 0.0)
    return float(np.percentile(exts, 95)) if exts else 1.0


def robot_scaled_arm_targets(fk, u_hat, f_hat):
    """Both elbow AND wrist targets, using the ROBOT's own limb lengths.

    Same construction as robot_scaled_wrist_target, but also returns the
    intermediate elbow point, which the H2O-style keypoint baseline needs
    (H2O matches elbow and wrist positions, not just the end effector).

    Building targets from the robot's limb lengths is the analogue of H2O's
    SMPL shape-fitting step, which optimises human bone lengths to match the
    robot before retargeting. Here the same effect is had in closed form.

    Returns a callable build(q_prev) -> (elbow_target, wrist_target).
    """
    def build(q_prev):
        a, e, w = fk.positions(q_prev)
        l_up = float(np.linalg.norm(e - a))
        l_fo = float(np.linalg.norm(w - e))
        elbow_t = a + l_up * u_hat
        wrist_t = elbow_t + l_fo * f_hat
        return elbow_t, wrist_t
    return build
