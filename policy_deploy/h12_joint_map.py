"""H1-2 joint mapping + training-time constants shared by all deploy scripts.

The single source of truth for:
  * SDK hardware index per joint name (from unitree_sdk2py's official H1_2_JointIndex)
  * The 13-joint training order used by ``Isaac-H12-Velocity-Legonly-v0``
  * Default joint angles (``_H12_FTP_FLOATING_DEFAULT_JOINT_POS`` in the sim repo)
  * Per-joint PD gains and torque limits matching the training actuator cfg
    (legs+torso block of ``H12_CFG_WITH_INSPIRE_WHOLEBODY`` with the
    env-cfg ankle override applied).

Cross-references (in unitree_sim_isaaclab_Krumi):
  * tasks/h1-2_tasks/h12_velocity/rough_env_cfg.py:46-60   (training joint order)
  * tasks/h1-2_tasks/h12_velocity/rough_env_cfg.py:237-243 (ankle override)
  * robots/unitree.py:1452-1465                            (default joint pose)
  * robots/unitree.py:1340-1394                            (PD gains, torque limits)

WARNING: the H1-2 SDK orders hip joints as (yaw, pitch, roll) while the
training joint-name list orders them (yaw, roll, pitch).  The SDK_INDEX map
below uses the SDK ordering — do NOT assume "i = index in POLICY_JOINT_NAMES
is the SDK motor index".  Use SDK_INDEX[name] explicitly.
"""

from __future__ import annotations

import numpy as np


# ── Training-time joint subset, in EXACT training order ────────────────────
# 13 joints: 12 leg joints + torso.  Order matters: the policy's obs and
# action vectors are built / unpacked in this order with preserve_order=True.
POLICY_JOINT_NAMES: list[str] = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
]
NUM_POLICY_JOINTS = len(POLICY_JOINT_NAMES)   # 13

# Total motor count on the H1-2.
H1_2_NUM_MOTOR = 27


# ── SDK hardware index per joint name ──────────────────────────────────────
# Source: H1_2_JointIndex in
#   Krumi_python_sdk_unitree/example/h1_2/low_level/h1_2_low_level_example.py
# (mirrors the official Unitree H1-2 firmware ordering).
SDK_INDEX: dict[str, int] = {
    # Left leg
    "left_hip_yaw_joint":       0,
    "left_hip_pitch_joint":     1,
    "left_hip_roll_joint":      2,
    "left_knee_joint":          3,
    "left_ankle_pitch_joint":   4,   # alias: LeftAnkleB
    "left_ankle_roll_joint":    5,   # alias: LeftAnkleA
    # Right leg
    "right_hip_yaw_joint":      6,
    "right_hip_pitch_joint":    7,
    "right_hip_roll_joint":     8,
    "right_knee_joint":         9,
    "right_ankle_pitch_joint": 10,   # alias: RightAnkleB
    "right_ankle_roll_joint":  11,   # alias: RightAnkleA
    # Torso
    "torso_joint":             12,
    # Left arm
    "left_shoulder_pitch_joint": 13,
    "left_shoulder_roll_joint":  14,
    "left_shoulder_yaw_joint":   15,
    "left_elbow_joint":          16,
    "left_wrist_roll_joint":     17,
    "left_wrist_pitch_joint":    18,
    "left_wrist_yaw_joint":      19,
    # Right arm
    "right_shoulder_pitch_joint":20,
    "right_shoulder_roll_joint": 21,
    "right_shoulder_yaw_joint":  22,
    "right_elbow_joint":         23,
    "right_wrist_roll_joint":    24,
    "right_wrist_pitch_joint":   25,
    "right_wrist_yaw_joint":     26,
}

# Convenience: list of 13 SDK indices in POLICY_JOINT_NAMES order.
POLICY_SDK_INDICES: list[int] = [SDK_INDEX[n] for n in POLICY_JOINT_NAMES]


# ── Default joint pose (q_default) ─────────────────────────────────────────
# Source: _H12_FTP_FLOATING_DEFAULT_JOINT_POS in robots/unitree.py:1452.
# Any joint not listed defaults to 0.0 (IsaacLab convention).
Q_DEFAULT_BY_NAME: dict[str, float] = {
    "left_hip_pitch_joint":   -0.20,
    "left_knee_joint":         0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_hip_pitch_joint":  -0.20,
    "right_knee_joint":        0.42,
    "right_ankle_pitch_joint":-0.23,
    # Arms (used when holding non-policy joints at default during deploy)
    "left_shoulder_pitch_joint":  0.35,
    "left_shoulder_roll_joint":   0.18,
    "left_elbow_joint":           0.87,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_elbow_joint":          0.87,
}

# 13-vector of defaults aligned with POLICY_JOINT_NAMES.
Q_DEFAULT: np.ndarray = np.array(
    [Q_DEFAULT_BY_NAME.get(n, 0.0) for n in POLICY_JOINT_NAMES],
    dtype=np.float32,
)


# ── PD gains and torque limits ─────────────────────────────────────────────
# Source: H12_CFG_WITH_INSPIRE_WHOLEBODY actuator block (robots/unitree.py:1340)
# with the env-cfg override applied (rough_env_cfg.py:237-243):
#     robot_cfg.actuators["feet"].effort_limit_sim = {ankle_*: 45.0}
#     robot_cfg.actuators["feet"].stiffness        = 20.0
#     robot_cfg.actuators["feet"].damping          = 2.5
KP_BY_NAME: dict[str, float] = {
    "left_hip_yaw_joint":   150.0, "right_hip_yaw_joint":   150.0,
    "left_hip_roll_joint":  150.0, "right_hip_roll_joint":  150.0,
    "left_hip_pitch_joint": 200.0, "right_hip_pitch_joint": 200.0,
    "left_knee_joint":      200.0, "right_knee_joint":      200.0,
    "left_ankle_pitch_joint": 20.0,"right_ankle_pitch_joint":20.0,
    "left_ankle_roll_joint":  20.0,"right_ankle_roll_joint": 20.0,
    "torso_joint":          200.0,
}
KD_BY_NAME: dict[str, float] = {
    "left_hip_yaw_joint":   5.0,  "right_hip_yaw_joint":   5.0,
    "left_hip_roll_joint":  5.0,  "right_hip_roll_joint":  5.0,
    "left_hip_pitch_joint": 5.0,  "right_hip_pitch_joint": 5.0,
    "left_knee_joint":      5.0,  "right_knee_joint":      5.0,
    "left_ankle_pitch_joint": 2.5,"right_ankle_pitch_joint":2.5,
    "left_ankle_roll_joint":  2.5,"right_ankle_roll_joint": 2.5,
    "torso_joint":          5.0,
}
TAU_MAX_BY_NAME: dict[str, float] = {
    "left_hip_yaw_joint":     88.0, "right_hip_yaw_joint":    88.0,
    "left_hip_roll_joint":   139.0, "right_hip_roll_joint":  139.0,
    "left_hip_pitch_joint":   88.0, "right_hip_pitch_joint":  88.0,
    "left_knee_joint":       139.0, "right_knee_joint":      139.0,
    "left_ankle_pitch_joint": 45.0, "right_ankle_pitch_joint":45.0,
    "left_ankle_roll_joint":  45.0, "right_ankle_roll_joint": 45.0,
    "torso_joint":            88.0,
}

# 13-vectors aligned with POLICY_JOINT_NAMES.
KP: np.ndarray      = np.array([KP_BY_NAME[n]      for n in POLICY_JOINT_NAMES], dtype=np.float32)
KD: np.ndarray      = np.array([KD_BY_NAME[n]      for n in POLICY_JOINT_NAMES], dtype=np.float32)
TAU_MAX: np.ndarray = np.array([TAU_MAX_BY_NAME[n] for n in POLICY_JOINT_NAMES], dtype=np.float32)


# ── Policy I/O constants ───────────────────────────────────────────────────
NUM_OBS      = 51              # see deploy_legonly.py docstring for layout
NUM_ACT      = 13
POLICY_SCALE = 0.5             # actions.joint_pos.scale from env cfg
CONTROL_HZ   = 50              # 1 / (physics_dt * decimation) = 1 / (1/200 * 4)
CONTROL_DT   = 1.0 / CONTROL_HZ


# ── Non-policy joint defaults for holding the arms / wrists ────────────────
# Used by deploy scripts to send a stable holding pose for the 14 joints the
# policy doesn't control (arms + wrists).  Soft gains so the held pose
# doesn't fight an operator who manually re-poses the arms.
ARM_HOLD_KP = 40.0
ARM_HOLD_KD = 2.0
