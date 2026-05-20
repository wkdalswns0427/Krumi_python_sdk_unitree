#!/usr/bin/env python3
"""Deploy ``Isaac-H12-Velocity-Legonly-v0`` policy on a real H1-2.

Loads a TorchScript-exported policy (``policy.pt``), closes a 50 Hz control
loop reading ``rt/lowstate`` and publishing ``rt/lowcmd`` with PD targets
matching the training-time actuator config.

Observation layout (51 dim, exactly as built at training time):

    [0:3]     base_lin_vel        (m/s, body frame)            ← see WARNING
    [3:6]     base_ang_vel        (rad/s, body frame)
    [6:9]     projected_gravity   (unit vec, body frame)
    [9:12]    velocity_commands   (cmd_vx, cmd_vy, cmd_wz)
    [12:25]   joint_pos_rel       (q - q_default), 13 joints in training order
    [25:38]   joint_vel_rel       (dq), same order
    [38:51]   last_action         (previous policy output)

Action layout (13 dim) → joint target:
    target_q[i] = q_default[i] + 0.5 * action[i]              (scale = 0.5)

WARNING on base_lin_vel
-----------------------
The H1-2 LowState does NOT publish a reliable linear velocity.  The
training-time policy used the ground-truth body-frame velocity.  Options:
    (A) Run a legged-state KF and fill base_lin_vel from it     (BEST)
    (B) Compute v_b from stance-foot Jacobian                   (medium)
    (C) Zero it (default below)                                 (BRING-UP ONLY)
    (D) Retrain a blind variant with base_lin_vel removed       (LONG TERM)
Option C is acceptable only at very low commanded speeds.  Expect drift.

SAFETY
------
First run: robot HANGING, ESTOP in reach, ``--cmd_vx 0``, ``--ramp_seconds 5``.

Usage:
    python3 deploy_legonly.py --iface eth0
    python3 deploy_legonly.py --iface eth0 --cmd_vx 0.3
    python3 deploy_legonly.py --iface eth0 \\
        --policy ./policies/legonly_locomotion/policy.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from h12_joint_map import (
    ARM_HOLD_KD,
    ARM_HOLD_KP,
    CONTROL_DT,
    CONTROL_HZ,
    H1_2_NUM_MOTOR,
    KD,
    KP,
    NUM_ACT,
    NUM_OBS,
    POLICY_JOINT_NAMES,
    POLICY_SCALE,
    POLICY_SDK_INDICES,
    Q_DEFAULT,
    Q_DEFAULT_BY_NAME,
    SDK_INDEX,
)


DEFAULT_POLICY = os.path.join(
    os.path.dirname(__file__), "policies", "legonly_locomotion", "policy.pt"
)
MODE_PR = 0   # series-control mode for pitch/roll (URDF-convention joint angles)


# Body-frame gravity from quaternion (Unitree IMU convention: w,x,y,z).
# Equivalent to R_w_b.T @ [0, 0, -1].
def _projected_gravity_from_quat(qw: float, qx: float, qy: float, qz: float
                                 ) -> np.ndarray:
    return np.array([
        2.0 * (qx * qz - qw * qy),
        2.0 * (qy * qz + qw * qx),
        -1.0 + 2.0 * (qw * qw + qz * qz),
    ], dtype=np.float32)


class LegonlyDeploy:
    """50 Hz inference loop: state → obs → policy → PD command."""

    def __init__(self, policy_path: str, cmd: np.ndarray,
                 ramp_seconds: float) -> None:
        self._policy = torch.jit.load(policy_path).cpu().eval()
        self._cmd_vec = cmd.astype(np.float32, copy=True)  # (3,) cmd_vx,vy,wz
        self._ramp_seconds = float(ramp_seconds)

        self._low_cmd: LowCmd_ = unitree_hg_msg_dds__LowCmd_()
        self._low_state: Optional[LowState_] = None
        self._mode_machine: int = 0
        self._mode_machine_seen: bool = False
        self._crc = CRC()

        self._last_action = np.zeros(NUM_ACT, dtype=np.float32)
        self._t_start: Optional[float] = None

        # Pre-cache: 14 non-policy motor indices + their hold targets so the
        # arms / wrists / etc. stay at a stable pose during deploy.
        self._nonpolicy_idx: list[int] = [
            i for i in range(H1_2_NUM_MOTOR) if i not in POLICY_SDK_INDICES
        ]
        inv = {v: k for k, v in SDK_INDEX.items()}
        self._nonpolicy_targets: dict[int, float] = {
            i: Q_DEFAULT_BY_NAME.get(inv[i], 0.0)
            for i in self._nonpolicy_idx if i in inv
        }

    # ── DDS plumbing ───────────────────────────────────────────────────────
    def init(self) -> None:
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        while result.get("name"):
            print(f"[deploy] Releasing motion mode: {result['name']}")
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1.0)
        print("[deploy] Motion mode released.")

        self._cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._cmd_pub.Init()

        self._state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._state_sub.Init(self._on_lowstate, 16)

        print("[deploy] Waiting for first lowstate...")
        while not self._mode_machine_seen:
            time.sleep(0.05)
        print(f"[deploy] First lowstate received "
              f"(mode_machine={self._mode_machine}).")

    def _on_lowstate(self, msg: LowState_) -> None:
        self._low_state = msg
        if not self._mode_machine_seen:
            self._mode_machine = msg.mode_machine
            self._mode_machine_seen = True

    def start(self) -> None:
        self._t_start = time.time()
        self._thread = RecurrentThread(
            interval=CONTROL_DT,
            target=self._step,
            name="legonly_ctrl",
        )
        self._thread.Start()

    # ── Inference + command loop ───────────────────────────────────────────
    def _build_obs(self, state: LowState_) -> np.ndarray:
        # 13-vector of policy-joint q, dq (SDK → policy order)
        q  = np.array([state.motor_state[i].q  for i in POLICY_SDK_INDICES],
                      dtype=np.float32)
        dq = np.array([state.motor_state[i].dq for i in POLICY_SDK_INDICES],
                      dtype=np.float32)

        imu = state.imu_state
        base_ang_vel = np.array(imu.gyroscope, dtype=np.float32)   # rad/s
        qw, qx, qy, qz = imu.quaternion                            # (w,x,y,z)
        proj_g = _projected_gravity_from_quat(qw, qx, qy, qz)
        base_lin_vel = np.zeros(3, dtype=np.float32)               # see WARNING

        obs = np.concatenate([
            base_lin_vel,                                          # 3
            base_ang_vel,                                          # 3
            proj_g,                                                # 3
            self._cmd_vec,                                         # 3
            (q - Q_DEFAULT),                                       # 13
            dq,                                                    # 13
            self._last_action,                                     # 13
        ])
        assert obs.shape[0] == NUM_OBS, (obs.shape, NUM_OBS)
        return obs

    def _step(self) -> None:
        if self._low_state is None:
            return
        state = self._low_state

        obs = self._build_obs(state)

        with torch.no_grad():
            action = self._policy(
                torch.from_numpy(obs).unsqueeze(0)
            ).squeeze(0).cpu().numpy().astype(np.float32)

        self._last_action = action
        target_q = Q_DEFAULT + POLICY_SCALE * action

        # Ramp: linearly blend from current q → target_q AND 0 → KP/KD over
        # ``ramp_seconds`` so the very first cmd doesn't step from limp to
        # full gains.
        elapsed = time.time() - self._t_start
        if self._ramp_seconds > 0 and elapsed < self._ramp_seconds:
            alpha = elapsed / self._ramp_seconds
            q_now = np.array(
                [state.motor_state[i].q for i in POLICY_SDK_INDICES],
                dtype=np.float32,
            )
            target_q = (1.0 - alpha) * q_now + alpha * target_q
            kp_now, kd_now = alpha * KP, alpha * KD
        else:
            kp_now, kd_now = KP, KD

        # Fill LowCmd: 13 policy joints with policy targets + ramped gains.
        self._low_cmd.mode_pr = MODE_PR
        self._low_cmd.mode_machine = self._mode_machine
        for k, hw_i in enumerate(POLICY_SDK_INDICES):
            m = self._low_cmd.motor_cmd[hw_i]
            m.mode = 1
            m.q    = float(target_q[k])
            m.dq   = 0.0
            m.kp   = float(kp_now[k])
            m.kd   = float(kd_now[k])
            m.tau  = 0.0

        # Hold every non-policy motor (arms, wrists) at its default pose
        # with soft gains so an operator can re-pose them by hand if needed.
        for hw_i in self._nonpolicy_idx:
            m = self._low_cmd.motor_cmd[hw_i]
            m.mode = 1
            m.q    = float(self._nonpolicy_targets.get(hw_i, 0.0))
            m.dq   = 0.0
            m.kp   = ARM_HOLD_KP
            m.kd   = ARM_HOLD_KD
            m.tau  = 0.0

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._cmd_pub.Write(self._low_cmd)


def _print_safety_banner(cmd_vx: float, cmd_vy: float, cmd_wz: float,
                         ramp: float, policy_path: str) -> None:
    print("=" * 72)
    print(" H1-2 LEGONLY POLICY DEPLOY")
    print("=" * 72)
    print(f"   policy       : {policy_path}")
    print(f"   command      : (vx, vy, wz) = ({cmd_vx:+.2f}, {cmd_vy:+.2f},"
          f" {cmd_wz:+.2f})")
    print(f"   ramp seconds : {ramp:.2f}")
    print(f"   control rate : {CONTROL_HZ} Hz")
    print(f"   PD gains     : training-time values from "
          f"H12_CFG_WITH_INSPIRE_WHOLEBODY + env override")
    print(f"   base_lin_vel : ZEROED (LowState lacks reliable lin vel — see "
          f"file header)")
    print("=" * 72)
    print(" SAFETY:")
    print("   - Robot HANGING on overhead support for first run")
    print("   - ESTOP within arm's reach")
    print("   - Run damping_test.py FIRST to verify wiring")
    print("   - Start with --cmd_vx 0; ramp commanded speed in 0.1 m/s steps")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--iface", type=str, default=None,
                   help="NIC connected to the robot (e.g. eth0).  If omitted, "
                        "uses the SDK default network init.")
    p.add_argument("--policy", type=str, default=DEFAULT_POLICY,
                   help=f"Path to TorchScript policy.pt "
                        f"(default: {DEFAULT_POLICY}).")
    p.add_argument("--cmd_vx", type=float, default=0.0)
    p.add_argument("--cmd_vy", type=float, default=0.0)
    p.add_argument("--cmd_wz", type=float, default=0.0)
    p.add_argument("--ramp_seconds", type=float, default=5.0,
                   help="Linear blend from current pose/0 gains to policy "
                        "output/full gains over this duration.  Set 0 to "
                        "skip the ramp (dangerous).")
    p.add_argument("--yes", action="store_true",
                   help="Skip the safety-banner Enter prompt.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.policy):
        print(f"[deploy] policy not found: {args.policy}", file=sys.stderr)
        sys.exit(2)

    _print_safety_banner(args.cmd_vx, args.cmd_vy, args.cmd_wz,
                         args.ramp_seconds, args.policy)
    if not args.yes:
        input("Press Enter to continue (Ctrl+C to abort)...")

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    cmd = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float32)
    deploy = LegonlyDeploy(args.policy, cmd, args.ramp_seconds)
    deploy.init()
    deploy.start()

    print(f"[deploy] Inference loop running at {CONTROL_HZ} Hz. "
          "Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
            st = deploy._low_state
            if st is not None:
                rpy = list(st.imu_state.rpy)
                print(f"[deploy] rpy = [{rpy[0]:+.2f}, {rpy[1]:+.2f}, "
                      f"{rpy[2]:+.2f}]")
    except KeyboardInterrupt:
        print("\n[deploy] Stopped.")


if __name__ == "__main__":
    main()
