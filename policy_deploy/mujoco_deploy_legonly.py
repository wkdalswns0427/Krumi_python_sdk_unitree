#!/usr/bin/env python3
"""Sim-to-sim validator: run ``Isaac-H12-Velocity-Legonly-v0`` in MuJoCo.

Mirrors ``deploy_legonly.py`` (real-robot via Unitree SDK) but uses MuJoCo
as the physics backend instead of low-level DDS to the robot.  Useful for:

  * checking that the exported policy.pt behaves outside IsaacLab without
    risking the real H1-2,
  * comparing IsaacLab vs MuJoCo dynamics (a common sim-to-real proxy),
  * iterating on the deploy loop (obs construction, PD gains, joint map)
    against a known-good visualization with a viewer.

Reuses ``h12_joint_map.py`` for the authoritative training-time joint
order, default pose, kp/kd, and torque limits.  Joint→qpos/qvel/ctrl
indices are resolved from the MJCF at startup.

Observation layout (51 dim, same order as training):

    [0:3]   base_lin_vel_b   (m/s)  ← world-frame qvel[0:3] rotated by R_b_w
    [3:6]   base_ang_vel_b   (rad/s) ← local-frame qvel[3:6] (already body)
    [6:9]   projected_gravity_b     ← R_b_w @ [0, 0, -1]
    [9:12]  velocity_commands       ← [cmd_vx, cmd_vy, cmd_wz]
    [12:25] joint_pos_rel           ← q - q_default, 13 policy joints
    [25:38] joint_vel_rel           ← dq, same order
    [38:51] last_action             ← previous policy output

Action layout (13) → joint torque (motor actuators):
    target_q = q_default + 0.5 * action
    tau      = clip(KP * (target_q - q) + KD * (0 - dq), ±TAU_MAX)
    data.ctrl[i] = tau                    (per policy joint)

Non-policy joints (arms/wrists/torso-yaw if not in subset) are held at
their default pose with soft gains so the upper body doesn't flop.

Usage:
    python3 mujoco_deploy_legonly.py
    python3 mujoco_deploy_legonly.py --cmd_vx 0.5
    python3 mujoco_deploy_legonly.py --mjcf /path/to/h1_2.xml --no_viewer
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import mujoco
import numpy as np
import torch

from h12_joint_map import (
    ARM_HOLD_KD,
    ARM_HOLD_KP,
    CONTROL_HZ,
    KD as POLICY_KD,
    KP as POLICY_KP,
    NUM_ACT,
    NUM_OBS,
    POLICY_JOINT_NAMES,
    POLICY_SCALE,
    Q_DEFAULT,
    Q_DEFAULT_BY_NAME,
    TAU_MAX as POLICY_TAU_MAX,
)


DEFAULT_MJCF = "/home/mchang344/mj_ws/assets/h1_2_description/h1_2.xml"
DEFAULT_POLICY = os.path.join(
    os.path.dirname(__file__), "policies", "legonly_locomotion", "policy.pt"
)

PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION   # = 1 / CONTROL_HZ = 0.02 s

# Spawn pose: matches IsaacLab env_cfg's H12RobotPresets.h12_27dof_inspire_
# wholebody_floating(init_pos=(0,0,1.0), init_rot=(1,0,0,0)).
SPAWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
# Default settle iter count — long enough to absorb the auto-snap-to-floor
# pose adjustment and let joint velocities damp to ~0.  Each iter advances
# physics DECIMATION steps; 20 / 50 Hz = 0.4 s of sim time.
SETTLE_ITERS = 20


# ── Helpers ────────────────────────────────────────────────────────────────

def _quat_rotate_inverse_wxyz(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    """Rotate v by the inverse of unit quaternion (w, x, y, z)."""
    w, x, y, z = (float(q_wxyz[0]), float(q_wxyz[1]),
                  float(q_wxyz[2]), float(q_wxyz[3]))
    qv = np.array([-x, -y, -z], dtype=np.float32)
    v  = v_xyz.astype(np.float32, copy=False)
    t  = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _projected_gravity_b(q_wxyz: np.ndarray) -> np.ndarray:
    """Body-frame gravity unit vector — same as IsaacLab projected_gravity."""
    return _quat_rotate_inverse_wxyz(
        q_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )


def _resolve_indices(model: mujoco.MjModel, joint_names: list[str]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each joint name, return (qpos_idx, dof_idx, ctrl_idx) arrays."""
    qpos_ids, dof_ids, ctrl_ids = [], [], []
    for n in joint_names:
        try:
            j = model.joint(n)
        except KeyError:
            raise RuntimeError(f"Joint '{n}' not found in MJCF.")
        qpos_ids.append(int(j.qposadr[0]))
        dof_ids.append(int(j.dofadr[0]))
        try:
            a = model.actuator(n)
        except KeyError:
            raise RuntimeError(
                f"Actuator named '{n}' not found in MJCF — this script "
                f"assumes motor actuators are named after the joint they "
                f"drive.  Patch the MJCF or update the lookup."
            )
        ctrl_ids.append(int(a.id))
    return (np.array(qpos_ids, dtype=np.int64),
            np.array(dof_ids,  dtype=np.int64),
            np.array(ctrl_ids, dtype=np.int64))


def _all_actuator_joint_names(model: mujoco.MjModel) -> list[str]:
    """List joint names driven by named motor actuators (1:1 by name)."""
    names = []
    for i in range(model.nu):
        n = model.actuator(i).name
        if not n:
            continue
        try:
            model.joint(n)
            names.append(n)
        except KeyError:
            continue
    return names


# ── Main loop ──────────────────────────────────────────────────────────────

class MujocoLegonlyDeploy:
    def __init__(self, mjcf_path: str, policy_path: str,
                 cmd: np.ndarray, action_smoothing: float) -> None:
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data  = mujoco.MjData(self.model)

        # Match training-time physics_dt (1/200).  But MuJoCo defaults to
        # explicit Euler — at dt=0.005 with kp=200 PD, that explodes.  Two
        # post-load patches that together stabilize the same dt training used:
        #   1. Switch to the implicit integrator (closer to IsaacLab's
        #      ImplicitActuatorCfg semantics).
        #   2. Set joint armature = 0.01 on all robot joints (training's
        #      uniform value).  Without armature MuJoCo treats joints as
        #      zero-inertia hinges and stiff PD diverges in a few steps.
        self.model.opt.timestep   = PHYSICS_DT
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = max(int(self.model.opt.iterations), 100)
        # `dof_armature` is indexed by DOF, not joint — set it on every DOF
        # belonging to a hinge/slide joint (skip the 6 free-joint DOFs).
        # Training uses 0.01 with PhysX implicit actuators; on MuJoCo with
        # position actuators (below) the implicit solver handles stability
        # at that value just fine.
        for j_id in range(self.model.njnt):
            j_type = self.model.jnt_type[j_id]
            if j_type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof_addr = int(self.model.jnt_dofadr[j_id])
            self.model.dof_armature[dof_addr] = 0.01
            self.model.dof_damping[dof_addr]  = 0.1

        # CRITICAL: convert the MJCF's <motor> actuators (raw torque, applied
        # against MuJoCo's physics step explicitly) to "position" actuators
        # where MuJoCo integrates the PD force inside the implicit solver.
        # This is much closer to IsaacLab's ImplicitActuatorCfg semantics
        # the policy was trained with — without this conversion the same
        # kp/kd applied as external torque produces destabilization that no
        # armature/damping tweak can fix.
        #
        # Position-actuator math (MuJoCo manual):
        #   force = gain @ [1, ctrl, qpos] + bias @ [1, qpos, qvel]
        # With gainprm=[kp,0,0] biasprm=[0,-kp,-kd]:
        #   force = kp*ctrl - kp*qpos - kd*qvel = kp*(ctrl - qpos) - kd*qvel
        # So ctrl is interpreted as the desired joint position.
        self._joint_name_to_actuator = {}
        for a_id in range(self.model.nu):
            name = self.model.actuator(a_id).name
            if not name:
                continue
            try:
                self.model.joint(name)
                self._joint_name_to_actuator[name] = a_id
            except KeyError:
                continue

        for name, a_id in self._joint_name_to_actuator.items():
            if name in POLICY_JOINT_NAMES:
                idx = POLICY_JOINT_NAMES.index(name)
                kp, kd, tau_max = (float(POLICY_KP[idx]),
                                   float(POLICY_KD[idx]),
                                   float(POLICY_TAU_MAX[idx]))
            else:
                kp, kd, tau_max = ARM_HOLD_KP, ARM_HOLD_KD, 50.0
            self.model.actuator_gaintype[a_id] = mujoco.mjtGain.mjGAIN_AFFINE
            self.model.actuator_biastype[a_id] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_gainprm[a_id]  = 0.0
            self.model.actuator_gainprm[a_id, 0] = kp
            self.model.actuator_biasprm[a_id]  = 0.0
            self.model.actuator_biasprm[a_id, 0] = 0.0
            self.model.actuator_biasprm[a_id, 1] = -kp
            self.model.actuator_biasprm[a_id, 2] = -kd
            # Enforce training-time torque limits via the actuator force
            # range so MuJoCo clamps internally — no Python clip needed.
            self.model.actuator_forcelimited[a_id] = True
            self.model.actuator_forcerange[a_id, 0] = -tau_max
            self.model.actuator_forcerange[a_id, 1] = +tau_max
        print(f"[mujoco] Converted {len(self._joint_name_to_actuator)} "
              f"motor actuators → position actuators with training-time "
              f"kp/kd/tau_max baked in.  PD is now solved implicitly by "
              f"MuJoCo (matches IsaacLab ImplicitActuatorCfg semantics).")

        self.policy = torch.jit.load(policy_path).cpu().eval()
        self.cmd_vec = cmd.astype(np.float32, copy=True)
        self.smoothing = float(action_smoothing)

        # Sanity probe — policy.pt MUST accept 51 → 13.
        with torch.no_grad():
            out = self.policy(torch.zeros(1, NUM_OBS))
            assert out.shape == (1, NUM_ACT), (
                f"Policy I/O mismatch: expected (1,{NUM_ACT}), got "
                f"{tuple(out.shape)}.  Wrong policy.pt?"
            )

        # Pelvis body id (root link the policy treats as "base").  Training
        # uses the asset's root rigid body for root_lin_vel_b / root_ang_vel_b;
        # H1-2's root link is the pelvis.
        try:
            self._pelvis_body_id = int(self.model.body("pelvis").id)
        except KeyError:
            raise RuntimeError("MJCF has no body named 'pelvis' — H1-2 "
                               "root link is expected to be 'pelvis'.")

        # 13 policy joints — qpos/qvel/ctrl indices in training order.
        self.pol_qpos, self.pol_dof, self.pol_ctrl = _resolve_indices(
            self.model, POLICY_JOINT_NAMES
        )
        # Non-policy actuator joints (arms/wrists/etc.) — held at default.
        all_act_joints = _all_actuator_joint_names(self.model)
        nonpolicy_names = [n for n in all_act_joints
                           if n not in POLICY_JOINT_NAMES]
        self.nonpol_qpos, self.nonpol_dof, self.nonpol_ctrl = _resolve_indices(
            self.model, nonpolicy_names
        )
        self.nonpol_qdefault = np.array(
            [Q_DEFAULT_BY_NAME.get(n, 0.0) for n in nonpolicy_names],
            dtype=np.float32,
        )
        print(f"[mujoco] Loaded MJCF: nu={self.model.nu}  nq={self.model.nq}  "
              f"nv={self.model.nv}")
        print(f"[mujoco] Policy joints: {len(POLICY_JOINT_NAMES)}  "
              f"Held joints: {len(nonpolicy_names)}")
        print(f"[mujoco] cmd=(vx={cmd[0]:+.2f}, vy={cmd[1]:+.2f}, "
              f"wz={cmd[2]:+.2f})  smoothing={self.smoothing:.2f}  "
              f"physics_dt={PHYSICS_DT}  decimation={DECIMATION}  "
              f"rate={CONTROL_HZ:.0f} Hz")

        self.last_action     = np.zeros(NUM_ACT, dtype=np.float32)
        self.smoothed_target = self._policy_qdefault().copy()

    def _policy_qdefault(self) -> np.ndarray:
        return Q_DEFAULT.astype(np.float32, copy=False)

    def reset(self) -> None:
        """Author default pose at a spawn height that allows a tiny freefall
        for contact onset.  Energy from that freefall is absorbed in the
        first damping pass of settle(), not by tracking PD that would fight
        the impact.
        """
        mujoco.mj_resetData(self.model, self.data)
        # Spawn pelvis ~5 cm above the natural stand height so the feet
        # have a small drop (~5 cm).  Less than this and intersection at
        # touchdown is possible; more and the impact velocity grows.
        self.data.qpos[0:3] = (0.0, 0.0, 1.00)
        self.data.qpos[3:7] = SPAWN_QUAT
        for k in range(len(POLICY_JOINT_NAMES)):
            self.data.qpos[self.pol_qpos[k]] = float(Q_DEFAULT[k])
        for k in range(len(self.nonpol_qpos)):
            self.data.qpos[self.nonpol_qpos[k]] = float(self.nonpol_qdefault[k])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        print(f"[mujoco] Reset — pelvis_z={float(self.data.qpos[2]):+.3f}")

    def _apply_pd(self, kp_scale: float, kd_scale: float,
                  track_position: bool) -> None:
        """One control iter — write POSITION targets to ctrl (PD is now
        implicit inside MuJoCo's position actuators).  ``kp_scale``/
        ``kd_scale`` are ignored here since gains are baked into the
        actuators at init.  ``track_position=False`` writes the current q
        as target (zero positional pull) for impact-absorbing settle.
        """
        if track_position:
            self.data.ctrl[self.pol_ctrl] = Q_DEFAULT.astype(np.float64)
            if len(self.nonpol_ctrl) > 0:
                self.data.ctrl[self.nonpol_ctrl] = self.nonpol_qdefault.astype(np.float64)
        else:
            self.data.ctrl[self.pol_ctrl] = self.data.qpos[self.pol_qpos].copy()
            if len(self.nonpol_ctrl) > 0:
                self.data.ctrl[self.nonpol_ctrl] = self.data.qpos[self.nonpol_qpos].copy()
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

    def settle(self) -> None:
        """Full PD-to-default for 1 s before policy takes over.

        With armature/damping authored to compensate for the IsaacLab→MuJoCo
        integrator gap, full PD is stable from t=0 and keeps the bent-knee
        default pose upright through the small freefall touchdown.  No
        damping-only pre-phase needed (and a damping-only phase actively
        hurts here — the legs go limp and the robot collapses before PD
        can catch it).
        """
        for _ in range(50):    # 1.0 s @ 50 Hz
            self._apply_pd(kp_scale=1.0, kd_scale=1.0, track_position=True)
        pz  = float(self.data.qpos[2])
        vxy = float(np.hypot(self.data.qvel[0], self.data.qvel[1]))
        wxy = float(np.hypot(self.data.qvel[3], self.data.qvel[4]))
        max_jv = float(np.max(np.abs(self.data.qvel[self.pol_dof])))
        print(f"[mujoco] Settled (damping + tracking) — pelvis_z={pz:+.3f}  "
              f"|v_xy|={vxy:+.3f}  |w_xy|={wxy:+.3f}  max|jvel|={max_jv:+.3f}")

    def _build_obs(self) -> np.ndarray:
        base_quat = self.data.qpos[3:7].copy()  # (w,x,y,z)
        # Use mj_objectVelocity to get UNAMBIGUOUS body-frame velocities.
        # The raw qvel convention for free joints differs across MuJoCo
        # versions/forks (linear is consistently world; angular has flipped
        # between world and body) — this API call is the canonical fix.
        # flg_local=1 → returned (angular, linear) both in body frame.
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            int(self._pelvis_body_id), vel6, 1,
        )
        ang_vel_b = vel6[0:3].astype(np.float32)   # angular first in mj_objectVelocity
        lin_vel_b = vel6[3:6].astype(np.float32)   # then linear, both body frame
        proj_g    = _projected_gravity_b(base_quat)

        q  = self.data.qpos[self.pol_qpos].astype(np.float32)
        dq = self.data.qvel[self.pol_dof ].astype(np.float32)

        obs = np.concatenate([
            lin_vel_b,                                            # 3
            ang_vel_b,                                            # 3
            proj_g,                                               # 3
            self.cmd_vec,                                         # 3
            (q - Q_DEFAULT).astype(np.float32),                   # 13
            dq,                                                   # 13
            self.last_action,                                     # 13
        ])
        assert obs.shape[0] == NUM_OBS
        return obs

    def _step_policy_once(self) -> None:
        obs = self._build_obs()
        with torch.no_grad():
            a = self.policy(torch.from_numpy(obs).unsqueeze(0)
                            ).squeeze(0).cpu().numpy().astype(np.float32)
        self.last_action = a.copy()

        raw_target = Q_DEFAULT + POLICY_SCALE * a
        if self.smoothing > 0.0:
            self.smoothed_target[:] = (
                self.smoothing * self.smoothed_target
                + (1.0 - self.smoothing) * raw_target
            )
            target_q = self.smoothed_target
        else:
            target_q = raw_target

        # Position target → ctrl.  MuJoCo's position actuators compute
        # tau = kp*(ctrl - q) - kd*qvel internally, clamped to forcerange
        # (we set both at init).  This is the equivalent of IsaacLab's
        # ImplicitActuatorCfg path and is much more stable than computing
        # PD in Python.
        self.data.ctrl[self.pol_ctrl] = target_q.astype(np.float64)

        # Hold non-policy joints at their default pose.
        if len(self.nonpol_ctrl) > 0:
            self.data.ctrl[self.nonpol_ctrl] = self.nonpol_qdefault.astype(np.float64)

        # Advance physics DECIMATION times — each ctrl is held throughout,
        # matching env_cfg's physics_dt / decimation contract.
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mjcf",   type=str, default=DEFAULT_MJCF,
                   help=f"H1-2 MJCF path (default: {DEFAULT_MJCF}).")
    p.add_argument("--policy", type=str, default=DEFAULT_POLICY,
                   help=f"TorchScript policy.pt (default: {DEFAULT_POLICY}).")
    p.add_argument("--cmd_vx", type=float, default=0.5)
    p.add_argument("--cmd_vy", type=float, default=0.0)
    p.add_argument("--cmd_wz", type=float, default=0.0)
    p.add_argument("--action_smoothing", type=float, default=0.0,
                   help="EMA on target_q (0=passthrough, 0.5=moderate, "
                        "0.9=heavy).  Use for the Legonly policy at "
                        "low cmd_vx where it wasn't trained to stand still.")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Seconds to run (0 = forever).")
    p.add_argument("--no_viewer", action="store_true",
                   help="Headless run (no GUI).  Useful for logging-only "
                        "comparisons against IsaacLab.")
    return p.parse_args()


def _log_state(t: float, deploy: MujocoLegonlyDeploy) -> None:
    q = deploy.data.qpos
    v = deploy.data.qvel
    pelvis_z = float(q[2])
    vxy = float(np.hypot(v[0], v[1]))
    print(f"[mujoco] t={t:5.1f}s  pelvis_z={pelvis_z:+.3f}  "
          f"|v_xy|={vxy:+.3f} m/s  cmd_vx={deploy.cmd_vec[0]:+.2f}")


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.mjcf):
        print(f"[mujoco] MJCF not found: {args.mjcf}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.policy):
        print(f"[mujoco] policy not found: {args.policy}", file=sys.stderr)
        sys.exit(2)

    cmd = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float32)
    deploy = MujocoLegonlyDeploy(args.mjcf, args.policy, cmd,
                                 args.action_smoothing)
    deploy.reset()
    deploy.settle()

    # 50 Hz outer loop.
    t_start  = time.time()
    next_log = 1.0
    period   = CONTROL_DT

    if args.no_viewer:
        while True:
            iter_t0 = time.time()
            deploy._step_policy_once()
            t = time.time() - t_start
            if args.duration > 0.0 and t >= args.duration:
                break
            if t >= next_log:
                _log_state(t, deploy)
                next_log += 1.0
            sleep = period - (time.time() - iter_t0)
            if sleep > 0.0:
                time.sleep(sleep)
    else:
        # Passive viewer: we drive the sim ourselves, viewer just renders.
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(deploy.model, deploy.data) as v:
            while v.is_running():
                iter_t0 = time.time()
                deploy._step_policy_once()
                t = time.time() - t_start
                if args.duration > 0.0 and t >= args.duration:
                    break
                if t >= next_log:
                    _log_state(t, deploy)
                    next_log += 1.0
                v.sync()
                sleep = period - (time.time() - iter_t0)
                if sleep > 0.0:
                    time.sleep(sleep)


if __name__ == "__main__":
    main()
