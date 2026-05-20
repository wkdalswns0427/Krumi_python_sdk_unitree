# H1-2 Policy Deploy

Real-robot deployment scripts for RL policies trained in
`unitree_sim_isaaclab_Krumi`.

```
policy_deploy/
├── h12_joint_map.py        # SDK index ↔ training joint name; gains; defaults
├── damping_test.py         # safety bring-up (run FIRST every session)
├── deploy_legonly.py       # Isaac-H12-Velocity-Legonly-v0 deploy loop
├── policies/
│   ├── legonly_locomotion/
│   │   ├── policy.pt       # TorchScript-exported actor MLP (51 → 13)
│   │   └── policy.onnx
│   └── squat/              # reserved
└── README.md
```

## Prerequisites

```bash
pip install torch numpy
# unitree_sdk2py is the local SDK in ../unitree_sdk2py — already importable
# from this folder if the SDK is installed in editable mode (pip install -e ..).
```

Confirm the policy file matches what's in the simulator repo:

```bash
md5sum policies/legonly_locomotion/policy.pt
md5sum <unitree_sim_isaaclab_Krumi>/logs/rsl_rl/h12_velocity_legonly/<run>/exported/policy.pt
```

## Safety prerequisites — read every time

- **Hang the robot on an overhead support so the feet are barely touching.**
- **ESTOP within arm's reach.** Unitree H1-2 ESTOP cuts low-level motor power.
- **No bystanders in the swing volume.**
- **Run `damping_test.py` first** to verify wiring before any policy command goes out.

## Bring-up procedure (every session, in order)

1. **Power on the robot**, wait for the embedded controller to settle,
   confirm the ESTOP is *not* engaged.
2. **Network**: plug Ethernet to the robot, set your NIC to the robot's
   subnet (default `192.168.123.x/24`), verify `ping 192.168.123.161`.
3. **Damping test**:

   ```bash
   python3 damping_test.py eth0
   ```

   Expected: robot goes limp with mild damping. No twitches, no oscillation,
   IMU rpy printed once per second. If a joint snaps, your CRC is wrong,
   `mode_machine` isn't set, or the embedded motion mode wasn't released —
   **stop here and debug**.
4. **Deploy legonly policy with zero command** and a long ramp:

   ```bash
   python3 deploy_legonly.py --iface eth0 --cmd_vx 0 --ramp_seconds 5
   ```

   Gains and target ramp linearly from `(0, current_q)` to `(KP, policy_q)`
   over 5 seconds. Robot should hold its standing pose.
5. **Watch for**:
   - **Ankle chatter** → gain mismatch; recheck `h12_joint_map.py:KP/KD`.
   - **Persistent forward/backward drift at `cmd_vx=0`** → the
     `base_lin_vel` zeroing problem (see `deploy_legonly.py` header).
     Acceptable for bring-up; for any real walking you need a state
     estimator (option A or B in the file header) or a blind-retrained
     policy (option D).
   - **One-leg pendulum** → obs is built wrong; verify joint order
     mapping against `policy.pt`'s 51-dim input layout.
6. **Lower the gantry so feet take ~30% body weight.** Re-run zero command.
7. **Ramp commanded speed**: `--cmd_vx 0.1` → `0.2` → ... in 0.1 m/s steps,
   30 s hold at each. Training range was `(0.0, 2.0)` m/s for x — don't
   deploy near the top of the training range until small commands are
   reliable.
8. **Don't unhang the robot** until:
   - `cmd_vx=0` holds pose >60 s
   - `cmd_vx=0.3` walks stable for 30 s
   - Stop (`--cmd_vx 0`) and turn (`--cmd_wz ±0.3`) work without falling.

## Joint mapping (the single biggest bug source)

`h12_joint_map.py` maps the 13 training-time joint names to their H1-2 SDK
hardware indices. **The H1-2 SDK orders hip joints as
`(yaw, pitch, roll)`** — the training joint-name list orders them
`(yaw, roll, pitch)`. The map handles this correctly, but if you ever
hand-edit the constants, verify by physically moving one joint at a time
while logging `motor_state[i].q` and confirming which SDK index changes.

```python
SDK_INDEX = {
    "left_hip_yaw_joint":  0,  "left_hip_pitch_joint":  1,  "left_hip_roll_joint":  2,
    "left_knee_joint":     3,  "left_ankle_pitch_joint":4,  "left_ankle_roll_joint":5,
    "right_hip_yaw_joint": 6,  "right_hip_pitch_joint": 7,  "right_hip_roll_joint": 8,
    "right_knee_joint":    9,  "right_ankle_pitch_joint":10,"right_ankle_roll_joint":11,
    "torso_joint":        12,
    # arms / wrists at 13..26 — held at default pose during legonly deploy
}
```

## Observation contract (51 dim)

| Slice | Source signal |
|---|---|
| `[0:3]` `base_lin_vel` | body-frame linear velocity — **currently zeroed** (see file header) |
| `[3:6]` `base_ang_vel` | `imu.gyroscope` |
| `[6:9]` `projected_gravity` | derived from `imu.quaternion` (w, x, y, z) |
| `[9:12]` `velocity_commands` | `[cmd_vx, cmd_vy, cmd_wz]` from CLI |
| `[12:25]` `joint_pos_rel` | `q - q_default` for the 13 policy joints, training order |
| `[25:38]` `joint_vel_rel` | `dq` for same 13 joints |
| `[38:51]` `last_action` | previous policy output |

Action contract: `target_q = q_default + 0.5 * action` (scale from
`actions.joint_pos.scale` in the env config).

## Authoritative source (`unitree_sim_isaaclab_Krumi`)

- Joint name list, action scale, env-cfg overrides:
  `tasks/h1-2_tasks/h12_velocity/rough_env_cfg.py`
- Default joint pose:
  `robots/unitree.py` → `_H12_FTP_FLOATING_DEFAULT_JOINT_POS`
- PD gains, torque limits:
  `robots/unitree.py` → `H12_CFG_WITH_INSPIRE_WHOLEBODY` actuator block
- Sim-side reference inference loop:
  `scene/joint_motion.py` — useful as a golden reference; if the same policy
  walks correctly there but the real robot misbehaves, the bug is in the
  deploy plumbing, not the policy.

## Failure-mode reference

| Symptom | Likely cause |
|---|---|
| Joint snaps violently on first cmd | SDK index map wrong, or skipped damping ramp |
| All joints whine, no motion | `mode_machine` not propagated, CRC wrong, or `motor_cmd[].mode != 1` |
| Drifts forward/backward at `cmd_vx=0` | `projected_gravity` quat convention wrong, OR base_lin_vel zeroing (see file header) |
| Falls backward consistently | `Q_DEFAULT` mismatch (must equal training defaults to ~0.01 rad) |
| Ankles oscillate | Don't stiffen ankles; training used soft `kp=20` so the policy could command corrections |
| Embedded controller overwriting cmd | `MotionSwitcherClient.ReleaseMode()` didn't run or didn't take — re-init |
