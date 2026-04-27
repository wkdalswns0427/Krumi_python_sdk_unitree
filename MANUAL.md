# User Manual: Activating Inspire Hands on H1-2 (unitree_sdk2_py_Krumi)

---

## Important Disclaimer

**This repository does NOT contain Inspire-hand-specific examples or DDS topic names for the hands.** It provides the IDL message types (`HandCmd_`, `HandState_`) that you need to build hand control, but the actual DDS topic names must be confirmed from [Unitree's official H1-2 developer docs](https://support.unitree.com). Everything below is grounded exactly in what the repository contains.

---

## 1. Prerequisites

**Hardware:** H1-2 humanoid robot with Inspire DEX3 dexterous hands physically installed and powered on.

**Software dependencies:**

```bash
# Python >= 3.8
sudo apt install python3-pip

# CycloneDDS (exact version required)
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install

# Install the SDK
cd ~/mj_ws/unitree_sdk2_py_Krumi
export CYCLONEDDS_HOME=~/cyclonedds/install
pip3 install -e .
```

**Network:** Connect your PC to the robot's Ethernet port. Identify your network interface name (e.g. `enp2s0`) with `ip link show`.

---

## 2. Architecture Overview

The H1-2 uses two independent DDS control channels:

| Channel | Topic (body) | Message Type | Purpose |
|---|---|---|---|
| Body low-level | `rt/lowcmd` | `HGLowCmd_` | Controls 27 body motors |
| Body state | `rt/lowstate` | `HGLowState_` | Reads body + IMU state |
| **Hand command** | `rt/inspire/lefthand` / `rt/inspire/righthand` | `HGHandCmd_` | Controls 7 hand motors each |
| **Hand state** | (corresponding state topic) | `HGHandState_` | Reads hand motor + pressure state |

> **Verify the hand topic names** against your firmware version at support.unitree.com. The body topics (`rt/lowcmd`, `rt/lowstate`) are confirmed in [example/h1_2/low_level/h1_2_low_level_example.py](example/h1_2/low_level/h1_2_low_level_example.py).

---

## 3. H1-2 Body Joint Map (27 motors)

From [example/h1_2/low_level/h1_2_low_level_example.py](example/h1_2/low_level/h1_2_low_level_example.py):

```
Index  Name
────────────────────────────────
 0     LeftHipYaw
 1     LeftHipPitch
 2     LeftHipRoll
 3     LeftKnee
 4     LeftAnklePitch  (= LeftAnkleB)
 5     LeftAnkleRoll   (= LeftAnkleA)
 6     RightHipYaw
 7     RightHipPitch
 8     RightHipRoll
 9     RightKnee
10     RightAnklePitch (= RightAnkleB)
11     RightAnkleRoll  (= RightAnkleA)
12     WaistYaw
13     LeftShoulderPitch
14     LeftShoulderRoll
15     LeftShoulderYaw
16     LeftElbow
17     LeftWristRoll
18     LeftWristPitch
19     LeftWristYaw
20     RightShoulderPitch
21     RightShoulderRoll
22     RightShoulderYaw
23     RightElbow
24     RightWristRoll
25     RightWristPitch
26     RightWristYaw
```

The hands (fingers) are **not** part of this 27-motor array — they are separate devices.

---

## 4. Hand Message Types

From [unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandCmd_.py](unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandCmd_.py) and [_HandState_.py](unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandState_.py):

**`HandCmd_`** — what you publish to move fingers:
```python
HandCmd_:
  motor_cmd: sequence[MotorCmd_]   # 7 elements, one per finger/digit
  reserve:   uint32[4]
```

**`HandState_`** — what you read back:
```python
HandState_:
  motor_state:        sequence[MotorState_]        # 7 finger states
  press_sensor_state: sequence[PressSensorState_]  # 7 tactile sensors
  imu_state:          IMUState_
  power_v:   float32   # power rail voltage
  power_a:   float32   # current draw
  system_v:  float32
  device_v:  float32
  error:     uint32[2]
  reserve:   uint32[2]
```

**`MotorCmd_`** — per-finger command ([_MotorCmd_.py](unitree_sdk2py/idl/unitree_hg/msg/dds_/_MotorCmd_.py)):
```python
MotorCmd_:
  mode:    uint8    # 1 = Enable, 0 = Disable
  q:       float32  # target position [rad]
  dq:      float32  # target velocity [rad/s]
  tau:     float32  # feedforward torque [Nm]
  kp:      float32  # position gain
  kd:      float32  # damping gain
  reserve: uint32
```

**`PressSensorState_`** — per-finger tactile reading ([_PressSensorState_.py](unitree_sdk2py/idl/unitree_hg/msg/dds_/_PressSensorState_.py)):
```python
PressSensorState_:
  pressure:    float32[12]  # 12 pressure cells per finger
  temperature: float32[12]
  lost:        uint32       # sensor loss flag
  reserve:     uint32
```

---

## 5. Step-by-Step Activation

### Step 1 — Release the body's motion controller

The body's high-level motion service conflicts with low-level control. Based on the pattern in [example/h1_2/low_level/h1_2_low_level_example.py](example/h1_2/low_level/h1_2_low_level_example.py):

```python
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

msc = MotionSwitcherClient()
msc.SetTimeout(5.0)
msc.Init()

status, result = msc.CheckMode()
while result['name']:          # keep releasing until no mode is active
    msc.ReleaseMode()
    status, result = msc.CheckMode()
    time.sleep(1)
```

### Step 2 — Initialize the DDS channel factory

```python
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# Pass your network interface as the first argument, e.g. "enp2s0"
if len(sys.argv) > 1:
    ChannelFactoryInitialize(0, sys.argv[1])
else:
    ChannelFactoryInitialize(0)   # uses default interface
```

### Step 3 — Set up hand publishers and subscribers

```python
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__HandState_,
)

# Publishers (commands TO the hands)
# NOTE: Verify these topic names against your firmware's documentation
left_hand_pub  = ChannelPublisher("rt/inspire/lefthand",  HandCmd_)
right_hand_pub = ChannelPublisher("rt/inspire/righthand", HandCmd_)
left_hand_pub.Init()
right_hand_pub.Init()

# Subscribers (state FROM the hands)
left_hand_state  = None
right_hand_state = None

def left_hand_state_handler(msg: HandState_):
    global left_hand_state
    left_hand_state = msg

def right_hand_state_handler(msg: HandState_):
    global right_hand_state
    right_hand_state = msg

left_hand_sub  = ChannelSubscriber("rt/inspire/lefthand/state",  HandState_)
right_hand_sub = ChannelSubscriber("rt/inspire/righthand/state", HandState_)
left_hand_sub.Init(left_hand_state_handler, 10)
right_hand_sub.Init(right_hand_state_handler, 10)
```

### Step 4 — Build and send a hand command

```python
import time

# Create a default-initialized HandCmd_ (7 motors, all zeroed)
left_cmd  = unitree_hg_msg_dds__HandCmd_()   # from default.py
right_cmd = unitree_hg_msg_dds__HandCmd_()

# Enable all 7 finger motors and command them to open position (q=0)
for i in range(7):
    left_cmd.motor_cmd[i].mode = 1    # 1 = Enable
    left_cmd.motor_cmd[i].q    = 0.0  # open position
    left_cmd.motor_cmd[i].dq   = 0.0
    left_cmd.motor_cmd[i].tau  = 0.0
    left_cmd.motor_cmd[i].kp   = 1.0  # tune for Inspire hardware
    left_cmd.motor_cmd[i].kd   = 0.1

    right_cmd.motor_cmd[i].mode = 1
    right_cmd.motor_cmd[i].q    = 0.0
    right_cmd.motor_cmd[i].dq   = 0.0
    right_cmd.motor_cmd[i].tau  = 0.0
    right_cmd.motor_cmd[i].kp   = 1.0
    right_cmd.motor_cmd[i].kd   = 0.1

# Publish at ~100 Hz
while True:
    left_hand_pub.Write(left_cmd)
    right_hand_pub.Write(right_cmd)
    time.sleep(0.01)
```

### Step 5 — Command a grip (close all fingers)

```python
# Typical close position — verify actual angle limits with Unitree docs
CLOSE_POSITION = 1.5  # radians, tune to your hardware

for i in range(7):
    left_cmd.motor_cmd[i].q  = CLOSE_POSITION
    right_cmd.motor_cmd[i].q = CLOSE_POSITION

left_hand_pub.Write(left_cmd)
right_hand_pub.Write(right_cmd)
```

### Step 6 — Read tactile feedback

```python
if left_hand_state is not None:
    for finger_idx in range(7):
        sensor = left_hand_state.press_sensor_state[finger_idx]
        print(f"Finger {finger_idx} pressures: {sensor.pressure}")
        print(f"  sensor lost: {sensor.lost}")
    print(f"Power: {left_hand_state.power_v:.2f}V  {left_hand_state.power_a:.2f}A")
```

---

## 6. Safe Defaults for Motor Gains

The SDK doesn't document Inspire-specific PID values. Start conservative and tune up:

| Parameter | Starting value | Notes |
|---|---|---|
| `kp` | 1.0 | Position gain — increase for stiffer hold |
| `kd` | 0.1 | Damping — increase to reduce overshoot |
| `tau` | 0.0 | Use pure PD first; add feedforward only if needed |
| Control rate | 100 Hz (0.01 s) | Body uses 500 Hz; hands are slower |

For comparison, the body arm/wrist motors use `kp=50.0, kd=1.0` at 500 Hz — see [example/h1_2/low_level/h1_2_low_level_example.py](example/h1_2/low_level/h1_2_low_level_example.py).

---

## 7. Full Minimal Example Script

```python
#!/usr/bin/env python3
"""Minimal Inspire hand activation for H1-2."""
import sys, time
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # Release body motion controller
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    _, result = msc.CheckMode()
    while result['name']:
        msc.ReleaseMode()
        time.sleep(1)
        _, result = msc.CheckMode()

    # Publishers — VERIFY topic names for your firmware
    lh_pub = ChannelPublisher("rt/inspire/lefthand",  HandCmd_)
    rh_pub = ChannelPublisher("rt/inspire/righthand", HandCmd_)
    lh_pub.Init()
    rh_pub.Init()

    lh_cmd = unitree_hg_msg_dds__HandCmd_()
    rh_cmd = unitree_hg_msg_dds__HandCmd_()

    print("Opening hands...")
    for i in range(7):
        for cmd in (lh_cmd, rh_cmd):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q    = 0.0
            cmd.motor_cmd[i].kp   = 1.0
            cmd.motor_cmd[i].kd   = 0.1

    for _ in range(200):   # 2 seconds at 100 Hz
        lh_pub.Write(lh_cmd)
        rh_pub.Write(rh_cmd)
        time.sleep(0.01)

    print("Closing hands...")
    for i in range(7):
        for cmd in (lh_cmd, rh_cmd):
            cmd.motor_cmd[i].q = 1.5   # tune this value

    for _ in range(200):
        lh_pub.Write(lh_cmd)
        rh_pub.Write(rh_cmd)
        time.sleep(0.01)

    print("Done.")

if __name__ == "__main__":
    main()
```

Run with:
```bash
python3 inspire_hand_test.py enp2s0
```

---

## 8. What This Repository Does NOT Provide

These items are absent from the codebase and must be sourced from Unitree's official documentation:

1. **Confirmed DDS topic names** for the Inspire hand command and state topics
2. **Finger angle limits** (min/max `q` in radians per motor)
3. **Finger index mapping** (which of the 7 `motor_cmd` indices corresponds to thumb, index, middle, ring, pinky, and the two remaining DOFs)
4. **Recommended PID gains** tuned for Inspire DEX3 hardware
5. **Hand firmware version compatibility** notes
6. **CRC requirement** — the body `LowCmd_` requires a CRC field; verify whether `HandCmd_` also requires one for your firmware

Official reference: `https://support.unitree.com/home/en/developer`

---

## 9. Key Source Files Summary

| File | Purpose |
|---|---|
| [example/h1_2/low_level/h1_2_low_level_example.py](example/h1_2/low_level/h1_2_low_level_example.py) | Only H1-2 example; body motor control pattern |
| [unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandCmd_.py](unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandCmd_.py) | HandCmd_ IDL definition |
| [unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandState_.py](unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandState_.py) | HandState_ IDL definition |
| [unitree_sdk2py/idl/unitree_hg/msg/dds_/_MotorCmd_.py](unitree_sdk2py/idl/unitree_hg/msg/dds_/_MotorCmd_.py) | Per-motor command fields |
| [unitree_sdk2py/idl/unitree_hg/msg/dds_/_PressSensorState_.py](unitree_sdk2py/idl/unitree_hg/msg/dds_/_PressSensorState_.py) | Tactile sensor data structure |
| [unitree_sdk2py/idl/default.py](unitree_sdk2py/idl/default.py) | Factory functions (zero-initialized structs) |
| [unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py](unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py) | Required to release body motion mode |
