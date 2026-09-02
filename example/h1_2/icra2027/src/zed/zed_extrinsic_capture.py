#!/usr/bin/env python3
"""zed_extrinsic_capture.py - collect a hand-eye dataset for the ZED->torso_link
extrinsic (D1). Hardware: drives one arm through poses and, at each, grabs the
ZED left image, detects a checkerboard on the hand, and records the arm joints
plus the board pose in the ZED frame.

Output feeds zed_extrinsic_solve.py, which runs the eye-to-hand solve offline.
The board-in-hand offset is never measured; it cancels in the hand-eye loop, so
the board just needs to be RIGIDLY taped to the hand and visible to the ZED.

    conda activate rical_unitree        # the env with pyzed + unitree_sdk2py + cv2
    python zed_extrinsic_capture.py enp128s31f6 \
        --side right --cols 9 --rows 6 --square 0.025 \
        --out ../../results/extrinsic/capture.json
    python zed_extrinsic_solve.py ../../results/extrinsic/capture.json

The ZED is a forward-facing scene camera (it watches the person in front, not the
robot's own hands), so the poses REACH THE HAND FORWARD into that forward view -
not up at the face - with the board mounted on the side of the hand that faces
BACK toward the head, so it looks into the camera. Poses spread vertically so some
land in the FOV whatever the camera tilt; wrist roll/pitch are varied (kept
moderate so the board stays camera-facing and detectable) for the rotation
diversity hand-eye needs. Poses where the board is not detected are skipped, so a
few out-of-frame poses are fine as long as >=3 (ideally 8+) are seen.

Safety, same as wave_on_person.py: takes over only the arm overlay; a slow, gentle
ramp; a human on the e-stop who is not the operator. Keep the robot on the GANTRY:
reaching the arm forward shifts the CoM forward, and free-standing that makes loco
step and moves the torso mid-sweep, which corrupts the calibration. The arm still
needs rt/arm_sdk control and rt/lowstate (FSM 204 or equivalent).

Use --list-poses to print the pose set without touching hardware or the ZED.
"""

import argparse
import json
import math
import os
import sys
import time


# ── H1-2 arm_sdk joint layout (same as wave_on_person.py) ───────────────────
class J:
    WaistYaw = 12
    LShoulderPitch, LShoulderRoll, LShoulderYaw, LElbow = 13, 14, 15, 16
    LWristRoll, LWristPitch, LWristYaw = 17, 18, 19
    RShoulderPitch, RShoulderRoll, RShoulderYaw, RElbow = 20, 21, 22, 23
    RWristRoll, RWristPitch, RWristYaw = 24, 25, 26
    Weight = 27


ARM_JOINTS = [J.LShoulderPitch, J.LShoulderRoll, J.LShoulderYaw, J.LElbow,
              J.LWristRoll, J.LWristPitch, J.LWristYaw,
              J.RShoulderPitch, J.RShoulderRoll, J.RShoulderYaw, J.RElbow,
              J.RWristRoll, J.RWristPitch, J.RWristYaw, J.WaistYaw]
ARM_KP = [80, 80, 60, 60, 30, 30, 30, 80, 80, 60, 60, 30, 30, 30, 150]
ARM_KD = [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0,
          2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 2.0]

# Motor indices for one side's 7 joints in H12ArmFK chain order:
# shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll/pitch/yaw.
SIDE_MOTORS = {
    "right": [J.RShoulderPitch, J.RShoulderRoll, J.RShoulderYaw, J.RElbow,
              J.RWristRoll, J.RWristPitch, J.RWristYaw],
    "left":  [J.LShoulderPitch, J.LShoulderRoll, J.LShoulderYaw, J.LElbow,
              J.LWristRoll, J.LWristPitch, J.LWristYaw],
}
# Index of the side's 7 joints inside the 15-element arm_sdk / override vector.
SIDE_SLOTS = {"right": list(range(7, 14)), "left": list(range(0, 7))}


def calib_poses(side):
    """Calibration poses that reach the hand forward into the scene camera's
    view. Each is a 7-vector in the H12ArmFK chain order, shoulder to wrist. The
    board faces back toward the head. Shoulder pitch is spread across three
    heights so poses land in the FOV whatever the camera tilt. Pitch 0 is
    arm-down and -1.57 is horizontal forward. Elbow stays at 0.8 for a moderate
    forward extension that is not singular. Wrist roll and pitch are swept for
    rotation diversity, kept moderate so the board stays camera facing. This set
    detected 11 of 11 on hardware. Widening the reach or adding lateral swing
    pushed poses out of the FOV, so keep the hand near this fixed forward reach."""
    r0 = float(-0.15 if side == "right" else 0.15)      # keep the arm centered in front
    elb = 0.8                                            # moderate forward extension, detects well
    poses = []
    for shp in (-1.20, -1.40, -1.55):                   # vertical spread, all forward
        for wr, wp in ((-0.5, 0.30), (0.5, -0.30), (0.0, 0.35)):   # wrist rotation diversity
            poses.append([shp, r0, 0.0, elb, wr, wp, 0.0])
    # two more with a small shoulder-yaw and wrist-yaw for extra rotation spread
    poses.append([-1.35, r0, 0.20, elb, 0.0, 0.0, 0.4])
    poses.append([-1.35, r0, -0.20, elb, 0.0, 0.0, -0.4])
    return poses


# ── minimal arm_sdk controller (gentle 50 Hz pump, subset of wave_on_person) ─
class ArmController:
    CTRL_DT = 0.02
    MAX_VEL = 0.25            # rad/s, gentle
    WEIGHT_RATE = 0.5

    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        self._mk = unitree_hg_msg_dds__LowCmd_
        self.msg = self._mk()
        self.weight = 0.0
        self.current = [0.0] * len(ARM_JOINTS)
        self.rest = None
        self.lowstate = None
        self._got = False
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._on_state, 10)
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()

    def _on_state(self, msg):
        self.lowstate = msg
        self._got = True

    def init(self):
        print("[arm_sdk] waiting for lowstate...")
        while not self._got:
            time.sleep(0.1)
        snap = [self.lowstate.motor_state[j].q for j in ARM_JOINTS]
        self.current = list(snap)
        self.rest = list(snap)
        print(f"[arm_sdk] REST captured: {[round(q,2) for q in snap]}")

    def _publish(self, target):
        step = self.MAX_VEL * self.CTRL_DT
        for i in range(len(ARM_JOINTS)):
            err = max(-step, min(step, target[i] - self.current[i]))
            self.current[i] += err
        for i, j in enumerate(ARM_JOINTS):
            mc = self.msg.motor_cmd[j]
            mc.mode, mc.q, mc.dq, mc.tau = 1, self.current[i], 0.0, 0.0
            mc.kp, mc.kd = ARM_KP[i], ARM_KD[i]
        self.msg.motor_cmd[J.Weight].q = self.weight
        self.pub.Write(self.msg)

    def ramp_weight(self, target, duration):
        n = max(1, int(duration / self.CTRL_DT))
        d = (target - self.weight) / n
        for _ in range(n):
            self.weight = max(0.0, min(1.0, self.weight + d))
            self._publish(self.current)
            time.sleep(self.CTRL_DT)
        self.weight = target

    def goto(self, target, settle=1.0, timeout=8.0):
        """Ramp to target, then hold settle seconds. Returns when arrived."""
        t0 = time.monotonic()
        while True:
            self._publish(target)
            time.sleep(self.CTRL_DT)
            if all(abs(target[i] - self.current[i]) < 0.02
                   for i in range(len(ARM_JOINTS))):
                break
            if time.monotonic() - t0 > timeout:
                print("  [warn] goto timed out before arrival")
                break
        t1 = time.monotonic()
        while time.monotonic() - t1 < settle:
            self._publish(target)
            time.sleep(self.CTRL_DT)

    def side_joints(self, side):
        return [self.lowstate.motor_state[j].q for j in SIDE_MOTORS[side]]


# ── ZED + checkerboard ──────────────────────────────────────────────────────
class ZedBoard:
    def __init__(self, cols, rows, square):
        import pyzed.sl as sl
        import numpy as np
        self.sl, self.np = sl, np
        self.cols, self.rows, self.square = cols, rows, square
        self.objp = np.zeros((rows * cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 60
        init.coordinate_units = sl.UNIT.METER
        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise SystemExit("[zed] failed to open camera")
        cam = self.zed.get_camera_information().camera_configuration
        c = cam.calibration_parameters.left_cam
        self.K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], float)
        self.dist = np.zeros(5)          # VIEW.LEFT is rectified
        self.fx, self.fy, self.cx, self.cy = c.fx, c.fy, c.cx, c.cy
        self._mat = sl.Mat()
        print(f"[zed] open HD720@60  fx={c.fx:.1f} cx={c.cx:.1f}")

    def detect(self):
        """Grab one frame; return (rvec, tvec, n_corners, reproj_px) or None."""
        import cv2
        np = self.np
        if self.zed.grab() != self.sl.ERROR_CODE.SUCCESS:
            return None
        self.zed.retrieve_image(self._mat, self.sl.VIEW.LEFT)
        gray = cv2.cvtColor(self._mat.get_data(), cv2.COLOR_BGRA2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (self.cols, self.rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        ok, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.dist)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.dist)
        reproj = float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
        return rvec.ravel().tolist(), tvec.ravel().tolist(), len(corners), reproj

    def close(self):
        self.zed.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("interface", nargs="?", help="robot NIC, e.g. enp128s31f6")
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square", type=float, default=0.025, help="square size, m")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "extrinsic", "capture.json"))
    ap.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"))
    ap.add_argument("--retries", type=int, default=8,
                    help="ZED grab attempts per pose before giving up")
    ap.add_argument("--list-poses", action="store_true",
                    help="print the pose set and exit (no hardware)")
    args = ap.parse_args()

    poses = calib_poses(args.side)
    if args.list_poses:
        print(f"{len(poses)} calibration poses for the {args.side} arm "
              f"[ShP, ShR, ShY, Elb, WrR, WrP, WrY]:")
        for i, p in enumerate(poses):
            print(f"  {i:2d}: " + " ".join(f"{v:+.2f}" for v in p))
        return

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    print("=" * 68)
    print(f"ZED->torso_link hand-eye capture  ({args.side} arm, "
          f"{args.cols}x{args.rows} board @ {args.square*1000:.0f} mm)")
    print(" * Board must be RIGIDLY taped to the hand and visible to the ZED.")
    print(" * Robot must already be balance-standing (FSM 204).")
    print("=" * 68)
    input("Press Enter to engage arm_sdk (Ctrl+C aborts)...")

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    arm = ArmController()
    arm.init()
    zed = ZedBoard(args.cols, args.rows, args.square)
    records = []
    try:
        print("[arm_sdk] weight 0->1")
        arm.ramp_weight(1.0, 1.5)
        slots = SIDE_SLOTS[args.side]
        for i, pose7 in enumerate(poses):
            target = list(arm.rest)
            for k, s in enumerate(slots):
                target[s] = pose7[k]
            print(f"[pose {i+1}/{len(poses)}] moving...")
            arm.goto(target, settle=1.2)
            det = None
            for _ in range(args.retries):
                det = zed.detect()
                if det is not None:
                    break
                time.sleep(0.05)
            if det is None:
                print("  board NOT detected - skipped")
                continue
            rvec, tvec, n, reproj = det
            records.append(dict(joints=arm.side_joints(args.side),
                                rvec=rvec, tvec=tvec, n_corners=n,
                                reproj_px=round(reproj, 3)))
            print(f"  detected {n} corners, reproj {reproj:.2f} px, "
                  f"depth {tvec[2]:.2f} m")
    finally:
        print("[arm_sdk] returning to REST, fading weight")
        try:
            arm.goto(arm.rest, settle=0.5)
            arm.ramp_weight(0.0, 1.5)
        except Exception as e:
            print(f"  shutdown error: {e}")
        zed.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(side=args.side, urdf=args.urdf,
                   board=dict(cols=args.cols, rows=args.rows, square_m=args.square),
                   camera=dict(fx=zed.fx, fy=zed.fy, cx=zed.cx, cy=zed.cy),
                   poses=records), open(args.out, "w"), indent=2)
    print(f"\n[capture] {len(records)}/{len(poses)} poses detected -> {args.out}")
    if len(records) < 3:
        print("[capture] WARNING: <3 detections; the solve needs at least 3 "
              "(ideally 8+ with varied wrist orientation).")
    else:
        print(f"[capture] next: zed_extrinsic_solve.py {args.out}")


if __name__ == "__main__":
    main()
