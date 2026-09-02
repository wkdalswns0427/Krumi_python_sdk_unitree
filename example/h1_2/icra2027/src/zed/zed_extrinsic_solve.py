#!/usr/bin/env python3
"""zed_extrinsic_solve.py - solve the ZED-optical -> torso_link extrinsic (D1)
from a hand-eye capture, offline. No hardware, no ZED.

Eye-to-hand hand-eye: the ZED is fixed to the torso (the H1-2 has no head joint,
so it is rigid to torso_link while standing) and a checkerboard is fixed to a
hand. `zed_extrinsic_capture.py` moves the arm through poses and, per pose,
records the arm joint angles and the board pose in the ZED frame. This script
turns each pose's joints into the wrist-in-torso transform via H12ArmFK, then
calls cv2.calibrateHandEye in the eye-to-hand configuration to recover the
camera-in-torso transform. That transform is what maps ZED-frame human joints
into torso_link, replacing the identity placeholder in zed_realtime_node.py.

The board-in-hand offset never has to be measured: it is the constant that
cancels in the hand-eye loop.

    zed_extrinsic_solve.py capture.json --out results/extrinsic/zed_to_torso.yaml
    zed_extrinsic_solve.py --self-test        # verify the math, no inputs
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments"))
URDF = os.path.expanduser("~/mj_ws/assets/h1_2_description/h1_2.urdf")

HANDEYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def mat_to_quat(R):
    """3x3 rotation -> quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([x, y, z, w])


def solve_handeye(grippers2base, targets2cam, method=cv2.CALIB_HAND_EYE_PARK):
    """Eye-to-hand solve. grippers2base[i]: 4x4 wrist-in-torso (FK). targets2cam[i]:
    4x4 board-in-camera (solvePnP). Returns cam2base (ZED-optical -> torso_link)
    as a 4x4. Feeds base2gripper (the inverse) so OpenCV's eye-in-hand routine
    returns the eye-to-hand camera-in-base transform."""
    base2gripper = [np.linalg.inv(T) for T in grippers2base]
    R_b2g = [T[:3, :3] for T in base2gripper]
    t_b2g = [T[:3, 3] for T in base2gripper]
    R_t2c = [T[:3, :3] for T in targets2cam]
    t_t2c = [T[:3, 3] for T in targets2cam]
    R, t = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=method)
    return homogeneous(R, t)


def residual(cam2base, grippers2base, targets2cam):
    """Self-consistency of the solution: board-in-gripper implied by each pose
    (inv(g2b) @ cam2base @ t2c) must be constant. Returns its spread as
    translation RMS (mm) and rotation RMS (deg) about the mean."""
    Zs = [np.linalg.inv(g) @ cam2base @ t for g, t in zip(grippers2base, targets2cam)]
    tvecs = np.array([Z[:3, 3] for Z in Zs])
    t_rms_mm = float(np.sqrt(np.mean(np.sum((tvecs - tvecs.mean(0)) ** 2, axis=1)))) * 1000.0
    Rmean = _avg_rot([Z[:3, :3] for Z in Zs])
    angs = []
    for Z in Zs:
        dR = Rmean.T @ Z[:3, :3]
        angs.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return dict(t_rms_mm=t_rms_mm, r_rms_deg=float(np.sqrt(np.mean(np.square(angs)))))


def _avg_rot(Rs):
    """Chordal-mean rotation via SVD of the summed matrices."""
    M = sum(Rs)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def load_capture(path):
    d = json.load(open(path))
    side = d.get("side", "right")
    urdf = d.get("urdf", URDF)
    from h12_experiments.fk_h12 import H12ArmFK
    fk = H12ArmFK(urdf, side=side)
    g2b, t2c = [], []
    for p in d["poses"]:
        g2b.append(fk.fk(p["joints"]))
        rvec = np.asarray(p["rvec"], float).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        t2c.append(homogeneous(R, np.asarray(p["tvec"], float)))
    return side, urdf, g2b, t2c, d


def write_yaml(path, cam2base, side, n, method_name, res, source):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    R = cam2base[:3, :3]
    t = cam2base[:3, 3]
    q = mat_to_quat(R)
    stamp = datetime.date.today().isoformat()
    lines = [
        "# ZED left-optical -> torso_link extrinsic (D1). Hand-eye, eye-to-hand.",
        "# p_torso = R @ p_zed_optical + t. Replaces the identity placeholder in",
        "# zed_realtime_node.py. Regenerate with zed_extrinsic_solve.py.",
        f"solved_on: {stamp}",
        f"method: {method_name}",
        f"n_poses: {n}",
        f"arm_side: {side}",
        f"source_capture: {source}",
        f"residual_translation_mm: {res['t_rms_mm']:.2f}",
        f"residual_rotation_deg: {res['r_rms_deg']:.3f}",
        "frame_parent: torso_link",
        "frame_child: zed_left_optical",
        "translation_m: {{x: {:.6f}, y: {:.6f}, z: {:.6f}}}".format(*t),
        "rotation_quat_xyzw: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q),
        "rotation_matrix:",
    ]
    for r in R:
        lines.append("  - [{:.6f}, {:.6f}, {:.6f}]".format(*r))
    lines.append("homogeneous:")
    for row in cam2base:
        lines.append("  - [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*row))
    open(path, "w").write("\n".join(lines) + "\n")


def _rand_se3(rng, tscale=0.5):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = rng.uniform(0.2, 1.2)
    R, _ = cv2.Rodrigues(axis * ang)
    return homogeneous(R, rng.uniform(-tscale, tscale, 3))


def self_test():
    """Synthesize a known cam2base and board-in-gripper, generate the exact
    target2cam each pose implies, solve, and confirm recovery. Verifies the
    eye-to-hand wiring end to end without hardware."""
    rng = np.random.default_rng(0)
    Y_true = _rand_se3(rng, 0.8)          # cam2base = ZED -> torso, ground truth
    Z_true = _rand_se3(rng, 0.15)         # board-in-gripper, the nuisance constant
    g2b, t2c = [], []
    for _ in range(14):
        G = _rand_se3(rng, 0.5)            # a gripper (wrist-in-torso) pose
        # board-in-base via gripper = via cam:  G @ Z = Y @ t2c  ->  t2c = Yinv G Z
        T = np.linalg.inv(Y_true) @ G @ Z_true
        g2b.append(G); t2c.append(T)
    ok = True
    for name, m in HANDEYE_METHODS.items():
        Y = solve_handeye(g2b, t2c, method=m)
        dt = np.linalg.norm(Y[:3, 3] - Y_true[:3, 3]) * 1000.0
        dR = np.degrees(np.arccos(np.clip(
            (np.trace(Y[:3, :3].T @ Y_true[:3, :3]) - 1) / 2, -1, 1)))
        res = residual(Y, g2b, t2c)
        good = dt < 1e-3 and dR < 1e-3
        ok &= good
        print(f"  {name:11} recover err: {dt:7.4f} mm  {dR:7.4f} deg  | "
              f"self-residual {res['t_rms_mm']:.4f} mm {res['r_rms_deg']:.4f} deg  "
              f"{'OK' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", help="capture JSON from zed_extrinsic_capture.py")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results",
                    "extrinsic", "zed_to_torso.yaml"))
    ap.add_argument("--method", choices=list(HANDEYE_METHODS), default="PARK")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.capture:
        ap.error("capture JSON required (or use --self-test)")

    side, urdf, g2b, t2c, d = load_capture(args.capture)
    if len(g2b) < 3:
        ap.error(f"need >=3 poses, capture has {len(g2b)}")
    Y = solve_handeye(g2b, t2c, method=HANDEYE_METHODS[args.method])
    res = residual(Y, g2b, t2c)
    write_yaml(args.out, Y, side, len(g2b), args.method, res,
               os.path.basename(args.capture))
    print(f"[extrinsic] {len(g2b)} poses, method {args.method}")
    print(f"[extrinsic] translation (m): {Y[:3,3].round(4).tolist()}")
    print(f"[extrinsic] self-residual: {res['t_rms_mm']:.2f} mm, "
          f"{res['r_rms_deg']:.3f} deg  (lower is better; >5 mm or >1 deg "
          f"means poor pose diversity or board detections)")
    print(f"[extrinsic] wrote {args.out}")


if __name__ == "__main__":
    main()
