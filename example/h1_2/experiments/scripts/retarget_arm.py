#!/usr/bin/env python3
"""retarget_arm.py - human joints CSV -> H1-2 arm joint trajectory (Block 2).

Takes the per-frame 3D joints CSV from bag_to_joints (MONO recommended:
complete joint coverage, person frame) and produces a 50 Hz H1-2 arm
trajectory CSV that replay_arm.py can play through rt/arm_sdk.

Method (scope-frozen: no new IK claim, geometric direction matching):
1. Per frame, build a torso frame from the subject's shoulders and hips
   (z up along the trunk, y toward the subject's left, x forward).
2. Express the upper-arm and forearm DIRECTIONS in that frame. Directions,
   not positions, so subject size cancels; the robot's own limb lengths
   come from the URDF.
3. Per arm, solve shoulder pitch/roll/yaw + elbow with damped Gauss-Newton
   on the URDF forward kinematics (fk_h12 chain), warm-started from the
   previous frame. Residual = achieved vs target unit vectors of both limb
   segments.
4. Shoulder-yaw suppression near the straight-arm singularity: when the
   elbow is nearly straight the swivel is unobservable, so yaw is frozen at
   its previous value and the event is COUNTED (the Block 2 success
   criterion logs this counter).
5. URDF joint limits clamped and counted. Wrists and waist stay at zero
   (protocol scope).
6. Input conditioning (added 2026-07-13 after the first robot replay showed
   the quasi-static left arm shaking): a dt-aware teleport gate rejects
   frames whose limb direction rotates faster than --max-dir-rate relative
   to the last accepted frame (MediaPipe keypoint teleports, re-lock after
   --relock consecutive rejections, same pattern as the frozen depth
   policy), then a One-Euro filter on the direction vectors kills the
   static-pose jitter while letting the fast hammer whip through
   (velocity-adaptive cutoff). Both are counted and printed.
7. Undetected frames are gap-interpolated, the series is resampled to the
   output rate and low-pass filtered with a zero-phase Butterworth
   (--lp-cutoff, offline so no lag; spectral evidence from M1_R_1: real
   motion incl. the hammer whip is < 3 Hz in direction space, the shake is
   the 3-15 Hz band). --smooth-window moving average retained as a final
   polish.

Output CSV: t_s + the 15 arm/waist joints in h12_experiments joints order
(left arm 7, right arm 7, waist_yaw). Stats printed: IK residuals, yaw
suppression count, limit clamps, peak joint velocity (pre-flight for the
replayer's rate limit).

Usage:
    /usr/bin/python3 retarget_arm.py iphone_data/M1_R_1/b2g/iph_mono.csv \
        --urdf ~/mj_ws/assets/h1_2_description/h1_2.urdf
    # -> iphone_data/M1_R_1/b2g/traj_iph_mono.csv (b2g default)
"""

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

# fk_h12 lives in the h12_experiments ROS2 package; import it directly.
FK_PKG = os.path.expanduser("~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments")
sys.path.insert(0, FK_PKG)
from h12_experiments.fk_h12 import (  # noqa: E402
    UrdfChain, rpy_to_matrix, axis_angle_matrix, homogeneous, BASE_LINK,
    DEFAULT_TIP)

SIDES = ("left", "right")
ARM_SEGS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw")
OUT_JOINTS = [f"{s}_{seg}" for s in SIDES for seg in ARM_SEGS] + ["waist_yaw"]

STRAIGHT_ARM_DEG = 12.0   # elbow flexion below this = swivel unobservable


# ── URDF helpers ─────────────────────────────────────────────────────────────
def load_limits(urdf_path):
    """{joint_name: (lower, upper)} for revolute joints."""
    out = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        lim = j.find("limit")
        if j.get("type") in ("revolute", "continuous") and lim is not None:
            out[j.get("name")] = (float(lim.get("lower", -np.pi)),
                                  float(lim.get("upper", np.pi)))
    return out


class ArmChainFK:
    """FK over the 7-joint arm chain returning intermediate joint positions."""

    def __init__(self, urdf_path, side):
        self.urdf = UrdfChain(urdf_path)
        self.side = side
        self.chain = self.urdf.chain(BASE_LINK, DEFAULT_TIP[side])
        self.actuated = [j for j in self.chain
                         if self.urdf.joints[j]["type"] in ("revolute",
                                                            "continuous")]
        self.anchor_joint = f"{side}_shoulder_roll_joint"
        self.elbow_joint = f"{side}_elbow_joint"
        self.tip_joint = f"{side}_wrist_yaw_joint"

    def positions(self, q4):
        """Positions of anchor/elbow/tip for [pitch, roll, yaw, elbow]."""
        q = list(q4) + [0.0, 0.0, 0.0]
        t = np.eye(4)
        pos, qi = {}, 0
        for jn in self.chain:
            jd = self.urdf.joints[jn]
            t = t @ homogeneous(rpy_to_matrix(*jd["rpy"]), jd["xyz"])
            pos[jn] = t[:3, 3].copy()
            if jd["type"] in ("revolute", "continuous"):
                t = t @ homogeneous(axis_angle_matrix(jd["axis"], q[qi]),
                                    [0, 0, 0])
                qi += 1
        return (pos[self.anchor_joint], pos[self.elbow_joint],
                pos[self.tip_joint])


def _unit(v):
    n = np.linalg.norm(v)
    return None if n < 1e-9 else v / n


class OneEuro:
    """One-Euro filter (Casiez 2012) over an N-dim signal.

    Cutoff = min_cutoff + beta * |dx/dt|: heavy smoothing while the signal
    is slow (static-pose jitter), little smoothing when it moves fast (the
    hammer whip passes through). Pure numpy, dt-aware for gapped frames.
    """

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-6)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


def solve_arm(fk, u_hat, f_hat, q_prev, limits4, suppress_yaw, yaw_reg=0.0,
              yaw_neutral=0.0):
    """Damped Gauss-Newton: match both limb segment directions.

    Returns (q4, residual_deg, clamped_flags). suppress_yaw freezes q[2].

    Shoulder yaw (index 2) is the DOF whose direction-residual gradient goes
    shallow on forward-reaching poses (M2/M3): the forearm direction is
    matched by TWO very different humeral rotations (a natural one near
    neutral, and a mirrored one ~150 deg away near the joint limit, which
    looks like an inverted arm). Two regularizers on yaw ONLY, both leaving
    the observable DOFs (pitch/roll/elbow) untouched:
      yaw_neutral > 0 : pull toward 0. Breaks the branch ambiguity toward the
        anatomically natural pose (the mirrored branch is ~150 deg off, so a
        small weight rejects it while barely touching real +/-50 deg motion).
      yaw_reg > 0     : continuity, pull toward the previous frame's yaw.
    Use yaw_neutral for correct branch SELECTION and yaw_reg for smoothness.
    """
    q = np.array(q_prev, dtype=float)
    yaw_ref = float(q_prev[2])
    lo = np.array([l for l, _ in limits4])
    hi = np.array([h for _, h in limits4])
    clamped = np.zeros(4, dtype=bool)

    def residual(qv):
        a, e, w = fk.positions(qv)
        r1 = _unit(e - a)
        r2 = _unit(w - e)
        if r1 is None or r2 is None:
            return None
        return np.concatenate([r1 - u_hat, r2 - f_hat])

    for _ in range(10):
        r = residual(q)
        if r is None:
            break
        J = np.zeros((6, 4))
        eps = 1e-4
        for k in range(4):
            if suppress_yaw and k == 2:
                continue                     # frozen: zero column
            qp = q.copy()
            qp[k] += eps
            rp = residual(qp)
            if rp is None:
                continue
            J[:, k] = (rp - r) / eps
        JtJ = J.T @ J + 1e-2 * np.eye(4)
        rhs = -J.T @ r
        if not suppress_yaw:
            if yaw_reg > 0:                          # continuity (prev frame)
                JtJ[2, 2] += yaw_reg
                rhs[2] -= yaw_reg * (q[2] - yaw_ref)
            if yaw_neutral > 0:                      # branch select (toward 0)
                JtJ[2, 2] += yaw_neutral
                rhs[2] -= yaw_neutral * q[2]
        dq = np.linalg.solve(JtJ, rhs)
        if suppress_yaw:
            dq[2] = 0.0
        q = q + dq
        newq = np.clip(q, lo, hi)
        clamped |= (np.abs(newq - q) > 1e-9)
        q = newq
        if np.linalg.norm(dq) < 1e-4:
            break

    r = residual(q)
    res_deg = float("nan")
    if r is not None:
        a1 = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(r[:3]) / 2, 0, 1)))
        a2 = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(r[3:]) / 2, 0, 1)))
        res_deg = max(a1, a2)
    return q, res_deg, clamped


# ── Human side ───────────────────────────────────────────────────────────────
def filter_depth_axis(frames, fps, cutoff_hz):
    """Median-5 + zero-phase low-pass on the z (depth) of every joint.

    MediaPipe mono depth is the weak axis: on M1_R_1 the raised-but-static
    left wrist measured 2.0 / 2.8 / 11.6 cm std in x / y / z during the
    hammering phase, the z oscillation phase-locked to the right-arm
    strikes (model crosstalk). Real image-plane motion is untouched; only
    z is conditioned. Runs before the torso frame is built, so shoulder /
    hip depth noise stops rocking the torso frame too.
    """
    from scipy.signal import butter, filtfilt, medfilt
    joints = {j for f in frames.values() for j in f}
    for j in joints:
        fis = [fi for fi in sorted(frames) if j in frames[fi]]
        if len(fis) < 16:
            continue
        t = np.array([fi / fps for fi in fis])
        z = np.array([frames[fi][j][2] for fi in fis])
        # uniform grid at the median input rate for the IIR filter
        dt = float(np.median(np.diff(t)))
        tu = np.arange(t[0], t[-1] + dt / 2, dt)
        zu = np.interp(tu, t, z)
        zu = medfilt(zu, 5)                    # isolated teleports
        b, a = butter(2, min(cutoff_hz * 2 * dt, 0.99))
        zu = filtfilt(b, a, zu)
        zf = np.interp(t, tu, zu)
        for fi, v in zip(fis, zf):
            frames[fi][j][2] = v


def load_joints_csv(path, conf_min):
    """{frame: {joint: np.array(xyz)}} keeping joints with conf >= conf_min."""
    frames = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c = float(row.get("conf", 1.0) or 1.0)
            if c < conf_min:
                continue
            fi = int(float(row["frame"]))
            frames.setdefault(fi, {})[row["joint"].strip()] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])])
    return frames


def torso_frame(j):
    """3x3 basis [x fwd, y left, z up] from shoulders + hips, or None."""
    need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(n not in j for n in need):
        return None
    up = _unit((j["left_shoulder"] + j["right_shoulder"]) / 2
               - (j["left_hip"] + j["right_hip"]) / 2)
    left = j["left_shoulder"] - j["right_shoulder"]
    if up is None:
        return None
    left = _unit(left - np.dot(left, up) * up)
    if left is None:
        return None
    fwd = np.cross(left, up)
    return np.stack([fwd, left, up], axis=1)   # columns: x, y, z


def arm_directions(j, R, side):
    """(u_hat, f_hat, flex_deg) for one arm in the torso frame, or None."""
    need = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
    if any(n not in j for n in need):
        return None
    u = _unit(R.T @ (j[f"{side}_elbow"] - j[f"{side}_shoulder"]))
    fv = _unit(R.T @ (j[f"{side}_wrist"] - j[f"{side}_elbow"]))
    if u is None or fv is None:
        return None
    flex = np.degrees(np.arccos(np.clip(np.dot(u, fv), -1, 1)))
    return u, fv, flex


# ── Pipeline ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("joints_csv", help="bag_to_joints output (mono recommended)")
    p.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"))
    p.add_argument("--out", help="default: <dir>/traj_<stem>.csv next to input")
    p.add_argument("--fps", type=float, default=60.0,
                   help="capture fps: frame index -> seconds (default 60)")
    p.add_argument("--rate", type=float, default=50.0,
                   help="output trajectory rate Hz (default 50, rt/arm_sdk)")
    p.add_argument("--conf-min", type=float, default=0.3,
                   help="drop joints below this confidence (default 0.3)")
    p.add_argument("--smooth-window", type=int, default=7,
                   help="moving-average window on the output series (odd, "
                        "default 7 = 0.14 s at 50 Hz; 1 disables)")
    p.add_argument("--straight-arm-deg", type=float, default=STRAIGHT_ARM_DEG)
    p.add_argument("--max-dir-rate", type=float, default=600.0,
                   help="teleport gate: reject a frame whose limb direction "
                        "rotates faster than this vs the last accepted frame "
                        "(deg/s; real hammer whip peaks ~460; 0 disables)")
    p.add_argument("--relock", type=int, default=5,
                   help="accept unconditionally after this many consecutive "
                        "gate rejections (the arm really moved)")
    p.add_argument("--euro-min-cutoff", type=float, default=1.0,
                   help="One-Euro static cutoff Hz on limb directions "
                        "(default 1.0; 0 disables the filter)")
    p.add_argument("--euro-beta", type=float, default=0.5,
                   help="One-Euro speed coefficient (cutoff rises by "
                        "beta * |d dir/dt|, so fast motion passes)")
    p.add_argument("--lp-cutoff", type=float, default=3.0,
                   help="zero-phase Butterworth low-pass on the output "
                        "joint series, Hz (default 3.0; 0 disables)")
    p.add_argument("--z-cutoff", type=float, default=0.8,
                   help="zero-phase low-pass on the keypoint DEPTH axis "
                        "before retargeting, Hz (mono z is the noisy axis; "
                        "default 0.8; 0 disables)")
    p.add_argument("--yaw-neutral", type=float, default=0.0,
                   help="shoulder-yaw branch-selection weight: pull yaw toward "
                        "0 so the IK picks the natural humeral rotation, not "
                        "the mirrored branch near the joint limit (which looks "
                        "like an inverted arm). FROZEN GLOBAL 0.02 (paired "
                        "with --yaw-reg 0.05).")
    p.add_argument("--yaw-reg", type=float, default=0.0,
                   help="shoulder-yaw continuity penalty weight in the IK "
                        "(quadratic pull toward the previous frame's yaw; "
                        "0 = off; kills single-frame flips). FROZEN GLOBAL "
                        "0.05, paired with --yaw-neutral 0.02 (branch select)")
    args = p.parse_args()

    if not os.path.isfile(args.joints_csv):
        sys.exit(f"not found: {args.joints_csv}")
    if not os.path.isfile(args.urdf):
        sys.exit(f"URDF not found: {args.urdf}")
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.joints_csv)),
        "traj_" + os.path.splitext(os.path.basename(args.joints_csv))[0] + ".csv")

    frames = load_joints_csv(args.joints_csv, args.conf_min)
    if not frames:
        sys.exit("no usable frames in the joints CSV")
    if args.z_cutoff > 0:
        filter_depth_axis(frames, args.fps, args.z_cutoff)
    idxs = sorted(frames)
    limits = load_limits(args.urdf)
    fks = {s: ArmChainFK(args.urdf, s) for s in SIDES}
    lim4 = {s: [limits[f"{s}_{seg}_joint"] for seg in ARM_SEGS[:4]]
            for s in SIDES}

    # per-frame IK
    t_in, q_in = [], []                       # q_in rows: 8 values (L4 + R4)
    q_prev = {s: np.array([0.0, 0.15 if s == "left" else -0.15, 0.0, 0.3])
              for s in SIDES}
    stats = {s: dict(solved=0, skipped=0, gated=0, suppress=0, res=[],
                     ee=[], clamp=np.zeros(4))
             for s in SIDES}
    last_dir = {s: None for s in SIDES}       # (t, u_hat, f_hat) accepted
    rejects = {s: 0 for s in SIDES}
    euro = {s: (OneEuro(args.euro_min_cutoff, args.euro_beta)
                if args.euro_min_cutoff > 0 else None) for s in SIDES}

    def dir_rate_deg(a, b, dt):
        return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))) / dt

    for fi in idxs:
        j = frames[fi]
        R = torso_frame(j)
        if R is None:
            continue
        t_now = fi / args.fps
        row = np.full(8, np.nan)
        for si, s in enumerate(SIDES):
            d = arm_directions(j, R, s)
            st = stats[s]
            if d is None:
                st["skipped"] += 1
                continue
            u_hat, f_hat, _ = d

            # teleport gate vs the last ACCEPTED directions (dt-aware)
            if args.max_dir_rate > 0 and last_dir[s] is not None:
                t_ref, u_ref, f_ref = last_dir[s]
                dt = max(t_now - t_ref, 1e-6)
                rate = max(dir_rate_deg(u_hat, u_ref, dt),
                           dir_rate_deg(f_hat, f_ref, dt))
                if rate > args.max_dir_rate and rejects[s] < args.relock:
                    rejects[s] += 1
                    st["gated"] += 1
                    continue
            rejects[s] = 0
            last_dir[s] = (t_now, u_hat, f_hat)

            # One-Euro on the direction components, renormalized
            if euro[s] is not None:
                filt = euro[s](np.concatenate([u_hat, f_hat]), t_now)
                u_f, f_f = _unit(filt[:3]), _unit(filt[3:])
                if u_f is not None and f_f is not None:
                    u_hat, f_hat = u_f, f_f
            flex = np.degrees(np.arccos(np.clip(np.dot(u_hat, f_hat), -1, 1)))

            suppress = flex < args.straight_arm_deg
            st["suppress"] += 1 if suppress else 0
            q, res, clamped = solve_arm(fks[s], u_hat, f_hat, q_prev[s],
                                        lim4[s], suppress, args.yaw_reg,
                                        args.yaw_neutral)
            q_prev[s] = q
            st["solved"] += 1
            st["res"].append(res)
            st["clamp"] += clamped
            # EE tracking error: achieved FK tip vs the tip implied by the
            # human directions and the robot's own limb lengths (metres).
            anchor, elbow, tip = fks[s].positions(q)
            l_up = np.linalg.norm(elbow - anchor)
            l_fo = np.linalg.norm(tip - elbow)
            target_tip = anchor + l_up * u_hat + l_fo * f_hat
            st["ee"].append(float(np.linalg.norm(tip - target_tip)))
            row[si * 4:si * 4 + 4] = q
        t_in.append(t_now)
        q_in.append(row)

    if not t_in:
        sys.exit("no frames had a usable torso frame")
    t_in = np.array(t_in)
    q_in = np.array(q_in)

    # gap interpolation per channel, then uniform resample to --rate
    t_out = np.arange(t_in[0], t_in[-1], 1.0 / args.rate)
    q_out = np.zeros((len(t_out), 8))
    for k in range(8):
        col = q_in[:, k]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            sys.exit(f"channel {k}: not enough solved frames")
        q_out[:, k] = np.interp(t_out, t_in[ok], col[ok])

    if args.lp_cutoff > 0:
        from scipy.signal import butter, filtfilt
        b, a = butter(2, args.lp_cutoff / (args.rate / 2))
        for k in range(8):
            q_out[:, k] = filtfilt(b, a, q_out[:, k])

    if args.smooth_window > 1:
        w = args.smooth_window | 1            # force odd
        kern = np.ones(w) / w
        pad = w // 2
        for k in range(8):
            ext = np.pad(q_out[:, k], pad, mode="edge")
            q_out[:, k] = np.convolve(ext, kern, mode="valid")

    # write: 15 columns, wrists + waist zero
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + OUT_JOINTS)
        for i, t in enumerate(t_out):
            full = np.zeros(15)
            full[0:4] = q_out[i, 0:4]         # left arm 4 actuated
            full[7:11] = q_out[i, 4:8]        # right arm 4 actuated
            w.writerow([f"{t - t_out[0]:.4f}"] + [f"{v:.6f}" for v in full])

    vel = np.abs(np.diff(q_out, axis=0)) * args.rate
    dur = t_out[-1] - t_out[0]
    print(f"[retarget] {args.joints_csv} -> {out}")
    print(f"[retarget] frames in={len(idxs)}  solved rows={len(t_in)}  "
          f"output={len(t_out)} @ {args.rate:g} Hz  duration={dur:.1f} s")
    for s in SIDES:
        st = stats[s]
        res = np.array(st["res"]) if st["res"] else np.array([np.nan])
        ee = np.array(st["ee"]) * 100 if st["ee"] else np.array([np.nan])
        print(f"[retarget] {s:>5}: solved={st['solved']} skipped={st['skipped']}"
              f"  gated={st['gated']}  yaw-suppressed={st['suppress']}"
              f"  IK residual mean={np.nanmean(res):.2f} deg"
              f" p95={np.nanpercentile(res, 95):.2f} deg"
              f"  EE-track mean={np.nanmean(ee):.2f} cm p95={np.nanpercentile(ee, 95):.2f} cm"
              f"  limit-clamps p/r/y/e={st['clamp'].astype(int).tolist()}")
    names = [f"{s}_{seg}" for s in SIDES for seg in ARM_SEGS[:4]]
    peak = vel.max(axis=0)
    top = np.argsort(peak)[::-1][:3]
    print(f"[retarget] peak joint velocity: "
          + "  ".join(f"{names[k]}={peak[k]:.2f} rad/s" for k in top)
          + "   (replayer rate limit must exceed these, or scale speed)")


if __name__ == "__main__":
    main()
