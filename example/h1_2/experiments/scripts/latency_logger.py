#!/usr/bin/env python3
"""latency_logger.py - Block 3 real-time RGBD retargeting node + latency log.

The live RGBD -> pose -> IK -> rt/arm_sdk node does NOT exist yet in any
workspace (vision today is YOLO + an offline MediaPipe path). This file is
that node's SKELETON, built around the four Block-3 timestamp points:

    t0 : image message arrival at the retargeting node
    t1 : pose (MediaPipe) output ready
    t2 : IK solution complete
    t3 : command published to rt/arm_sdk

The timestamping, CSV logging, and latency statistics are complete and
testable now. The camera source, pose estimator, and IK backend are behind
small interfaces; the real implementations are marked `# TODO(real-pipeline)`
and can be dropped in without touching the instrumentation.

    --source   realsense | synthetic     (default: synthetic)
    --pose     mediapipe | synthetic      (default: synthetic)
    --ik       stub      | synthetic      (default: synthetic)
    --synthetic is shorthand for all three synthetic + --no-publish, so the
    whole loop runs with no camera and no robot.

Usage:
    # self-test, no hardware:
    python3 latency_logger.py --synthetic --out /tmp/latency.csv --duration 5

    # live (once pose/IK/camera are wired):
    python3 latency_logger.py --source realsense --pose mediapipe --ik stub \
        --interface enp128s31f6 --out latency_live.csv --duration 300
"""

import argparse
import os
import sys
import time

import numpy as np


# ── Frame sources ────────────────────────────────────────────────────────────
class SyntheticSource:
    """Blank frames at a target fps. t0 stamped at 'arrival'."""

    def __init__(self, fps, width=640, height=480):
        self.period = 1.0 / fps
        self.width, self.height = width, height
        self._next = None
        self._i = 0

    def read(self):
        now = time.perf_counter()
        if self._next is None:
            self._next = now
        sleep = self._next - now
        if sleep > 0:
            time.sleep(sleep)
        self._next += self.period
        t0 = time.perf_counter()
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        fid = self._i
        self._i += 1
        return fid, img, t0

    def close(self):
        pass


class RealSenseSource:
    """D435i aligned color stream via pyrealsense2. t0 = frame arrival."""

    def __init__(self, fps=30, width=640, height=480):
        import pyrealsense2 as rs  # lazy: only needed for live runs
        self._rs = rs
        self._i = 0
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(cfg)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        t0 = time.perf_counter()  # arrival at this node
        frames = self.align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        fid = self._i
        self._i += 1
        return fid, color, t0

    def close(self):
        self.pipeline.stop()


# ── Pose estimators ──────────────────────────────────────────────────────────
class SyntheticPose:
    """Simulated MediaPipe compute cost; returns dummy landmarks."""

    def __init__(self, sim_ms=22.0):
        self.sim = sim_ms / 1000.0

    def estimate(self, image):
        time.sleep(self.sim)
        return np.zeros((33, 3), dtype=float)  # placeholder landmark array

    def close(self):
        pass


class MediaPipePose:
    """Real MediaPipe Pose wrapper (world landmarks)."""

    def __init__(self):
        import mediapipe as mp  # lazy: only needed for live runs
        self._pose = mp.solutions.pose.Pose(
            model_complexity=1, enable_segmentation=False)
        self._mp = mp

    def estimate(self, image_bgr):
        # TODO(real-pipeline): confirm color order / resize policy for the
        # retargeting node. MediaPipe expects RGB.
        rgb = image_bgr[:, :, ::-1]
        res = self._pose.process(rgb)
        if not res.pose_world_landmarks:
            return None
        return np.array([[lm.x, lm.y, lm.z]
                         for lm in res.pose_world_landmarks.landmark])

    def close(self):
        self._pose.close()


# ── IK backends ──────────────────────────────────────────────────────────────
class SyntheticIK:
    """Simulated IK cost; returns a 15-vector arm_sdk command."""

    def __init__(self, sim_ms=4.0):
        self.sim = sim_ms / 1000.0

    def solve(self, landmarks):
        time.sleep(self.sim)
        return np.zeros(15, dtype=float)

    def close(self):
        pass


class StubIK:
    """Placeholder for the real retargeting IK backend."""

    def solve(self, landmarks):
        # TODO(real-pipeline): map landmarks -> H1-2 15-joint arm_sdk vector
        # via the project IK backend, with the shoulder-yaw singularity
        # suppression counter exposed for the Block-2 success criterion.
        raise NotImplementedError(
            "StubIK.solve: wire the real IK backend before a live Block-3 run")

    def close(self):
        pass


# ── arm_sdk publisher ────────────────────────────────────────────────────────
class ArmPublisher:
    """Publishes a 15-joint command to rt/arm_sdk. no-op if disabled."""

    # arm_sdk vector order matches wave_on_person.py / replay_logger.py.
    JOINT_IDX = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 12]
    WEIGHT_IDX = 27
    KP = [80, 80, 60, 60, 30, 30, 30, 80, 80, 60, 60, 30, 30, 30, 150]
    KD = [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0,
          2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 2.0]

    def __init__(self, enabled, interface):
        self.enabled = enabled
        if not enabled:
            return
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelPublisher)
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        if interface:
            ChannelFactoryInitialize(0, interface)
        else:
            ChannelFactoryInitialize(0)
        self._msg = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()

    def publish(self, q15):
        if not self.enabled:
            return
        for i, j in enumerate(self.JOINT_IDX):
            mc = self._msg.motor_cmd[j]
            mc.mode, mc.q, mc.dq, mc.tau = 1, float(q15[i]), 0.0, 0.0
            mc.kp, mc.kd = self.KP[i], self.KD[i]
        self._msg.motor_cmd[self.WEIGHT_IDX].q = 1.0
        self._msg.crc = self._crc.Crc(self._msg)
        self._pub.Write(self._msg)


# ── Latency logging ──────────────────────────────────────────────────────────
class LatencyLog:
    def __init__(self, out_path):
        self.out_path = out_path
        self._fh = open(out_path, "w", newline="")
        import csv
        self._w = csv.writer(self._fh)
        self._w.writerow(["frame", "t0", "t1", "t2", "t3",
                          "pose_ms", "ik_ms", "pub_ms", "e2e_ms"])
        self.e2e = []
        self.stage = {"pose": [], "ik": [], "pub": []}
        self.t_first = None
        self.t_last = None
        self.n = 0

    def add(self, fid, t0, t1, t2, t3):
        pose_ms = (t1 - t0) * 1e3
        ik_ms = (t2 - t1) * 1e3
        pub_ms = (t3 - t2) * 1e3
        e2e_ms = (t3 - t0) * 1e3
        self._w.writerow([fid, f"{t0:.6f}", f"{t1:.6f}", f"{t2:.6f}",
                          f"{t3:.6f}", f"{pose_ms:.3f}", f"{ik_ms:.3f}",
                          f"{pub_ms:.3f}", f"{e2e_ms:.3f}"])
        self.stage["pose"].append(pose_ms)
        self.stage["ik"].append(ik_ms)
        self.stage["pub"].append(pub_ms)
        self.e2e.append(e2e_ms)
        if self.t_first is None:
            self.t_first = t0
        self.t_last = t3
        self.n += 1

    def close(self):
        self._fh.flush()
        self._fh.close()

    def report(self):
        print("=" * 62)
        print("Block 3 - real-time pipeline latency")
        print("=" * 62)
        print(f"  frames logged : {self.n}  -> {self.out_path}")
        if self.n == 0:
            print("=" * 62)
            return

        def med_p95(a):
            a = np.array(a)
            return np.median(a), np.percentile(a, 95)

        print(f"  {'stage':<18}{'median(ms)':>12}{'p95(ms)':>12}")
        for label, key in [("pose  (t1-t0)", "pose"), ("ik    (t2-t1)", "ik"),
                           ("pub   (t3-t2)", "pub")]:
            m, p = med_p95(self.stage[key])
            print(f"  {label:<18}{m:>12.2f}{p:>12.2f}")
        m, p = med_p95(self.e2e)
        print(f"  {'end-to-end(t3-t0)':<18}{m:>12.2f}{p:>12.2f}")
        print("-" * 62)
        span = (self.t_last - self.t_first) if self.n > 1 else 0.0
        rate = (self.n / span) if span > 0 else float("nan")
        print(f"  effective output rate : {rate:.2f} commands/s "
              f"(published over {span:.1f}s)")
        print("=" * 62)


# ── Factories ────────────────────────────────────────────────────────────────
def make_source(kind, fps):
    if kind == "realsense":
        return RealSenseSource(fps=fps)
    return SyntheticSource(fps=fps)


def make_pose(kind):
    return MediaPipePose() if kind == "mediapipe" else SyntheticPose()


def make_ik(kind):
    return StubIK() if kind == "stub" else SyntheticIK()


# ── Main loop ────────────────────────────────────────────────────────────────
def run(args):
    source = make_source(args.source, args.fps)
    pose = make_pose(args.pose)
    ik = make_ik(args.ik)
    pub = ArmPublisher(enabled=not args.no_publish, interface=args.interface)
    log = LatencyLog(args.out)

    print(f"[latency_logger] source={args.source} pose={args.pose} "
          f"ik={args.ik} publish={not args.no_publish}")
    print(f"[latency_logger] logging to {args.out} "
          f"({'Ctrl+C to stop' if not args.duration else f'{args.duration}s'})")
    t_start = time.perf_counter()
    try:
        while True:
            fid, image, t0 = source.read()
            landmarks = pose.estimate(image)
            t1 = time.perf_counter()
            if landmarks is None:      # no pose this frame; don't command
                continue
            q15 = ik.solve(landmarks)
            t2 = time.perf_counter()
            pub.publish(q15)
            t3 = time.perf_counter()
            log.add(fid, t0, t1, t2, t3)
            if args.duration and (time.perf_counter() - t_start) >= args.duration:
                print(f"[latency_logger] reached --duration {args.duration}s")
                break
    except KeyboardInterrupt:
        print("\n[latency_logger] stopped by user")
    finally:
        for obj in (source, pose, ik):
            try:
                obj.close()
            except Exception:
                pass
        log.close()
        log.report()


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output latency CSV path")
    p.add_argument("--source", choices=["realsense", "synthetic"],
                   default="synthetic")
    p.add_argument("--pose", choices=["mediapipe", "synthetic"],
                   default="synthetic")
    p.add_argument("--ik", choices=["stub", "synthetic"], default="synthetic")
    p.add_argument("--interface", help="robot interface (for publishing)")
    p.add_argument("--no-publish", action="store_true",
                   help="do not publish to rt/arm_sdk (measure up to t3=t2 stamp)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="synthetic source fps (default: 30)")
    p.add_argument("--duration", type=float, default=None,
                   help="auto-stop after N seconds (protocol Block 3: 300)")
    p.add_argument("--synthetic", action="store_true",
                   help="shorthand: synthetic source+pose+ik and --no-publish")
    return p.parse_args()


def main():
    args = parse_args()
    if args.synthetic:
        args.source, args.pose, args.ik, args.no_publish = \
            "synthetic", "synthetic", "synthetic", True
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
