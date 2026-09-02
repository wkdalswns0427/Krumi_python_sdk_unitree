#!/usr/bin/env python3
"""zed_realtime_node.py - live ZED skeleton -> H1-2 arm_sdk @ ~50 Hz.

Track C real-time teleop node. Feeds live ZED body-tracking keypoints
through the IROS-frozen retarget (direction-matching Gauss-Newton IK on the
H1-2 URDF, --yaw-neutral 0.02 --yaw-reg 0.05 global) and streams to
rt/arm_sdk with the same weight-fade / gravity-feedforward pattern as
replay_arm.py. Causal only: One-Euro on joint targets (no zero-phase
filters), per-joint velocity clamp, dead-man ENTER at arm and start.

Latency (t0..t3) is logged per frame; --print-stats prints median / p95
at exit.

    t0 : ZED grab returned
    t1 : skeleton -> canonical joints -> arm directions ready
    t2 : per-side IK solved, full pose assembled and filtered
    t3 : command published on rt/arm_sdk

Reuses:
  - ../../../experiments/scripts/retarget_arm.py  (IK, torso frame, OneEuro)
  - ../../../experiments/scripts/replay_arm.py    (ArmSdk, GravityFF)
  - ./zed_to_joints.py                            (canonical joint mapping)

Usage:
    # pre-flight, no DDS, no robot; can run without a person if you also
    # pass --allow-no-body (holds the home pose):
    /usr/bin/python3 zed_realtime_node.py --dry-run

    # on the robot (conda env with unitree_sdk2py, robot STANDING in balance):
    python3 zed_realtime_node.py --interface enp128s31f6

    # self-test (no hardware, no ZED):
    /usr/bin/python3 zed_realtime_node.py --self-test

    # skeleton drawn on the live camera view, no robot involved:
    /usr/bin/python3 zed_realtime_node.py --dry-run --ros --ros-image --static

    # add depth and the coloured point cloud:
    /usr/bin/python3 zed_realtime_node.py --dry-run --ros --ros-image \
        --ros-depth --ros-cloud --static

    # same, replaying a recording instead of the live camera (dev check):
    /usr/bin/python3 zed_realtime_node.py --dry-run --ros --svo capture.svo2

ROS / RViz (--ros): publishes visualization_msgs/MarkerArray on
/zed/human_skeleton (bones + keypoints, live) and sensor_msgs/JointState on
/joint_states (the retargeted arm targets, every other URDF joint at 0). Run
robot_state_publisher on the same URDF to drive an RViz RobotModel from it -
optional, and only worth it when you want to see what the retarget produced.

--ros-image additionally publishes the left camera frame on /zed/left/image
with /zed/left/camera_info, which is what you want to see the skeleton ON the
video. Two overlays come with it, and they are independent:
  - the 2D skeleton is drawn into the image itself from ZED's own pixel
    keypoints (--no-ros-overlay turns this off), so it cannot drift from what
    the detector saw, and shows up in rqt_image_view too;
  - RViz's Camera display projects the 3D MarkerArray onto the same image
    using CameraInfo and the published camera->optical TF.

--ros-depth publishes the depth map on /zed/depth/image (32FC1, metres, NaN
where unmeasured) with /zed/depth/camera_info. --ros-cloud publishes the
coloured cloud on /zed/points. The cloud is in the ZED body frame, the same
one the skeleton markers use, so in RViz the skeleton sits inside the cloud
with no extra transform.
The ZED frame is RIGHT_HANDED_Z_UP_X_FWD, which already matches REP-103, so
the keypoints publish as-is. Needs an rclpy-capable interpreter: pyzed is
installed for both /usr/bin/python3 (3.10, the one with Humble rclpy) and the
rical_unitree conda env (3.11, no rclpy), so use /usr/bin/python3 for --ros.

SAFETY: workspace clear, arms only, second person on the e-stop. The node
snapshots the current arm pose, fades the arm_sdk weight in, ramps to a
neutral home pose, then arms streaming only after the operator presses
ENTER twice. Ctrl+C at any point freezes the last commanded pose and fades
the weight out cleanly.
"""

import argparse
import array
import csv
import os
import signal
import sys
import time

import numpy as np

# ── Locate the IROS-frozen retarget + replay modules ─────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "experiments", "scripts"))
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _HERE)

from zed_to_joints import (  # noqa: E402
    canonical_index_map, pick_primary_body)
from retarget_arm import (  # noqa: E402
    ArmChainFK, solve_arm, torso_frame, arm_directions, OneEuro,
    load_limits, STRAIGHT_ARM_DEG)
from replay_arm import (  # noqa: E402
    ArmSdk, GravityFF, ARM_SEGS, FROZEN_GRAVITY_GAIN)

SIDES = ("left", "right")
OUT_JOINTS = [f"{s}_{seg}" for s in SIDES for seg in ARM_SEGS] + ["waist_yaw"]

# Neutral "home" pose = all zeros. Arms hang at the sides (H1-2 zero pose).
HOME_POSE = [0.0] * len(OUT_JOINTS)


_stop = False


def _sigint(signum, frame):
    global _stop
    _stop = True
    print("\n[realtime] Ctrl+C -> freezing pose, fading weight out")


def body_to_joints_dict(body, index_map):
    """{canonical_name: np.array([x,y,z])} for one body. NaN keypoints dropped."""
    out = {}
    if body is None:
        return out
    kp = body.keypoint
    for zed_idx, canonical in index_map.items():
        if zed_idx >= len(kp):
            continue
        x, y, z = float(kp[zed_idx][0]), float(kp[zed_idx][1]), float(kp[zed_idx][2])
        if x != x or y != y or z != z:
            continue
        out[canonical] = np.array([x, y, z])
    return out


def clamp_velocity(prev_pose, target_pose, max_vel, dt):
    """Per-joint rate limit: |q_new - q_prev| <= max_vel * dt."""
    if prev_pose is None:
        return list(target_pose)
    out = []
    max_step = max_vel * dt
    for p, t in zip(prev_pose, target_pose):
        d = max(-max_step, min(max_step, t - p))
        out.append(p + d)
    return out


def assemble_pose(q_per_side):
    """15-joint pose ordered per OUT_JOINTS. Wrists + waist held at zero."""
    pose = []
    for side in SIDES:
        q4 = q_per_side[side]
        pose.extend([float(q4[0]), float(q4[1]), float(q4[2]), float(q4[3]),
                     0.0, 0.0, 0.0])
    pose.append(0.0)
    return pose


# Bones drawn in RViz. The arm/torso bones match visualize_joints.py's SKELETON;
# the live view additionally draws the legs (hip-knee-ankle), which BODY_38
# tracks but the offline upper-body renderer omits. (joint_a, joint_b, rgb).
# Left side blue, right side red, consistent across arms and legs.
SKELETON_BONES = (
    ("left_shoulder",  "left_elbow",     (0.13, 0.59, 0.95)),  # left arm
    ("left_elbow",     "left_wrist",     (0.13, 0.59, 0.95)),
    ("right_shoulder", "right_elbow",    (0.96, 0.26, 0.21)),  # right arm
    ("right_elbow",    "right_wrist",    (0.96, 0.26, 0.21)),
    ("left_shoulder",  "right_shoulder", (0.30, 0.69, 0.31)),  # shoulders
    ("left_shoulder",  "left_hip",       (0.61, 0.15, 0.69)),  # torso
    ("right_shoulder", "right_hip",      (0.61, 0.15, 0.69)),
    ("left_hip",       "right_hip",      (0.61, 0.15, 0.69)),
    ("left_hip",       "left_knee",      (0.13, 0.59, 0.95)),  # left leg
    ("left_knee",      "left_ankle",     (0.13, 0.59, 0.95)),
    ("right_hip",      "right_knee",     (0.96, 0.26, 0.21)),  # right leg
    ("right_knee",     "right_ankle",    (0.96, 0.26, 0.21)),
)


def skeleton_segments(j):
    """[(p_a, p_b, rgb)] for the bones whose endpoints are both tracked.
    Pure, so --self-test covers it without ROS."""
    return [(j[a], j[b], rgb) for a, b, rgb in SKELETON_BONES
            if a in j and b in j]


def body_to_joints_2d(body, index_map):
    """{canonical: (u, v)} pixel keypoints for one body, from ZED's own 2D
    detection. Using these rather than reprojecting the 3D points means the
    overlay cannot drift from what the detector actually saw."""
    out = {}
    if body is None:
        return out
    kp2 = getattr(body, "keypoint_2d", None)
    if kp2 is None:
        return out
    for zed_idx, canonical in index_map.items():
        if zed_idx >= len(kp2):
            continue
        u, v = float(kp2[zed_idx][0]), float(kp2[zed_idx][1])
        if u != u or v != v:
            continue
        out[canonical] = (u, v)
    return out


def _blend(img, y, x, rgb, alpha=1.0):
    """Write one BGRA pixel if it is inside the image."""
    h, w = img.shape[:2]
    if 0 <= y < h and 0 <= x < w:
        b, g, r = int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)
        if alpha >= 1.0:
            img[y, x, 0], img[y, x, 1], img[y, x, 2] = b, g, r
        else:
            px = img[y, x]
            px[0] = int(px[0] * (1 - alpha) + b * alpha)
            px[1] = int(px[1] * (1 - alpha) + g * alpha)
            px[2] = int(px[2] * (1 - alpha) + r * alpha)


def draw_line(img, p0, p1, rgb, thickness=3):
    """Sampled line into a BGRA uint8 image. No OpenCV: cv2 in this
    environment wants numpy>=2, which ROS Humble cannot have."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    r = thickness // 2
    for t in np.linspace(0.0, 1.0, n):
        x, y = int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                _blend(img, y + dy, x + dx, rgb)


def draw_point(img, p, rgb, radius=4):
    """Filled square marker; cheap and unambiguous at these sizes."""
    x, y = int(round(float(p[0]))), int(round(float(p[1])))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                _blend(img, y + dy, x + dx, rgb)


def draw_skeleton_2d(img, joints2d, scale=1.0):
    """Draw the bones and keypoints onto a BGRA image in place. scale maps
    detector pixels to the published image size. Returns bones drawn.

    Keypoints outside the frame are dropped: ZED reports an untracked 2D
    keypoint as an out-of-range coordinate rather than NaN, so without this a
    lost wrist drags its bone across the picture as a stray diagonal."""
    h, w = img.shape[:2]

    def inside(p):
        x, y = p[0] * scale, p[1] * scale
        return (x, y) if (0 <= x < w and 0 <= y < h) else None

    pts = {j: xy for j, xy in ((j, inside(p)) for j, p in joints2d.items())
           if xy is not None}
    drawn = 0
    for a, b, rgb in SKELETON_BONES:
        if a in pts and b in pts:
            draw_line(img, pts[a], pts[b], rgb)
            drawn += 1
    for p in pts.values():
        draw_point(img, p, (1.0, 1.0, 1.0))
    return drawn


def urdf_joint_names(out_joints=OUT_JOINTS):
    """OUT_JOINTS -> H1-2 URDF joint names, so a JointState we publish drives
    robot_state_publisher on the same URDF. The waist yaw is 'torso_joint'."""
    return ["torso_joint" if n == "waist_yaw" else f"{n}_joint"
            for n in out_joints]


class RosView:
    """Publishes the live human skeleton and the retargeted H1-2 joint targets
    so both can be watched in RViz. Optional (--ros): rclpy is imported lazily,
    so the node still runs unchanged on a machine with no ROS sourced.

    Topics:
      /zed/human_skeleton  visualization_msgs/MarkerArray  (bones + joints)
      /joint_states        sensor_msgs/JointState          (15 arm targets;
                           every other URDF joint held at 0 so RViz's
                           RobotModel is fully defined)

    Publishing happens after the arm_sdk command, so it never sits inside the
    t0..t3 command path; its own cost is timed separately as ros_ms.
    """

    def __init__(self, frame_id, joint_names, decimate=1, image_decimate=3,
                 cloud_decimate=6,
                 joint_topic="/joint_states", marker_topic="/zed/human_skeleton",
                 image_topic="/zed/left/image"):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import (CameraInfo, Image, JointState,
                                     PointCloud2, PointField)
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster

        self._rclpy = rclpy
        self._Marker = Marker
        self._MarkerArray = MarkerArray
        self.frame_id = frame_id
        self.decimate = max(1, int(decimate))
        self.image_decimate = max(1, int(image_decimate))
        self.cloud_decimate = max(1, int(cloud_decimate))
        self.count = 0
        self.publish_ms = []
        self.image_count = 0
        self.image_ms = []
        self.depth_ms = []
        self.cloud_ms = []
        self.image_scale = 1.0
        self.image_size = (0, 0)
        # Optical frame: RViz projects 3D displays onto the Camera view using
        # this frame plus CameraInfo, so it has to be the optical convention
        # (x right, y down, z forward), not the body one.
        self.optical_frame = frame_id.replace("_link", "") + "_left_optical"
        self._Image = Image
        self._CameraInfo = CameraInfo
        self.cam_info = None

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("zed_realtime_view")
        self.marker_pub = self.node.create_publisher(MarkerArray, marker_topic, 1)
        self.joint_pub = self.node.create_publisher(JointState, joint_topic, 10)
        self.image_pub = self.node.create_publisher(Image, image_topic, 1)
        self.info_pub = self.node.create_publisher(
            CameraInfo, image_topic.rsplit("/", 1)[0] + "/camera_info", 1)
        self.image_topic = image_topic
        self.depth_pub = self.node.create_publisher(Image, "/zed/depth/image", 1)
        self.depth_info_pub = self.node.create_publisher(
            CameraInfo, "/zed/depth/camera_info", 1)
        self.cloud_pub = self.node.create_publisher(PointCloud2, "/zed/points", 1)
        self._PointCloud2 = PointCloud2
        self._PointField = PointField

        # Full URDF joint vector, zeros except the ones we retarget.
        self.all_names = list(joint_names)
        self.slot = {n: i for i, n in enumerate(self.all_names)}
        self.target_names = urdf_joint_names()
        missing = [n for n in self.target_names if n not in self.slot]
        if missing:
            print(f"[realtime] warning: not in URDF, will not be published: "
                  f"{missing}", file=sys.stderr)
        self.js = JointState()
        self.js.name = self.all_names
        self.js.position = [0.0] * len(self.all_names)

        # Identity world -> camera frame, so RViz works with Fixed Frame=world
        # out of the box. Replace with a measured extrinsic if you ever want
        # the human and the robot model in the same physical frame.
        self.static_tf = StaticTransformBroadcaster(self.node)
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = frame_id
        t.transform.rotation.w = 1.0

        # Body frame -> optical frame: the standard ROS camera rotation
        # (-90 deg about Z then -90 about X), so markers land on the image.
        o = TransformStamped()
        o.header.stamp = t.header.stamp
        o.header.frame_id = frame_id
        o.child_frame_id = self.optical_frame
        o.transform.rotation.x = -0.5
        o.transform.rotation.y = 0.5
        o.transform.rotation.z = -0.5
        o.transform.rotation.w = 0.5
        self.static_tf.sendTransform([t, o])
        print(f"[realtime] ROS view up: {marker_topic} (MarkerArray), "
              f"{joint_topic} (JointState), frame '{frame_id}'")

    def _stamp(self):
        return self.node.get_clock().now().to_msg()

    def _marker(self, ns, mid, mtype, scale):
        m = self._Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self._stamp()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = self._Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        # Opaque white fallback. Per-point colors win where we set them; this
        # keeps the marker visible rather than alpha-0 if they are ever absent.
        m.color = self._color((1.0, 1.0, 1.0))
        # Expire if the loop stops or the body is lost, so RViz never shows a
        # frozen skeleton as if it were live.
        m.lifetime = self._duration(0.25)
        return m

    def _duration(self, seconds):
        from builtin_interfaces.msg import Duration
        d = Duration()
        d.sec = int(seconds)
        d.nanosec = int((seconds - int(seconds)) * 1e9)
        return d

    def _point(self, p):
        from geometry_msgs.msg import Point
        pt = Point()
        pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
        return pt

    def _color(self, rgb, a=1.0):
        from std_msgs.msg import ColorRGBA
        c = ColorRGBA()
        c.r, c.g, c.b, c.a = float(rgb[0]), float(rgb[1]), float(rgb[2]), a
        return c

    def publish(self, joints, pose_cmd):
        """joints: {canonical: xyz} in the ZED frame. pose_cmd: 15 rad."""
        self.count += 1
        if self.count % self.decimate:
            return
        t_start = time.monotonic()

        arr = self._MarkerArray()
        bones = self._marker("bones", 0, self._Marker.LINE_LIST, 0.02)
        for pa, pb, rgb in skeleton_segments(joints):
            bones.points.extend([self._point(pa), self._point(pb)])
            bones.colors.extend([self._color(rgb), self._color(rgb)])
        arr.markers.append(bones)

        pts = self._marker("joints", 1, self._Marker.SPHERE_LIST, 0.04)
        pts.color = self._color((1.0, 1.0, 1.0))
        for p in joints.values():
            pts.points.append(self._point(p))
        arr.markers.append(pts)
        self.marker_pub.publish(arr)

        for name, q in zip(self.target_names, pose_cmd):
            i = self.slot.get(name)
            if i is not None:
                self.js.position[i] = float(q)
        self.js.header.stamp = self._stamp()
        self.joint_pub.publish(self.js)

        self.publish_ms.append((time.monotonic() - t_start) * 1000.0)

    def set_camera(self, fx, fy, cx, cy, width, height, out_width=0):
        """Store the left-camera intrinsics, scaled if we publish smaller
        frames. Called once after the ZED is open."""
        scale = 1.0
        if out_width and out_width < width:
            scale = float(out_width) / float(width)
        w, h = int(round(width * scale)), int(round(height * scale))
        info = self._CameraInfo()
        info.header.frame_id = self.optical_frame
        info.width, info.height = w, h
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5          # VIEW.LEFT is rectified
        k = [fx * scale, 0.0, cx * scale,
             0.0, fy * scale, cy * scale,
             0.0, 0.0, 1.0]
        info.k = k
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [k[0], 0.0, k[2], 0.0,
                  0.0, k[4], k[5], 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self.cam_info = info
        self.image_scale = scale
        self.image_size = (w, h)
        print(f"[realtime] camera info: {w}x{h} "
              f"fx={k[0]:.1f} fy={k[4]:.1f} cx={k[2]:.1f} cy={k[5]:.1f}"
              f"{' (downscaled)' if scale != 1.0 else ''}")
        return w, h

    def wants_image(self, frame_idx):
        return (self.cam_info is not None
                and frame_idx % self.image_decimate == 0)

    def wants_cloud(self, frame_idx):
        return frame_idx % self.cloud_decimate == 0

    def publish_image(self, bgra, joints2d=None):
        """bgra: HxWx4 uint8 from ZED. joints2d: detector-pixel keypoints to
        draw on top (already at the published scale if scale is 1)."""
        if self.cam_info is None or bgra is None or bgra.size == 0:
            return
        t_start = time.monotonic()
        img = np.ascontiguousarray(bgra)
        if joints2d:
            draw_skeleton_2d(img, joints2d, scale=self.image_scale)

        msg = self._Image()
        msg.header.stamp = self._stamp()
        msg.header.frame_id = self.optical_frame
        msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
        msg.encoding = "bgra8"
        msg.is_bigendian = 0
        msg.step = int(img.shape[1] * 4)
        # array.array('B', ...) hits rclpy's fast path. Assigning plain bytes
        # falls into a per-element Python validation loop instead: 31 ms for a
        # 640x360 frame here, vs 0.9 ms this way.
        msg.data = array.array("B", img.tobytes())
        self.image_pub.publish(msg)

        self.cam_info.header.stamp = msg.header.stamp
        self.info_pub.publish(self.cam_info)
        self.image_count += 1
        self.image_ms.append((time.monotonic() - t_start) * 1000.0)

    def publish_depth(self, depth):
        """depth: HxW float32 metres from MEASURE.DEPTH. Published as 32FC1 in
        the optical frame, which is correct because ZED's DEPTH is distance
        along the forward axis (verified: DEPTH == +X in Z_UP_X_FWD, exactly).
        Invalid pixels stay NaN/inf; RViz and depth_image_proc both expect that."""
        if self.cam_info is None or depth is None or depth.size == 0:
            return
        t_start = time.monotonic()
        d = np.ascontiguousarray(depth, dtype=np.float32)
        msg = self._Image()
        msg.header.stamp = self._stamp()
        msg.header.frame_id = self.optical_frame
        msg.height, msg.width = int(d.shape[0]), int(d.shape[1])
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = int(d.shape[1] * 4)
        msg.data = array.array("B", d.tobytes())
        self.depth_pub.publish(msg)

        self.cam_info.header.stamp = msg.header.stamp
        self.depth_info_pub.publish(self.cam_info)
        self.depth_ms.append((time.monotonic() - t_start) * 1000.0)

    def publish_cloud(self, xyzrgba):
        """xyzrgba: HxWx4 float32 from MEASURE.XYZRGBA. x,y,z floats then the
        color packed into the 4th float. ZED packs that color as RGBA in memory
        (bytes R,G,B,A), but the PointCloud2 'rgb' field that RViz renders as
        RGB8 reads the color float little-endian and expects BGRA byte order, so
        the two disagree and red/blue come out swapped. Swap R and B into BGRA
        before publishing. (VIEW.LEFT, used for the 2D image, really is BGRA;
        only this measure is RGBA.) XYZ is in the ZED body frame, the same frame
        as the skeleton markers, so cloud and skeleton line up in RViz."""
        if xyzrgba is None or xyzrgba.size == 0:
            return
        t_start = time.monotonic()
        a = np.ascontiguousarray(xyzrgba, dtype=np.float32)
        h, w = int(a.shape[0]), int(a.shape[1])
        # Swap the color's R and B (RGBA -> BGRA for the rgb field). Work on a
        # private byte copy so the SDK's Mat is never mutated; this is the one
        # buffer copy we would make for the message anyway.
        buf = bytearray(a.tobytes())
        u8 = np.frombuffer(buf, np.uint8).reshape(h, w, 16)
        u8[:, :, [12, 14]] = u8[:, :, [14, 12]]     # RGBA -> BGRA (fancy index copies the RHS)
        F = self._PointField
        msg = self._PointCloud2()
        msg.header.stamp = self._stamp()
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = h, w
        msg.fields = [
            F(name="x", offset=0, datatype=F.FLOAT32, count=1),
            F(name="y", offset=4, datatype=F.FLOAT32, count=1),
            F(name="z", offset=8, datatype=F.FLOAT32, count=1),
            F(name="rgb", offset=12, datatype=F.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * w
        msg.is_dense = False          # unmeasured pixels are NaN
        msg.data = array.array("B", buf)     # buf carries the R<->B swap
        self.cloud_pub.publish(msg)
        self.cloud_ms.append((time.monotonic() - t_start) * 1000.0)

    def close(self):
        try:
            self.node.destroy_node()
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass


def latency_stats(rows):
    """Median / p95 (ms) for each t0->t1, t1->t2, t2->t3, t0->t3."""
    if not rows:
        return {}
    a = np.array(rows)  # (n, 5): frame, t0, t1, t2, t3
    stages = {
        "grab->skeleton": (a[:, 2] - a[:, 1]) * 1000,
        "skeleton->ik":   (a[:, 3] - a[:, 2]) * 1000,
        "ik->publish":    (a[:, 4] - a[:, 3]) * 1000,
        "end-to-end":     (a[:, 4] - a[:, 1]) * 1000,
    }
    stats = {}
    for name, arr in stages.items():
        stats[name] = (float(np.median(arr)), float(np.percentile(arr, 95)),
                       float(np.max(arr)))
    if len(a) >= 2:
        dt = np.diff(a[:, 1])
        stats["achieved_hz"] = (float(1.0 / np.median(dt)),
                                float(1.0 / np.percentile(dt, 95)),
                                float(1.0 / np.max(dt)))
    return stats


def print_stats(stats, frames, held_frames):
    if not stats:
        print("[realtime] no frames logged")
        return
    print("=" * 66)
    print(f"[realtime] {frames} frames processed, {held_frames} with no body "
          f"(held last pose)")
    for name, (med, p95, mx) in stats.items():
        unit = "Hz" if name == "achieved_hz" else "ms"
        print(f"  {name:20s}  median={med:7.2f}{unit}  p95={p95:7.2f}{unit}  "
              f"max={mx:7.2f}{unit}")


def write_latency_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t0_grab", "t1_pose", "t2_ik", "t3_publish"])
        w.writerows(rows)
    print(f"[realtime] latency log -> {path}")


def wait_enter(prompt):
    """Blocking prompt; returns True on ENTER, False on Ctrl+C."""
    try:
        input(prompt)
        return True
    except (KeyboardInterrupt, EOFError):
        return False


def stream(args, arm_sdk, zed, sl, fks, limits, index_map, gravity, ros=None):
    """The 50-Hz control loop. Returns (rows, frames, held_frames)."""
    bodies = sl.Bodies()
    runtime = sl.BodyTrackingRuntimeParameters()
    runtime.detection_confidence_threshold = args.min_conf
    img_mat = sl.Mat() if (ros is not None and args.ros_image) else None
    depth_mat = sl.Mat() if (ros is not None and args.ros_depth) else None
    cloud_mat = sl.Mat() if (ros is not None and args.ros_cloud) else None
    cloud_res = (sl.Resolution(args.ros_cloud_width,
                               max(1, round(args.ros_cloud_width * 9 / 16)))
                 if cloud_mat is not None and args.ros_cloud_width else None)

    euro = OneEuro(min_cutoff=args.euro_min_cutoff, beta=args.euro_beta)
    q_prev = {s: np.zeros(4) for s in SIDES}
    pose_prev = HOME_POSE[:]

    rows, held_frames = [], 0
    frame_idx = 0
    last_t = time.monotonic()

    print("[realtime] streaming (Ctrl+C to stop)")
    while not _stop:
        grab_err = zed.grab()
        if grab_err != sl.ERROR_CODE.SUCCESS:
            if grab_err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            print(f"[realtime] grab error: {grab_err}", file=sys.stderr)
            continue
        t0 = time.monotonic()
        zed.retrieve_bodies(bodies, runtime)
        body = pick_primary_body(bodies)
        j = body_to_joints_dict(body, index_map)
        R = torso_frame(j) if j else None

        arms_ok = R is not None
        if arms_ok:
            targets_per_side = {}
            for side in SIDES:
                ad = arm_directions(j, R, side)
                if ad is None:
                    arms_ok = False
                    break
                u_hat, f_hat, flex_deg = ad
                targets_per_side[side] = (u_hat, f_hat, flex_deg)

        t1 = time.monotonic()

        if arms_ok:
            solved = {}
            for side in SIDES:
                u_hat, f_hat, flex_deg = targets_per_side[side]
                lim4 = limits[side]
                q4, _res, _cl = solve_arm(
                    fks[side], u_hat, f_hat, q_prev[side], lim4,
                    suppress_yaw=(flex_deg < STRAIGHT_ARM_DEG),
                    yaw_reg=args.yaw_reg, yaw_neutral=args.yaw_neutral)
                solved[side] = q4
                q_prev[side] = q4
            target_pose = assemble_pose(solved)
            target_pose = euro(target_pose, t=t1).tolist()
        else:
            held_frames += 1
            target_pose = pose_prev  # hold last commanded pose

        dt = max(t1 - last_t, 1e-3)
        last_t = t1
        pose_cmd = clamp_velocity(pose_prev, target_pose, args.max_vel, dt)
        t2 = time.monotonic()

        arm_sdk.publish(pose_cmd)
        pose_prev = pose_cmd
        t3 = time.monotonic()

        rows.append((frame_idx, t0, t1, t2, t3))
        # After t3 on purpose: RViz must never delay the robot command.
        if ros is not None:
            ros.publish(j, pose_cmd)
            if img_mat is not None and ros.wants_image(frame_idx):
                if ros.image_scale != 1.0:
                    zed.retrieve_image(img_mat, sl.VIEW.LEFT, sl.MEM.CPU,
                                       sl.Resolution(*ros.image_size))
                else:
                    zed.retrieve_image(img_mat, sl.VIEW.LEFT)
                kp2 = (body_to_joints_2d(body, index_map)
                       if args.ros_overlay else None)
                ros.publish_image(img_mat.get_data(), kp2)
            if depth_mat is not None and ros.wants_image(frame_idx):
                if ros.image_scale != 1.0:
                    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU,
                                         sl.Resolution(*ros.image_size))
                else:
                    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                ros.publish_depth(depth_mat.get_data())
            if cloud_mat is not None and ros.wants_cloud(frame_idx):
                if cloud_res is not None:
                    zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA,
                                         sl.MEM.CPU, cloud_res)
                else:
                    zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
                ros.publish_cloud(cloud_mat.get_data())
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    return rows, frame_idx, held_frames


def wait_for_person(zed, sl, index_map, args):
    """Return once we've seen a body in N consecutive frames (or Ctrl+C)."""
    bodies = sl.Bodies()
    runtime = sl.BodyTrackingRuntimeParameters()
    runtime.detection_confidence_threshold = args.min_conf
    print(f"[realtime] waiting for a person in the ZED FoV "
          f"(need {args.warmup_frames} consecutive detections)...")
    hit = 0
    while not _stop and hit < args.warmup_frames:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.02)
            continue
        zed.retrieve_bodies(bodies, runtime)
        body = pick_primary_body(bodies)
        j = body_to_joints_dict(body, index_map)
        hit = hit + 1 if torso_frame(j) is not None else 0
    if not _stop:
        print(f"[realtime] person detected ({hit} consecutive frames)")


def run(args):
    global _stop
    signal.signal(signal.SIGINT, _sigint)

    import pyzed.sl as sl

    # ── URDF / IK setup ─────────────────────────────────────────────────────
    if not os.path.isfile(args.urdf):
        sys.exit(f"[realtime] URDF not found: {args.urdf}")
    fks = {s: ArmChainFK(args.urdf, s) for s in SIDES}
    all_limits = load_limits(args.urdf)
    limits = {s: [all_limits[f"{s}_shoulder_pitch_joint"],
                  all_limits[f"{s}_shoulder_roll_joint"],
                  all_limits[f"{s}_shoulder_yaw_joint"],
                  all_limits[f"{s}_elbow_joint"]] for s in SIDES}

    # ── Gravity feedforward (default ON, frozen gain) ───────────────────────
    ros = None
    if args.ros:
        ros = RosView(args.ros_frame, sorted(all_limits),
                      decimate=args.ros_decimate,
                      image_decimate=args.ros_image_decimate,
                      cloud_decimate=args.ros_cloud_decimate)

    gravity = {}
    if args.gravity_ff:
        for s in SIDES:
            gravity[s] = GravityFF(args.urdf, s)

    # ── Open ZED + body tracking (FAST for latency) ─────────────────────────
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    if args.svo:
        # Dev only: replay a recording through the live path so the RViz view
        # and the IK can be checked without a person in front of the camera.
        # Not a latency measurement - SVO grab pacing is not the camera's.
        if not os.path.isfile(args.svo):
            sys.exit(f"[realtime] SVO not found: {args.svo}")
        init.set_from_svo_file(args.svo)
        print(f"[realtime] SVO playback (dev): {args.svo}")
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"[realtime] ZED open failed: {err}")

    try:
        if args.record:
            rec = sl.RecordingParameters(args.record, sl.SVO_COMPRESSION_MODE.H264)
            rerr = zed.enable_recording(rec)
            if rerr != sl.ERROR_CODE.SUCCESS:
                sys.exit(f"[realtime] enable_recording failed: {rerr}")
            print(f"[realtime] recording SVO to: {args.record}")

        pos = sl.PositionalTrackingParameters()
        pos.set_as_static = args.static
        zed.enable_positional_tracking(pos)

        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = False
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        body_params.body_format = getattr(sl.BODY_FORMAT, f"BODY_{args.body_format}")
        zed.enable_body_tracking(body_params)

        idx_map = canonical_index_map(body_params.body_format)

        if ros is not None and (args.ros_image or args.ros_depth):
            cc = zed.get_camera_information().camera_configuration
            lc = cc.calibration_parameters.left_cam
            ros.set_camera(lc.fx, lc.fy, lc.cx, lc.cy,
                           cc.resolution.width, cc.resolution.height,
                           args.ros_image_width)

        if not args.allow_no_body:
            wait_for_person(zed, sl, idx_map, args)
        if _stop:
            return 0

        # ── Arm sdk: snapshot, fade in, go to home ──────────────────────────
        arm_sdk = ArmSdk(OUT_JOINTS, args.interface, args.dry_run,
                         gravity=gravity, gravity_gain=args.gravity_gain)
        snapshot = arm_sdk.snapshot()
        arm_sdk.pose = snapshot
        print(f"[realtime] arm snapshot: "
              f"L_ShP={snapshot[0]:+.2f} R_ShP={snapshot[7]:+.2f}")

        if not wait_enter("[realtime] ENTER to arm arm_sdk (Ctrl+C to abort)... "):
            print("[realtime] aborted before arming")
            return 0

        print("[realtime] fading weight 0 -> 1 (2 s)")
        arm_sdk.ramp_weight(1.0, 2.0)
        print("[realtime] ramping to home pose (3 s)")
        arm_sdk.ramp_pose(HOME_POSE, 3.0)

        if not wait_enter("[realtime] ENTER to start streaming (Ctrl+C safe)... "):
            print("[realtime] aborted before streaming")
        else:
            rows, frames, held = stream(args, arm_sdk, zed, sl, fks, limits,
                                        idx_map, gravity, ros=ros)
            if args.latency_log:
                write_latency_csv(rows, args.latency_log)
            if args.print_stats:
                print_stats(latency_stats(rows), frames, held)
                if ros is not None and ros.publish_ms:
                    ms = np.array(ros.publish_ms)
                    print(f"  {'ros publish':<16s} median {np.median(ms):6.2f} ms"
                          f"   p95 {np.percentile(ms, 95):6.2f} ms"
                          f"   ({len(ms)} msgs)")
                for label, series in (("ros image", ros.image_ms if ros else []),
                                      ("ros depth", ros.depth_ms if ros else []),
                                      ("ros cloud", ros.cloud_ms if ros else [])):
                    if series:
                        ms = np.array(series)
                        print(f"  {label:<16s} median {np.median(ms):6.2f} ms"
                              f"   p95 {np.percentile(ms, 95):6.2f} ms"
                              f"   ({len(ms)} frames)")

        # ── Safe shutdown ───────────────────────────────────────────────────
        print("[realtime] ramping back to snapshot (3 s)")
        arm_sdk.ramp_pose(snapshot, 3.0)
        print("[realtime] fading weight 1 -> 0 (2 s)")
        arm_sdk.ramp_weight(0.0, 2.0)
        return 0
    finally:
        if ros is not None:
            ros.close()
        if args.record:
            zed.disable_recording()
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()


def self_test():
    """No ZED / DDS / URDF — exercise helpers only."""
    # assemble_pose
    pose = assemble_pose({"left": [0.1, 0.2, 0.3, 0.4],
                          "right": [-0.1, -0.2, -0.3, -0.4]})
    assert len(pose) == 15, len(pose)
    assert pose[:4] == [0.1, 0.2, 0.3, 0.4], pose[:4]
    assert pose[4:7] == [0.0, 0.0, 0.0], pose[4:7]
    assert pose[7:11] == [-0.1, -0.2, -0.3, -0.4], pose[7:11]
    assert pose[14] == 0.0

    # clamp_velocity: no prev -> passthrough
    assert clamp_velocity(None, [0.5], max_vel=4.0, dt=0.02) == [0.5]
    # clamp_velocity: 10 rad/s delta over 20 ms should clamp to 4*0.02=0.08 rad
    out = clamp_velocity([0.0], [1.0], max_vel=4.0, dt=0.02)
    assert abs(out[0] - 0.08) < 1e-9, out

    # body_to_joints_dict: NaN drop
    class FB: pass
    fb = FB()
    fb.keypoint = np.array([[0.1, 0.2, 0.3],
                            [float("nan"), 0.0, 0.0]])
    fb.keypoint_confidence = np.array([80.0, 50.0])
    d = body_to_joints_dict(fb, {0: "left_shoulder", 1: "left_elbow"})
    assert list(d.keys()) == ["left_shoulder"], d

    # latency_stats
    rows = [(0, 0.000, 0.010, 0.020, 0.030),
            (1, 0.020, 0.028, 0.036, 0.041),
            (2, 0.040, 0.049, 0.055, 0.062)]
    s = latency_stats(rows)
    assert "end-to-end" in s and "achieved_hz" in s
    assert s["end-to-end"][0] > 0

    # urdf_joint_names: waist yaw is the URDF torso_joint
    names = urdf_joint_names()
    assert len(names) == 15, names
    assert names[0] == "left_shoulder_pitch_joint", names[0]
    assert names[-1] == "torso_joint", names[-1]

    # skeleton_segments: only bones with both endpoints tracked
    jd = {"left_shoulder": np.zeros(3), "right_shoulder": np.ones(3),
          "left_elbow": np.zeros(3)}
    segs = skeleton_segments(jd)
    assert len(segs) == 2, segs        # L shoulder-elbow + shoulder-shoulder
    assert skeleton_segments({}) == []

    # body_to_joints_2d: NaN drop, same index map as the 3D path
    fb.keypoint_2d = np.array([[100.0, 200.0], [float("nan"), 5.0]])
    d2 = body_to_joints_2d(fb, {0: "left_shoulder", 1: "left_elbow"})
    assert d2 == {"left_shoulder": (100.0, 200.0)}, d2

    # draw_skeleton_2d: bones land on the image, scale is applied
    img = np.zeros((40, 40, 4), dtype=np.uint8)
    n = draw_skeleton_2d(img, {"left_shoulder": (10, 10), "right_shoulder": (30, 10),
                               "left_elbow": (10, 30)})
    assert n == 2, n                      # shoulders + left upper arm
    assert img[10, 20, 1] > 0, "shoulder bone not drawn"
    assert img[30, 10].any(), "elbow keypoint not drawn"
    img2 = np.zeros((40, 40, 4), dtype=np.uint8)
    draw_skeleton_2d(img2, {"left_shoulder": (20, 20), "right_shoulder": (60, 20)},
                     scale=0.5)
    assert img2[10, 20].any(), "scale not applied"
    # out-of-frame keypoints are dropped, not drawn as a stray bone
    img3 = np.zeros((40, 40, 4), dtype=np.uint8)
    n3 = draw_skeleton_2d(img3, {"left_shoulder": (10, 10),
                                 "left_elbow": (-500, -500)})
    assert n3 == 0, n3
    assert not img3[:, :3].any() or img3[10, 10].any()
    assert img3[0, 0].sum() == 0, "drew toward an out-of-frame joint"

    print("[self-test] OK - assemble_pose, clamp_velocity, body dict, "
          "latency stats, ROS joint names, skeleton bones, 2D overlay")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=os.path.expanduser(
        "~/mj_ws/assets/h1_2_description/h1_2.urdf"))
    ap.add_argument("--interface", default="",
                    help="robot network interface (e.g. enp128s31f6)")
    ap.add_argument("--body-format", choices=["18", "34", "38"], default="38")
    ap.add_argument("--resolution",
                    choices=["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"],
                    default="HD720", help="lower = faster grab (default HD720)")
    ap.add_argument("--min-conf", type=int, default=40)
    ap.add_argument("--static", action="store_true",
                    help="camera stationary; better positional-tracking perf")
    ap.add_argument("--record", default="",
                    help="also record the live stream to this SVO2 path")
    ap.add_argument("--svo", default="",
                    help="dev only: replay this SVO2 instead of the live "
                         "camera (for RViz / IK checks; latency numbers from "
                         "an SVO run are meaningless)")

    ap.add_argument("--yaw-neutral", type=float, default=0.02,
                    help="frozen 0.02 (branch-select toward natural yaw)")
    ap.add_argument("--yaw-reg", type=float, default=0.05,
                    help="frozen 0.05 (continuity)")
    ap.add_argument("--euro-min-cutoff", type=float, default=1.0)
    ap.add_argument("--euro-beta", type=float, default=0.5)
    ap.add_argument("--max-vel", type=float, default=4.0,
                    help="per-joint rate limit (rad/s)")

    ap.add_argument("--gravity-ff", action=argparse.BooleanOptionalAction,
                    default=True, help="gravity feedforward (default ON)")
    ap.add_argument("--gravity-gain", type=float, default=FROZEN_GRAVITY_GAIN)

    ap.add_argument("--warmup-frames", type=int, default=10,
                    help="require this many consecutive body detections before arming")
    ap.add_argument("--allow-no-body", action="store_true",
                    help="skip the person-detection warmup (dev only)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = no limit)")

    ap.add_argument("--ros", action="store_true",
                    help="publish the human skeleton (MarkerArray) and the "
                         "retargeted joint targets (JointState) for RViz")
    ap.add_argument("--ros-frame", default="zed_camera_link",
                    help="frame_id for the skeleton markers (default "
                         "zed_camera_link; an identity world-> this static TF "
                         "is published so RViz works with Fixed Frame=world)")
    ap.add_argument("--ros-decimate", type=int, default=1,
                    help="publish every Nth frame (default 1 = every frame)")
    ap.add_argument("--ros-image", action="store_true",
                    help="also publish the left camera image + CameraInfo, so "
                         "RViz's Camera display shows the skeleton on the video")
    ap.add_argument("--ros-overlay", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="draw the 2D skeleton into the published image "
                         "(default ON; the 3D markers overlay it either way)")
    ap.add_argument("--ros-image-width", type=int, default=640,
                    help="downscale published frames to this width "
                         "(0 = native; default 640 keeps DDS light)")
    ap.add_argument("--ros-image-decimate", type=int, default=3,
                    help="publish every Nth frame as an image (default 3)")
    ap.add_argument("--ros-depth", action="store_true",
                    help="publish the depth map (32FC1 metres) on "
                         "/zed/depth/image + /zed/depth/camera_info")
    ap.add_argument("--ros-cloud", action="store_true",
                    help="publish the coloured point cloud on /zed/points, in "
                         "the same frame as the skeleton markers")
    ap.add_argument("--ros-cloud-width", type=int, default=320,
                    help="point cloud width (0 = native; default 320, since a "
                         "640x360 cloud is 3.7 MB per message)")
    ap.add_argument("--ros-cloud-decimate", type=int, default=6,
                    help="publish every Nth frame as a cloud (default 6)")

    ap.add_argument("--latency-log", default="",
                    help="write t0..t3 CSV to this path")
    ap.add_argument("--print-stats", action="store_true", default=True,
                    help="print latency median/p95 at exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="no DDS: ZED + IK + logging only, no arm_sdk publish")
    ap.add_argument("--self-test", action="store_true")

    args = ap.parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
