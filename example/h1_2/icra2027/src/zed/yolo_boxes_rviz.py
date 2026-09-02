#!/usr/bin/env python3
"""yolo_boxes_rviz.py - draw YOLO detections as 3D bounding boxes in RViz.

Bridges yolo_h1's `yolo_detector` to the ZED RViz view. It subscribes to
`/yolo/detections` (vision_msgs/Detection2DArray) and publishes a
visualization_msgs/MarkerArray of wireframe 3D boxes on `/yolo/boxes`, plus a
floating text label per object. Add a MarkerArray display on `/yolo/boxes` to
zed_h12.rviz (already wired if you use the shipped config) and the boxes appear
in the same 3D scene as the point cloud and the human skeleton.

Why a separate node, not a change to zed_realtime_node.py: the realtime node
opens the ZED SDK directly and must stay on its 50 Hz control budget, and YOLO
already runs as its own node. This bridge is pure message translation, so it can
lag or die without touching the camera or the robot.

Each Detection2D carries the 2D box (bbox.center + size, pixels) and a 3D
deprojected centre (results[0].pose.pose.position, in the image's optical
frame). The box's metric width/height are recovered from the pixel size at that
depth using the camera intrinsics: w = size_x * z / fx, h = size_y * z / fy.
The depth extent is not measured, so the box is given a fixed, shallow depth
(--box-depth) rather than a fabricated one. Markers are stamped in the
detection's own frame_id, so this works against either the custom node
(optical frame from its static TF) or the ZED ROS 2 wrapper unchanged.

Run (custom-node pipeline; feed YOLO from the custom node's topics):
    source ~/mj_ws/h1-2_sensors/yolo_ws/install/setup.bash
    ros2 run yolo_h1 yolo_detector --ros-args \
        -p rgb_topic:=/zed/left/image \
        -p depth_topic:=/zed/depth/image \
        -p camera_info_topic:=/zed/left/camera_info
    /usr/bin/python3 yolo_boxes_rviz.py         # defaults match the custom node

For the ZED ROS 2 wrapper instead, leave yolo_detector on its defaults and run
    /usr/bin/python3 yolo_boxes_rviz.py --info-topic /zed/zed_node/rgb/camera_info
"""

import argparse
import colorsys

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


# Fixed colors for the classes yolo_detector prompts by default; any other
# open-vocabulary label gets a stable color hashed from its text.
CLASS_COLORS = {
    "person": (0.20, 0.80, 1.00),
    "traffic cone": (1.00, 0.55, 0.10),
}


def color_for(label):
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    h = (hash(label) % 360) / 360.0
    return colorsys.hsv_to_rgb(h, 0.65, 1.0)


def box_edges(cx, cy, cz, w, h, d):
    """24 endpoints (12 edges) of an axis-aligned box centred at (cx,cy,cz),
    in the optical frame: x right, y down, z forward. Half-extents w/2, h/2, d/2."""
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    z0, z1 = cz - d / 2, cz + d / 2
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),   # front face
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]   # back face
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),      # front
             (4, 5), (5, 6), (6, 7), (7, 4),      # back
             (0, 4), (1, 5), (2, 6), (3, 7)]      # connectors
    pts = []
    for a, b in edges:
        pts.append(c[a])
        pts.append(c[b])
    return pts


class YoloBoxesRviz(Node):

    def __init__(self, args):
        super().__init__("yolo_boxes_rviz")
        self.box_depth = float(args.box_depth)
        self.line_width = float(args.line_width)
        self.lifetime = float(args.lifetime)
        self.fx = self.fy = None
        self._warned_info = False

        self.create_subscription(CameraInfo, args.info_topic, self.info_cb, 1)
        self.create_subscription(
            Detection2DArray, args.det_topic, self.det_cb, 10)
        self.pub = self.create_publisher(MarkerArray, args.out_topic, 1)
        self.get_logger().info(
            f"in: {args.det_topic} + {args.info_topic}  out: {args.out_topic}  "
            f"box_depth={self.box_depth} m")

    def info_cb(self, msg):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.get_logger().info(f"intrinsics: fx={self.fx:.1f} fy={self.fy:.1f}")

    def _duration(self, seconds):
        from builtin_interfaces.msg import Duration
        d = Duration()
        d.sec = int(seconds)
        d.nanosec = int((seconds - int(seconds)) * 1e9)
        return d

    def det_cb(self, msg):
        arr = MarkerArray()
        # Clear the previous frame first so boxes vanish when objects leave,
        # rather than lingering until their lifetime expires.
        clear = Marker()
        clear.header = msg.header
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        if self.fx is None and not self._warned_info:
            self.get_logger().warn(
                "no CameraInfo yet; drawing fallback-sized boxes until it arrives")
            self._warned_info = True

        for i, det in enumerate(msg.detections):
            if not det.results:
                continue
            hyp = det.results[0]
            label = hyp.hypothesis.class_id
            score = hyp.hypothesis.score
            p = hyp.pose.pose.position
            z = p.z
            if self.fx:
                w = det.bbox.size_x * z / self.fx
                h = det.bbox.size_y * z / self.fy
            else:                       # no intrinsics yet: reasonable default
                w, h = 0.4, 0.6
            r, g, b = color_for(label)

            box = Marker()
            box.header = msg.header
            box.ns = "boxes"
            box.id = i
            box.type = Marker.LINE_LIST
            box.action = Marker.ADD
            box.pose.orientation.w = 1.0
            box.scale.x = self.line_width
            box.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)
            box.lifetime = self._duration(self.lifetime)
            box.points = [Point(x=float(x), y=float(y), z=float(zz))
                          for x, y, zz in box_edges(p.x, p.y, z, w, h,
                                                    self.box_depth)]
            arr.markers.append(box)

            txt = Marker()
            txt.header = msg.header
            txt.ns = "labels"
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = p.x
            txt.pose.position.y = p.y - h / 2 - 0.05      # just above the box top
            txt.pose.position.z = z
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.09                            # text height, metres
            txt.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)
            txt.lifetime = self._duration(self.lifetime)
            txt.text = f"{label} {score:.2f}  {z:.1f} m"
            arr.markers.append(txt)

        self.pub.publish(arr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-topic", default="/yolo/detections")
    ap.add_argument("--info-topic", default="/zed/left/camera_info",
                    help="CameraInfo for fx/fy (custom node default; use the "
                         "wrapper's rgb/camera_info if running the wrapper)")
    ap.add_argument("--out-topic", default="/yolo/boxes")
    ap.add_argument("--box-depth", type=float, default=0.30,
                    help="forward extent of the drawn box, metres (object depth "
                         "is not measured)")
    ap.add_argument("--line-width", type=float, default=0.012)
    ap.add_argument("--lifetime", type=float, default=0.5,
                    help="seconds a box persists if detections stop")
    args = ap.parse_args()

    rclpy.init()
    node = YoloBoxesRviz(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
