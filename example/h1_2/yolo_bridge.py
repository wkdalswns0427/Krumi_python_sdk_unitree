#!/usr/bin/env python3
"""ROS 2 -> UNIX socket bridge for /yolo/detections person events.

Runs OUTSIDE the conda env, in system Python 3.10 where ROS Humble's rclpy
loads correctly. Forwards a one-line JSON event per qualifying detection
to the UNIX socket at SOCKET_PATH so a non-ROS Python (e.g. conda 3.11) can
react. The socket lives inside the yolo_ws directory so it survives /tmp
being wiped and stays next to the ROS package that owns it.

Usage:
    # In a shell with NO conda active and ROS Humble sourced:
    source /opt/ros/humble/setup.bash
    python3 yolo_bridge.py

Event format (one JSON object per line, UTF-8):
    {"ts": 1717250000.123, "class": "person", "conf": 0.87}

Multiple clients may connect simultaneously; each gets a copy. The socket
is recreated on every start (any stale file is unlinked).
"""

import json
import os
import socket
import time

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray


SOCKET_PATH    = "/home/mchang344/mj_ws/h1-2_sensors/yolo_ws/yolo_bridge.sock"
MIN_CONF       = 0.5      # match wave_on_person.py default
# TARGET_CLASS   = "person"
TARGET_CLASSES = {"person", "traffic cone"}   # must match yolo_detector.py set_classes()
EVENT_THROTTLE = 0.1      # min seconds between forwarded events (per class)


class Bridge(Node):
    def __init__(self, listen_sock: socket.socket):
        super().__init__("yolo_bridge")
        self.listen_sock = listen_sock
        self._peers: list[socket.socket] = []
        self.last_emit = 0.0
        self.create_subscription(
            Detection2DArray, "/yolo/detections", self._on_detections, 10,
        )
        self.get_logger().info(
            f"bridging /yolo/detections -> {SOCKET_PATH} "
            f"(classes={sorted(TARGET_CLASSES)}, min_conf={MIN_CONF})"
        )

    def _on_detections(self, msg: Detection2DArray):
        now = time.time()
        if now - self.last_emit < EVENT_THROTTLE:
            return
        for det in msg.detections:
            for hyp in det.results:
                cls = hyp.hypothesis.class_id
                score = float(hyp.hypothesis.score)
                # if cls == TARGET_CLASS and score >= MIN_CONF:
                if cls in TARGET_CLASSES and score >= MIN_CONF:
                    payload = json.dumps(
                        {"ts": now, "class": cls, "conf": score}
                    ) + "\n"
                    self._broadcast(payload.encode("utf-8"))
                    self.last_emit = now
                    return

    def _broadcast(self, data: bytes):
        dead = []
        for c in self._peers:
            try:
                c.sendall(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead.append(c)
        for c in dead:
            try:
                c.close()
            except Exception:
                pass
            self._peers.remove(c)
            self.get_logger().info(f"client disconnected ({len(self._peers)} left)")

    def accept_pending(self):
        try:
            client, _ = self.listen_sock.accept()
        except BlockingIOError:
            return
        client.setblocking(False)
        self._peers.append(client)
        self.get_logger().info(f"client connected ({len(self._peers)} total)")


def main():
    # Remove any stale socket file
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass

    listen = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listen.bind(SOCKET_PATH)
    listen.listen(8)
    listen.setblocking(False)
    os.chmod(SOCKET_PATH, 0o666)   # let any user on the box connect

    rclpy.init()
    bridge = Bridge(listen)
    try:
        while rclpy.ok():
            rclpy.spin_once(bridge, timeout_sec=0.05)
            bridge.accept_pending()
    except KeyboardInterrupt:
        print("\n[bridge] shutting down")
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
        listen.close()
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
