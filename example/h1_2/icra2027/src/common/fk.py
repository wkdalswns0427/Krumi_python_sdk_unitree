#!/usr/bin/env python3
"""Thin wrapper over the H1-2 arm forward kinematics.

Reuses fk_h12 from the IROS pipeline (h1-2_sensors package) so this repo does
not fork the URDF parser. Exposes only the 4-DOF chain we solve over in this
paper: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow. Wrist joints stay
at zero (protocol scope inherited from IROS).
"""

import os
import sys

import numpy as np

FK_PKG = os.path.expanduser("~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments")
if FK_PKG not in sys.path:
    sys.path.insert(0, FK_PKG)
from h12_experiments.fk_h12 import (  # noqa: E402
    UrdfChain, rpy_to_matrix, axis_angle_matrix, homogeneous, BASE_LINK,
    DEFAULT_TIP)

SIDES = ("left", "right")
ARM_SEGS_4 = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
DEFAULT_URDF = os.path.expanduser("~/mj_ws/assets/h1_2_description/h1_2.urdf")


class ArmChainFK:
    """FK returning anchor / elbow / tip positions for a 4-DOF arm.

    Mirrors the ArmChainFK in experiments/scripts/retarget_arm.py; kept here
    to keep icra2027 self-contained if the IROS repo is rearranged."""

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
        """Positions of anchor / elbow / tip in the torso frame."""
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

    def tip(self, q4):
        return self.positions(q4)[2]

    def tip_jacobian(self, q4, eps=1e-4):
        """3x4 tip Jacobian by finite differences (fine for the analyses here)."""
        base = self.tip(q4)
        J = np.zeros((3, 4))
        for k in range(4):
            qp = np.asarray(q4, dtype=float).copy()
            qp[k] += eps
            J[:, k] = (self.tip(qp) - base) / eps
        return J


def load_limits_4(urdf_path):
    """(lo, hi) arrays for the 4-DOF chain per side."""
    import xml.etree.ElementTree as ET
    raw = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        lim = j.find("limit")
        if j.get("type") in ("revolute", "continuous") and lim is not None:
            raw[j.get("name")] = (float(lim.get("lower", -np.pi)),
                                  float(lim.get("upper", np.pi)),
                                  float(lim.get("effort", 0.0)),
                                  float(lim.get("velocity", 0.0)))
    out = {}
    for side in SIDES:
        lo = np.array([raw[f"{side}_{s}_joint"][0] for s in ARM_SEGS_4])
        hi = np.array([raw[f"{side}_{s}_joint"][1] for s in ARM_SEGS_4])
        eff = np.array([raw[f"{side}_{s}_joint"][2] for s in ARM_SEGS_4])
        vel = np.array([raw[f"{side}_{s}_joint"][3] for s in ARM_SEGS_4])
        out[side] = dict(lo=lo, hi=hi, effort=eff, velocity=vel)
    return out
