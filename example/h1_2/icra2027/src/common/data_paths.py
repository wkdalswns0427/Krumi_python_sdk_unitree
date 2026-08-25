#!/usr/bin/env python3
"""Single source of truth for capture paths.

The 2026-08-25 sensor pivot moved capture from iPhone-on-tripod to a ZED
depth camera mounted at the robot neck (see notes/decisions.md). Until the
first ZED capture session lands, the scripts continue to read the OLD
iPhone captures at MOTIONS_DATA_ROOT for preview runs. When the ZED
captures arrive:

  1. Set ZED_DATA_ROOT below to the parent folder that will hold the
     per-capture subdirectories.
  2. Set ACTIVE_DATA_ROOT = ZED_DATA_ROOT.
  3. Confirm each capture subfolder contains a joints CSV in the same
     schema as MotionsDataset0713/*/b2g/iph_mono.csv (frame,joint,x,y,z,conf),
     or wire a ZED-specific reader that produces that CSV.

Every experiment script imports ACTIVE_DATA_ROOT from here so the switch is
a single line edit.
"""

import os

MOTIONS_DATA_ROOT = os.path.expanduser(
    "~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/iphone_data/"
    "MotionsDataset0713")

# TODO(zed refilm): set this to the parent folder for ZED captures once the
# first session lands. Suggestion: keep the sibling layout, e.g.
# ~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/zed_data/<session>/
ZED_DATA_ROOT = None

# The scripts use this. Point at ZED_DATA_ROOT once the switch is real.
ACTIVE_DATA_ROOT = MOTIONS_DATA_ROOT

# Per-capture joints-CSV location relative to a capture folder.
# For iPhone this was b2g/iph_mono.csv. If the ZED path produces a different
# filename (e.g. b2g/zed_mono.csv), update JOINTS_CSV_REL and the scripts
# pick it up automatically.
JOINTS_CSV_REL = os.path.join("b2g", "iph_mono.csv")


def joints_csv_for(capture_name):
    """Full path to the per-frame joints CSV for a capture."""
    return os.path.join(ACTIVE_DATA_ROOT, capture_name, JOINTS_CSV_REL)
