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

ZED_SESSIONS_ROOT = os.path.expanduser(
    "~/mj_ws/Krumi_python_sdk_unitree/example/h1_2/experiments/zed_data")

# One folder per subject session. Add a line when a session lands.
SESSIONS = {
    "S1": "S1_2026-08-29",
    "S1refilm": "S1_refilm_2026-09-02",
    "S1sweep": "S1_sweep_frontal",
    "S2": "S2_2026-09-02",
    "S3": "S3_2026-09-02",
}
DEFAULT_SESSION = "S1"

# Select a session without editing this file:  ICRA_SESSION=S2 python3 ...
# An unknown value is a typo, and failing loudly beats silently analysing the
# wrong subject.
_sel = os.environ.get("ICRA_SESSION", DEFAULT_SESSION).strip()
if _sel not in SESSIONS:
    raise SystemExit(
        f"ICRA_SESSION={_sel!r} is not one of {sorted(SESSIONS)}. "
        f"Add it to SESSIONS in {__file__} if the session is new.")

ACTIVE_SESSION = _sel
ZED_DATA_ROOT = os.path.join(ZED_SESSIONS_ROOT, SESSIONS[_sel])

# The scripts use this.
ACTIVE_DATA_ROOT = ZED_DATA_ROOT

# Arm length, and the reason there are two columns.
#
# The skeleton normalization step and every extension ratio divide by "the arm
# length", and the quantity they need is SHOULDER TO WRIST, because that is
# where the body tracker places its wrist keypoint. The tape measurements taken
# so far were made shoulder to the BACK OF THE HAND, which is longer by roughly
# a hand.
#
# The gap is not small. S1 measures 0.5588 m at the wrist and 0.6096 m at the
# hand, so the constant used until 2026-09-01 was 9.1 percent too large.
# Dividing by a value that is too large pushes every extension ratio down, so
# the reported threshold moves. Note also that the tracker under-reports arm
# length by a further 4 cm, giving 0.515 m median for a straight arm against a
# true 0.5588 m. Trust the tape, not the tracker.
#
# Both columns are kept so the convention is never ambiguous again. Analysis
# uses `wrist`. `hand` is recorded only so the two can be reconciled.
ARM_LENGTH_M = {
    #        wrist,          hand,    note
    "S1": dict(wrist=0.5588, hand=0.6096, note="22 in wrist, 24 in hand"),
    "S1refilm": dict(wrist=0.5588, hand=0.6096, note="same subject as S1, sweep only"),
    "S1sweep": dict(wrist=0.5588, hand=0.6096, note="S1 frontal sweep, both sessions, d1-d6"),
    "S2": dict(wrist=0.4953, hand=0.5334, note="19.5 in wrist, 21 in hand"),
    "S3": dict(wrist=0.5715, hand=0.6350, note="22.5 in wrist, 25 in hand"),
}

# Everything analysed before 2026-09-01 divided by 0.6096, the hand measurement,
# which is 9.1 percent above S1's true 0.5588 m. Every extension ratio produced
# then is low by that factor and has to be recomputed, not rescaled in place,
# because the same constant also set the normalization clamp.
SUPERSEDED_ARM_M = 0.6096


def arm_length(session=None, convention="wrist"):
    """Measured arm length for a session, in metres.

    convention="wrist" is what the analysis needs. Asking for it before it has
    been measured raises, rather than silently falling back to the hand
    measurement, because that substitution is the error this table exists to
    prevent."""
    s = session or ACTIVE_SESSION
    rec = ARM_LENGTH_M.get(s)
    if rec is None:
        raise SystemExit(f"no arm-length record for session {s!r}.")
    L = rec.get(convention)
    if L is None:
        raise SystemExit(
            f"{convention} arm length for {s} is not recorded ({rec['note']}). "
            f"Measure shoulder joint centre to the wrist crease, then set "
            f"ARM_LENGTH_M[{s!r}]['{convention}'] in {__file__}. Do not "
            f"substitute the hand measurement, which is longer by about a hand.")
    return L

# ZED captures are flat CSVs at ACTIVE_DATA_ROOT/<capture>.csv, not the iPhone
# per-capture b2g/ layout. Set False to fall back to the iPhone layout.
FLAT_LAYOUT = True

# Per-capture joints-CSV location relative to a capture folder.
# For iPhone this was b2g/iph_mono.csv. If the ZED path produces a different
# filename (e.g. b2g/zed_mono.csv), update JOINTS_CSV_REL and the scripts
# pick it up automatically.
JOINTS_CSV_REL = os.path.join("b2g", "iph_mono.csv")


def joints_csv_for(capture_name):
    """Full path to the per-frame joints CSV for a capture."""
    if FLAT_LAYOUT:
        return os.path.join(ACTIVE_DATA_ROOT, capture_name + ".csv")
    return os.path.join(ACTIVE_DATA_ROOT, capture_name, JOINTS_CSV_REL)
