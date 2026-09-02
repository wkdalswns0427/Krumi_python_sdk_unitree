#!/usr/bin/env python3
"""Contact-event detection on the human wrist trajectory.

The task-centric objective is supposed to carry "task-relevant tolerance
including approach direction at contact events" (PLAN_v2 §Formulation). That
requires knowing WHICH frames are contact events, because constraining
approach direction everywhere reproduces exactly the problem the paper is
about: six residuals on a 4-DOF arm leaves no null space, so limit avoidance
gets nothing and the solve is back to pinning posture (see B3's h2o row).

So the gate is not a convenience, it is the mechanism. Approach direction is
constrained where the task needs it and free everywhere else.

Two sources, deliberately interchangeable:

  detect_contact_events()  automatic, from wrist speed minima. Defensible as
                           a heuristic, NOT as ground truth. Use for
                           development and for reporting a sensitivity band.
  load_contact_events()    manual annotation from a CSV. What the paper
                           should use once the ZED captures are annotated
                           (hammer strike for M1, block placement for M2,
                           ground contact for M3). See notes/open_questions.md
                           item 5.

Reporting discipline: whichever source is used must be named in the paper,
and if the automatic one is used the approach-error numbers must be reported
against both a tight and a loose window so the reader can see the
sensitivity.
"""

import csv
import os

import numpy as np


def wrist_speed(wrist_xyz, dt):
    """Per-frame speed of the wrist path, in metres per second.

    Endpoint frames repeat their neighbour so the output length matches the
    input, which keeps the gate array index-aligned with the solve loop.
    """
    p = np.asarray(wrist_xyz, dtype=float)
    if len(p) < 2:
        return np.zeros(len(p))
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    return np.concatenate([v[:1], v])


def detect_contact_events(wrist_xyz, dt, speed_percentile=25.0,
                          min_separation_s=0.30, smooth_frames=5):
    """Frame indices where the wrist is slow and locally slowest.

    A contact event in these motions (hammer strike, block placement, lift
    onset) shows up as the wrist decelerating into a local speed minimum. The
    detector takes local minima of a smoothed speed signal that also fall
    below `speed_percentile` of the rep's own speed distribution, then thins
    them so no two events sit closer than `min_separation_s`.

    Percentile rather than an absolute threshold because reps differ in
    overall pace, and the same person hammers faster on take three.

    Returns a sorted array of frame indices.
    """
    v = wrist_speed(wrist_xyz, dt)
    n = len(v)
    if n < 3:
        return np.array([], dtype=int)

    if smooth_frames > 1:
        k = np.ones(int(smooth_frames)) / float(smooth_frames)
        v = np.convolve(v, k, mode="same")

    thresh = np.percentile(v, speed_percentile)
    local_min = (v[1:-1] <= v[:-2]) & (v[1:-1] <= v[2:]) & (v[1:-1] <= thresh)
    idx = np.flatnonzero(local_min) + 1

    if len(idx) == 0:
        return idx
    min_sep = max(1, int(round(min_separation_s / dt)))
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] >= min_sep:
            keep.append(i)
        elif v[i] < v[keep[-1]]:
            keep[-1] = i          # keep the slower of two close candidates
    return np.array(keep, dtype=int)


def event_gate(n_frames, event_idx, dt, window_s=0.10):
    """Boolean array, True on frames within `window_s` of any contact event.

    The window exists because a contact is not one frame: the approach
    direction matters over the short interval where the hand is committing to
    the contact, not only at the instant of minimum speed.
    """
    gate = np.zeros(int(n_frames), dtype=bool)
    half = max(1, int(round(window_s / dt)))
    for i in np.asarray(event_idx, dtype=int):
        gate[max(0, i - half):min(n_frames, i + half + 1)] = True
    return gate


def load_contact_events(path, capture):
    """Manual annotations from a CSV with columns capture,frame.

    Returns a sorted array of frame indices for `capture`, or None if the
    file does not exist, so callers can fall back to the detector and say so.
    """
    if not path or not os.path.isfile(path):
        return None
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("capture") == capture:
                out.append(int(row["frame"]))
    return np.array(sorted(out), dtype=int)


def gate_for_capture(wrist_xyz, dt, capture=None, annotation_csv=None,
                     window_s=0.10, **detect_kw):
    """Contact gate for one rep, preferring annotation over detection.

    Returns (gate, source) where source is "annotated" or "detected", so the
    caller can report which one produced the numbers.
    """
    n = len(wrist_xyz)
    if capture is not None:
        idx = load_contact_events(annotation_csv, capture)
        if idx is not None and len(idx):
            return event_gate(n, idx, dt, window_s), "annotated"
    idx = detect_contact_events(wrist_xyz, dt, **detect_kw)
    return event_gate(n, idx, dt, window_s), "detected"
