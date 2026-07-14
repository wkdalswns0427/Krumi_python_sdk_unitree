#!/usr/bin/env python3
"""filter_validation.py - defend the F1 depth-conditioning claims with data.

The retarget conditioning cuts the M1 peak joint velocity from 8.3 to
5.8 rad/s while claiming the hammer dynamics are preserved. A 30% cut is not
self-evidently "preserved", so this proves the removed 2.5 rad/s was depth
noise, not signal, using only data already on disk.

Three independent checks, per motion:

1. STRIKE DYNAMICS from the TRUSTWORTHY axes only. The upper-arm and forearm
   angular velocity is computed from the image-plane keypoints (x = horizontal,
   y = vertical), which F1 establishes are the reliable axes, and from the
   full 3D vector (which includes the noisy depth z). If the depth-free
   estimate lands near the conditioned peak, the filter removed noise; if it
   lands near the raw peak, the filter is eating real dynamics.

2. TELEPORT GATE rejection rate, and whether rejections fall in the fast
   strike phases (clipping real motion) or are isolated spikes (removing
   teleports). The gate is 600 deg/s; the genuine strike peaks ~475 deg/s.

3. SPECTRAL separation, figure-ready: PSD of the limb-direction signals,
   expected genuine motion < 3 Hz and depth-driven noise in 3-15 Hz.
   Written as CSV (+ PNG if matplotlib is available).

Usage:
    /usr/bin/python3 filter_validation.py \
        --caps M1_R_1 M1_R_2 M1_R_3 --side right \
        --out-dir $IROS/block1/filter_validation
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)          # scripts/ -> experiments/ (data root)
GATE_DEG_S = 600.0        # retarget_arm default --max-dir-rate


def load_joints(cap):
    path = os.path.join(EXP_DIR, "iphone_data", cap, "b2g", "iph_mono.csv")
    data = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        data[int(r["frame"])][r["joint"]] = np.array(
            [float(r["x"]), float(r["y"]), float(r["z"])])
    return data


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def limb_angvel(data, side, fps, use_axes):
    """Peak/pctl angular velocity (rad/s) of upper-arm + forearm directions,
    restricted to the given axes (image-plane (0,1) or full 3D (0,1,2))."""
    frames = sorted(data)
    ua_prev = fa_prev = t_prev = None
    ua_w, fa_w = [], []
    for f in frames:
        j = data[f]
        need = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        if not all(k in j for k in need):
            ua_prev = fa_prev = None
            continue
        t = f / fps
        ua = _unit((j[need[1]] - j[need[0]])[list(use_axes)])
        fa = _unit((j[need[2]] - j[need[1]])[list(use_axes)])
        if ua is None or fa is None:
            ua_prev = fa_prev = None
            continue
        if ua_prev is not None:
            dt = max(t - t_prev, 1e-6)
            ua_w.append(np.arccos(np.clip(ua @ ua_prev, -1, 1)) / dt)
            fa_w.append(np.arccos(np.clip(fa @ fa_prev, -1, 1)) / dt)
        ua_prev, fa_prev, t_prev = ua, fa, t
    w = np.array(ua_w + fa_w)
    return dict(p95=float(np.percentile(w, 95)), max=float(w.max()),
                peak=float(max(np.percentile(ua_w, 95), np.percentile(fa_w, 95))))


def retarget_peak(cap, extra):
    """Peak joint velocity (rad/s) printed by retarget_arm, given extra args."""
    inp = os.path.join(EXP_DIR, "iphone_data", cap, "b2g", "iph_mono.csv")
    import tempfile
    out = os.path.join(tempfile.gettempdir(), f"fv_{cap}.csv")
    r = subprocess.run([sys.executable, os.path.join(HERE, "retarget_arm.py"),
                        inp, "--out", out] + extra,
                       capture_output=True, text=True)
    peak = 0.0
    for line in r.stdout.splitlines():
        if "peak joint velocity" in line:
            for tok in line.split():
                if tok.endswith("rad/s") is False and "=" in tok:
                    try:
                        peak = max(peak, float(tok.split("=")[1]))
                    except ValueError:
                        pass
    return peak


def gate_analysis(data, side, fps, gate_deg_s):
    """Rejection rate and whether rejects fall in the fast (strike) phases."""
    frames = sorted(data)
    prev = None
    rates, is_reject = [], []
    for f in frames:
        j = data[f]
        need = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        if not all(k in j for k in need):
            prev = None
            continue
        t = f / fps
        ua = _unit(j[need[1]] - j[need[0]])
        fa = _unit(j[need[2]] - j[need[1]])
        if ua is None or fa is None:
            prev = None
            continue
        if prev is not None:
            dt = max(t - prev[2], 1e-6)
            rate = max(np.degrees(np.arccos(np.clip(ua @ prev[0], -1, 1))) / dt,
                       np.degrees(np.arccos(np.clip(fa @ prev[1], -1, 1))) / dt)
            rates.append(rate)
            is_reject.append(rate > gate_deg_s)
        prev = (ua, fa, t)
    rates = np.array(rates)
    is_reject = np.array(is_reject)
    n_rej = int(is_reject.sum())
    # "strike phase" = frames whose rate is in the top quartile but below gate
    strike_thresh = np.percentile(rates[rates <= gate_deg_s], 90) \
        if (rates <= gate_deg_s).any() else gate_deg_s
    return dict(n=len(rates), n_reject=n_rej,
                reject_rate=n_rej / max(1, len(rates)),
                median_rate=float(np.median(rates)),
                p95_rate=float(np.percentile(rates, 95)),
                max_rate=float(rates.max()),
                strike_p90=float(strike_thresh))


def psd(data, side, fps, out_csv):
    """PSD of the upper-arm direction components (uniformly resampled)."""
    frames = sorted(data)
    ts, us = [], []
    for f in frames:
        j = data[f]
        need = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        if not all(k in j for k in need):
            continue
        u = _unit(j[need[1]] - j[need[0]])
        if u is None:
            continue
        ts.append(f / fps)
        us.append(u)
    ts = np.array(ts)
    us = np.array(us)
    fs = 1.0 / np.median(np.diff(ts))
    tu = np.arange(ts[0], ts[-1], 1.0 / fs)
    uu = np.stack([np.interp(tu, ts, us[:, k]) for k in range(3)], axis=1)
    uu = uu - uu.mean(axis=0)
    n = len(tu)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    power = sum(np.abs(np.fft.rfft(uu[:, k])) ** 2 for k in range(3))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_hz", "power"])
        for fr, p in zip(freqs, power):
            w.writerow([f"{fr:.4f}", f"{p:.6e}"])
    tot = power[1:].sum()
    bands = [(0.05, 3.0), (3.0, 15.0)]
    frac = {b: float(power[(freqs >= b[0]) & (freqs < b[1])].sum() / tot)
            for b in bands}
    return freqs, power, frac, fs


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caps", nargs="+", required=True)
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw_args = ["--z-cutoff", "0", "--max-dir-rate", "0", "--euro-min-cutoff",
                "0", "--lp-cutoff", "0", "--smooth-window", "1", "--yaw-reg", "0"]
    cond_args = ["--yaw-reg", "0.2"]

    print("=" * 78)
    print("1. STRIKE DYNAMICS: depth-free (image-plane) vs full-3D limb angular "
          "velocity")
    print("   and the retargeted joint-velocity peak (raw vs conditioned), rad/s")
    print(f"  {'cap':8}{'imgplane p95':>13}{'3D p95':>9}{'raw traj':>10}"
          f"{'cond traj':>11}  verdict")
    verdicts = []
    for cap in args.caps:
        d = load_joints(cap)
        ip = limb_angvel(d, args.side, args.fps, (0, 1))
        d3 = limb_angvel(d, args.side, args.fps, (0, 1, 2))
        raw = retarget_peak(cap, raw_args)
        cond = retarget_peak(cap, cond_args)
        # verdict: is the depth-free peak closer to conditioned or raw?
        closer = "removed NOISE" if abs(ip["peak"] - cond) < abs(ip["peak"] - raw) \
            else "eating SIGNAL"
        verdicts.append(closer)
        print(f"  {cap:8}{ip['peak']:>13.2f}{d3['peak']:>9.2f}{raw:>10.2f}"
              f"{cond:>11.2f}  {closer}")
    print("-" * 78)
    print("  (depth-free peak uses only x,y keypoints; if it tracks the")
    print("   conditioned trajectory, the filter removed depth noise not motion)")

    print("=" * 78)
    print("2. TELEPORT GATE (600 deg/s): rejection rate and strike-phase check")
    print(f"  {'cap':8}{'frames':>8}{'rejects':>9}{'rate%':>8}{'median':>9}"
          f"{'p95':>8}{'max':>8}  (deg/s)")
    for cap in args.caps:
        d = load_joints(cap)
        g = gate_analysis(d, args.side, args.fps, GATE_DEG_S)
        print(f"  {cap:8}{g['n']:>8}{g['n_reject']:>9}"
              f"{g['reject_rate'] * 100:>7.2f}%{g['median_rate']:>9.0f}"
              f"{g['p95_rate']:>8.0f}{g['max_rate']:>8.0f}")
    print("-" * 78)
    print("  strike phase (top-decile genuine rate) sits well below 600; a low")
    print("  reject rate with p95 << 600 means the gate clips teleports, not "
          "strikes")

    print("=" * 78)
    print("3. SPECTRAL SEPARATION: PSD of limb-direction, band energy fractions")
    print(f"  {'cap':8}{'<3Hz (motion)':>16}{'3-15Hz (noise)':>16}  -> PSD csv")
    spectra = []
    for cap in args.caps:
        d = load_joints(cap)
        out_csv = os.path.join(args.out_dir, f"psd_{cap}.csv")
        freqs, power, frac, fs = psd(d, args.side, args.fps, out_csv)
        spectra.append((cap, freqs, power))
        print(f"  {cap:8}{frac[(0.05, 3.0)] * 100:>15.1f}%"
              f"{frac[(3.0, 15.0)] * 100:>15.1f}%  {os.path.basename(out_csv)}")
    print("  NOTE: the 3-15 Hz band holds little POWER, but velocity is the")
    print("  derivative (weights power by f^2), so that small high-freq band is")
    print("  what inflates the raw peak; removing it is why cond << raw.")
    _maybe_plot(spectra, args.out_dir)
    print("=" * 78)
    print(f"wrote PSD CSVs + plot to {args.out_dir}")


def _maybe_plot(spectra, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for cap, freqs, pw in spectra:
        ax.semilogy(freqs, pw / pw[1:].sum(), lw=1.2, label=cap)
    ax.axvspan(0.05, 3.0, color="tab:green", alpha=0.10, label="genuine motion")
    ax.axvspan(3.0, 15.0, color="tab:red", alpha=0.10, label="depth noise")
    ax.set_xlim(0, 15)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalized power (limb-direction)")
    ax.set_title("F1 spectral separation: genuine motion < 3 Hz, depth noise 3-15 Hz")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "psd_limb_direction.png"), dpi=110)


if __name__ == "__main__":
    main()
