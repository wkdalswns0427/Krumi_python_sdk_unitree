#!/usr/bin/env python3
"""rula_reba_agreement.py - RULA/REBA risk-category agreement, mono vs RGBD.

Block 4 of the Humanoids 2026 protocol. Takes two per-frame ergonomic score
files (monocular pipeline, RGBD reference), bands each frame's grand score
into a discrete risk category with identical code, and reports how often the
two pipelines assign the SAME category.

Purpose: is monocular capture sufficient for ergonomic screening?

Input CSV (one per pipeline), any of:
    frame,rula_score,reba_score          # grand scores -> banded here
    frame,rula_category,reba_category    # categories given -> used directly
    (a mix; RULA and REBA handled independently)

Usage:
    python3 rula_reba_agreement.py MONO.csv RGBD.csv --out-prefix agree_M1
    python3 rula_reba_agreement.py --demo     # self-test on synthetic data

Reports, per metric (RULA, REBA): exact category agreement %, Cohen's kappa,
and a confusion matrix (printed; CSV with --out-prefix).
"""

import argparse
import csv
import os
import sys
import tempfile

import numpy as np


# Grand-score -> (action-level category, name). Documented in README.
RULA_BANDS = [(2, 1, "Acceptable"), (4, 2, "Investigate"),
              (6, 3, "Change soon"), (99, 4, "Change now")]
REBA_BANDS = [(1, 0, "Negligible"), (3, 1, "Low"), (7, 2, "Medium"),
              (10, 3, "High"), (99, 4, "Very high")]

BANDS = {"rula": RULA_BANDS, "reba": REBA_BANDS}
CATEGORIES = {"rula": [1, 2, 3, 4], "reba": [0, 1, 2, 3, 4]}


def band(metric, score):
    for hi, cat, _ in BANDS[metric]:
        if score <= hi:
            return cat
    return BANDS[metric][-1][1]


def cat_name(metric, cat):
    for _, c, name in BANDS[metric]:
        if c == cat:
            return name
    return "?"


# ── Loading ──────────────────────────────────────────────────────────────────
def load_categories(path, metric):
    """{frame_index: category} for one metric from one file."""
    score_col, cat_col = f"{metric}_score", f"{metric}_category"
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "frame" not in cols:
            sys.exit(f"{path}: missing 'frame' column")
        if cat_col not in cols and score_col not in cols:
            sys.exit(f"{path}: need '{cat_col}' or '{score_col}' column")
        use_cat = cat_col in cols
        for row in reader:
            val = row.get(cat_col) if use_cat else row.get(score_col)
            if val is None or val == "":
                continue
            fi = int(float(row["frame"]))
            out[fi] = int(float(val)) if use_cat else band(metric, float(val))
    return out


# ── Agreement ────────────────────────────────────────────────────────────────
def confusion(mono, rgbd, categories):
    """Confusion matrix rows=mono, cols=rgbd, over common frames."""
    idx = {c: i for i, c in enumerate(categories)}
    n = len(categories)
    mat = np.zeros((n, n), dtype=int)
    common = sorted(set(mono) & set(rgbd))
    for fi in common:
        a, b = mono[fi], rgbd[fi]
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1
    return mat, len(common)


def cohen_kappa(mat):
    total = mat.sum()
    if total == 0:
        return float("nan")
    po = np.trace(mat) / total
    row = mat.sum(axis=1) / total
    col = mat.sum(axis=0) / total
    pe = float(np.sum(row * col))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def analyze(metric, mono_path, rgbd_path):
    mono = load_categories(mono_path, metric)
    rgbd = load_categories(rgbd_path, metric)
    cats = CATEGORIES[metric]
    mat, n_common = confusion(mono, rgbd, cats)
    total = int(mat.sum())
    agree = int(np.trace(mat))
    pct = (agree / total * 100.0) if total else float("nan")
    return dict(metric=metric, matrix=mat, cats=cats, n_common=n_common,
                total=total, agree=agree, pct=pct, kappa=cohen_kappa(mat))


# ── Output ───────────────────────────────────────────────────────────────────
def print_result(res):
    m = res["metric"].upper()
    cats = res["cats"]
    print("=" * 66)
    print(f"{m} category agreement (monocular vs RGBD)")
    print("=" * 66)
    print(f"  frames compared : {res['total']}")
    print(f"  exact agreement : {res['agree']}/{res['total']}  "
          f"= {res['pct']:.1f}%")
    print(f"  Cohen's kappa   : {res['kappa']:.3f}")
    print("-" * 66)
    print("  confusion matrix (rows = monocular, cols = RGBD)")
    header = " " * 14 + "".join(f"{c:>8}" for c in cats) + f"{'sum':>8}"
    print("  " + header)
    mat = res["matrix"]
    for i, c in enumerate(cats):
        label = f"{c}:{cat_name(res['metric'], c)[:9]}"
        row = "".join(f"{v:>8}" for v in mat[i])
        print(f"  {label:<14}{row}{mat[i].sum():>8}")
    colsum = "".join(f"{v:>8}" for v in mat.sum(axis=0))
    print(f"  {'sum':<14}{colsum}{mat.sum():>8}")
    print("=" * 66)


def write_confusion_csv(res, path):
    cats = res["cats"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mono\\rgbd"] + [f"cat_{c}" for c in cats])
        for i, c in enumerate(cats):
            w.writerow([f"cat_{c}"] + list(res["matrix"][i]))


def write_summary_csv(results, path, tag):
    fields = ["metric", "frames", "agreement_pct", "kappa"]
    if tag is not None:
        fields = ["tag"] + fields
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for res in results:
            row = [res["metric"], res["total"], f"{res['pct']:.2f}",
                   f"{res['kappa']:.4f}"]
            if tag is not None:
                row = [tag] + row
            w.writerow(row)


# ── Demo / self-test ─────────────────────────────────────────────────────────
def _write_demo_pair(dirpath):
    rng = np.random.default_rng(1)
    mono_p = os.path.join(dirpath, "demo_mono_scores.csv")
    rgbd_p = os.path.join(dirpath, "demo_rgbd_scores.csv")
    with open(mono_p, "w", newline="") as fm, open(rgbd_p, "w", newline="") as fg:
        wm, wg = csv.writer(fm), csv.writer(fg)
        wm.writerow(["frame", "rula_score", "reba_score"])
        wg.writerow(["frame", "rula_score", "reba_score"])
        for fi in range(300):
            rula = int(np.clip(rng.integers(1, 8), 1, 7))
            reba = int(np.clip(rng.integers(1, 12), 1, 15))
            wg.writerow([fi, rula, reba])
            # mono agrees mostly, jitters by +/-1 sometimes.
            dr = rng.choice([0, 0, 0, 0, 1, -1])
            db = rng.choice([0, 0, 0, 1, -1, 2])
            wm.writerow([fi, int(np.clip(rula + dr, 1, 7)),
                         int(np.clip(reba + db, 1, 15))])
    return mono_p, rgbd_p


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mono_file", nargs="?", help="monocular pipeline scores")
    p.add_argument("rgbd_file", nargs="?", help="RGBD reference scores")
    p.add_argument("--out-prefix", help="write <prefix>_summary.csv and "
                                        "<prefix>_{rula,reba}_confusion.csv")
    p.add_argument("--tag", help="label added to the summary CSV rows")
    p.add_argument("--metrics", default="rula,reba",
                   help="comma list of metrics to compare (default: rula,reba)")
    p.add_argument("--demo", action="store_true",
                   help="run on generated synthetic data")
    return p.parse_args()


def main():
    args = parse_args()

    tmp = None
    if args.demo:
        tmp = tempfile.mkdtemp(prefix="block4_demo_")
        args.mono_file, args.rgbd_file = _write_demo_pair(tmp)
        print(f"[demo] synthetic pair in {tmp}\n")
    elif not (args.mono_file and args.rgbd_file):
        sys.exit("error: provide MONO and RGBD files (or --demo)")

    for pth in (args.mono_file, args.rgbd_file):
        if not os.path.isfile(pth):
            sys.exit(f"error: file not found: {pth}")

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    for m in metrics:
        if m not in BANDS:
            sys.exit(f"error: unknown metric '{m}' (known: {list(BANDS)})")

    results = []
    for m in metrics:
        res = analyze(m, args.mono_file, args.rgbd_file)
        print_result(res)
        results.append(res)

    prefix = args.out_prefix
    if prefix is None and args.demo:
        prefix = os.path.join(tmp, "agreement")
    if prefix:
        write_summary_csv(results, f"{prefix}_summary.csv", args.tag)
        for res in results:
            write_confusion_csv(res, f"{prefix}_{res['metric']}_confusion.csv")
        print(f"\nWrote {prefix}_summary.csv and per-metric confusion CSVs")


if __name__ == "__main__":
    main()
