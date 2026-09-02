#!/usr/bin/env python3
"""trim_joints.py - drop the first and last seconds of a joints CSV.

Refilm captures begin with the subject walking to the mark and end with the walk
back to the keyboard. This drops those. Works on the joints CSV that
zed_to_joints.py writes. The SVO2 keeps the full recording, so trimming the CSV
is non-destructive to the raw data.

    trim_joints.py M2_1.csv --fps 60 --start 5 --end 5            # -> M2_1_trim.csv
    trim_joints.py M2_1.csv --fps 60 --start 5 --end 5 --inplace  # overwrite

--fps must match the capture. HD720 refilm is 60. It converts seconds to the
number of samples to drop from each end.
"""

import argparse
import csv
import os


def trim(rows, fps, start_s, end_s):
    """Return the rows whose frame is kept after dropping start_s at the head and
    end_s at the tail. Works in sample space, so the frame step does not matter."""
    frames = sorted({int(r["frame"]) for r in rows})
    n = len(frames)
    n_start = int(round(start_s * fps))
    n_end = int(round(end_s * fps))
    if n_start + n_end >= n:
        raise SystemExit(
            f"trim of {start_s:g}s + {end_s:g}s is {n_start + n_end} samples, "
            f"but the clip has only {n}. Nothing would be left.")
    keep = set(frames[n_start:n - n_end])
    return [r for r in rows if int(r["frame"]) in keep], n, len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--fps", type=float, required=True, help="capture frame rate")
    ap.add_argument("--start", type=float, default=5.0, help="seconds dropped at the head")
    ap.add_argument("--end", type=float, default=5.0, help="seconds dropped at the tail")
    ap.add_argument("--out", help="output CSV (default is <stem>_trim.csv)")
    ap.add_argument("--inplace", action="store_true", help="overwrite the input CSV")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit("empty CSV")
    kept, n_in, n_out = trim(rows, args.fps, args.start, args.end)

    out = args.csv if args.inplace else (
        args.out or os.path.splitext(args.csv)[0] + "_trim.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    print(f"[trim] {os.path.basename(args.csv)}: {n_in} to {n_out} samples "
          f"({n_in / args.fps:.1f}s to {n_out / args.fps:.1f}s), dropped "
          f"{args.start:g}s head and {args.end:g}s tail -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
