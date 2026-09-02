#!/usr/bin/env bash
# Re-run every data-dependent analysis after a new capture session.
#
# Everything here reads its captures through src/common/data_paths.py, so the
# ONLY thing to change after filming is ACTIVE_DATA_ROOT in that file. Nothing
# in src/common/ needs re-running: the solvers, metrics and contact detector
# are method code and survive the refilm untouched.
#
# Usage:
#   ./rerun_all.sh                       # default capture names
#   ./rerun_all.sh M1_R_1 M2_1 M3_1      # explicit subset
#
# Runtime is roughly 45 to 70 minutes unattended. Logs land in results/_rerun/.

set -uo pipefail
cd "$(dirname "$0")"
PY=/usr/bin/python3
LOG=results/_rerun
mkdir -p "$LOG"

CAPS=("$@")
if [ ${#CAPS[@]} -eq 0 ]; then
    CAPS=(M2_1 M2_2 M2_3 M3_1 M3_2 M3_3)
fi
echo "captures: ${CAPS[*]}"

# Fail fast rather than burning an hour on a bad path.
$PY - "${CAPS[@]}" <<'EOF' || exit 1
import sys, os
sys.path.insert(0, "src")
from common.data_paths import ACTIVE_DATA_ROOT, joints_csv_for
print(f"[check] ACTIVE_DATA_ROOT = {ACTIVE_DATA_ROOT}")
missing = [c for c in sys.argv[1:] if not os.path.isfile(joints_csv_for(c))]
if missing:
    print(f"[check] FAIL: no joints CSV for {missing}")
    print("[check] set ZED_DATA_ROOT and ACTIVE_DATA_ROOT in "
          "src/common/data_paths.py, and JOINTS_CSV_REL if the filename changed")
    sys.exit(1)
print(f"[check] all {len(sys.argv)-1} joints CSVs found")
EOF

run () {  # run <name> <command...>
    local name=$1; shift
    echo "--- $name ---"
    if "$@" > "$LOG/$name.log" 2>&1; then
        tail -n 4 "$LOG/$name.log"
    else
        echo "  FAILED, see $LOG/$name.log"
        tail -n 8 "$LOG/$name.log"
    fi
}

# C1 evidence
run x6   $PY src/x6_taskcentric.py    --side right --captures "${CAPS[@]}" \
             --out results/x6/zed.json --frames-dir results/x6/zed
run x2   $PY src/x2_costoffidelity.py --side right --captures "${CAPS[@]}" \
             --out results/x2/zed.json --frames-dir results/x2/zed
run x8   $PY results/b3/factorial_2x2.py

# C2 evidence
run b2   $PY src/b2_control.py            --side right --captures "${CAPS[@]}" \
             --out results/b2/zed.json --frames-dir results/b2/zed
run b3   $PY src/b3_published_baselines.py --side right --captures "${CAPS[@]}" \
             --out results/b3/zed.json

# C3 evidence
run x10  $PY src/x10_gated_approach.py --side right --captures "${CAPS[@]}" \
             --out results/x10/zed.json

# supporting
run strawman $PY results/strawman_check/regularized_baseline.py
run x1       $PY src/x1_generated_across_motions.py

echo
echo "done. logs in $LOG/"
echo "NOT run automatically, they need decisions or hardware:"
echo "  X9  reach-extent sweep   src/reach_extent_sweep.py  (skeleton, needs the sweep captures)"
echo "  X3  hardware replay      src/x3_hardware_replay.py  (skeleton, needs the robot + e-stop)"
echo "  X7  dual mode            src/x7_dual_mode.py        (skeleton, pressure valve #2)"
