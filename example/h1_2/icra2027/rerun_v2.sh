#!/usr/bin/env bash
# Full re-analysis after two corrections:
#   1. human_directions.per_frame_inputs now applies the teleport gate and the
#      One-Euro direction filter that the IROS retargeter applies in its main
#      loop. Without them the solver saw raw per-frame directions and the
#      shoulder-yaw branch flipped constantly, which inflated flip counts and
#      limit contact for every fidelity condition.
#   2. Both arms are analysed. The earlier right-arm-only choice was inherited
#      from the IROS flip case and is not defensible for the bimanual motions
#      M2, M3 and M4, where the left arm does 54 to 77 percent of the work.
#
# Results are written per side into results/<exp>/v2_<side>.json.
#
#   ./rerun_v2.sh              # both arms, all 25 captures
#   ./rerun_v2.sh right        # one side only
set -uo pipefail
cd "$(dirname "$0")"
PY=/usr/bin/python3
LOG=results/_rerun_v2
mkdir -p "$LOG"

SIDES=("$@")
if [ ${#SIDES[@]} -eq 0 ]; then SIDES=(right left); fi

CAPS=()
for m in 1 2 3 4 5; do for r in 1 2 3 4 5; do CAPS+=("M${m}_${r}"); done; done
RS=()
for d in 1 2 3 4; do for r in 1 2 3 4 5; do RS+=("RS_d${d}_${r}"); done; done

echo "sides: ${SIDES[*]}"
echo "captures: ${#CAPS[@]} motion, ${#RS[@]} sweep"

run () {  # run <logname> <command...>
    local name=$1; shift
    echo "--- $name ---"
    if "$@" > "$LOG/$name.log" 2>&1; then
        tail -n 3 "$LOG/$name.log"
    else
        echo "  FAILED, see $LOG/$name.log"; tail -n 6 "$LOG/$name.log"
    fi
}

for S in "${SIDES[@]}"; do
  echo "================ SIDE: $S ================"
  run "x6_$S"  $PY src/x6_taskcentric.py    --side "$S" --captures "${CAPS[@]}" \
        --out results/x6/v2_$S.json --frames-dir results/x6/v2_$S
  run "x2_$S"  $PY src/x2_costoffidelity.py --side "$S" --captures "${CAPS[@]}" \
        --out results/x2/v2_$S.json
  run "b2_$S"  $PY src/b2_control.py        --side "$S" --captures "${CAPS[@]}" \
        --out results/b2/v2_$S.json
  run "b3_$S"  $PY src/b3_published_baselines.py --side "$S" --captures "${CAPS[@]}" \
        --out results/b3/v2_$S.json
  run "x8_$S"  $PY results/b3/factorial_2x2.py --side "$S" --captures "${CAPS[@]}" \
        --out results/b3/factorial_v2_$S.json
  run "x10_$S" $PY src/x10_gated_approach.py --side "$S" --captures "${CAPS[@]}" \
        --out results/x10/v2_$S.json
  run "x9_$S"  $PY src/reach_extent_sweep.py --side "$S" --captures "${RS[@]}" \
        --out results/reach_extent/v2_$S.json
done

echo
echo "done. logs in $LOG/"
