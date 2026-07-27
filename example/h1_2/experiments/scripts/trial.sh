#!/usr/bin/env bash
# trial.sh - one Block 2 trial, bookkeeping-safe. Run in the LOGGER terminal
# (conda rical_unitree). It resolves the trajectory, refuses to overwrite an
# existing log, starts the logger with a clean M<m>_R<rep>_T<trial> name, and
# prints the exact replay command to paste in the OTHER terminal.
#
#   Usage:  ./trial.sh <motion 1|2|3> <rep 1|2|3> <trial N>
#   e.g.    ./trial.sh 2 1 3      # motion M2, reference rep 1, trial 3
#
# Then: paste the printed replay command in terminal B, run it, and when the
# replay prints "overlay released", press Ctrl+C here to stop the logger.

set -euo pipefail
M=${1:?motion (1|2|3)}; R=${2:?rep (1|2|3)}; T=${3:?trial number}
IROS=${IROS:-$HOME/mj_ws/h1-2_sensors/experiments/iros2026ws}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IFACE=${IFACE:-enp128s31f6}

# motion 1 folders are M1_R_<rep>; motions 2,3 are M<m>_<rep>
if [ "$M" = "1" ]; then CAP="M1_R_${R}"; else CAP="M${M}_${R}"; fi
# prep/tail trim, calibrated per motion (see replay_arm --task-thresh help).
# M1_R_1 is a manual clip: its trajectory STARTS with a half-raised arm
# (interp artifact), which fools the z signal; true task window is 5.8-37.3s.
case "$M" in
  1) TASKARGS="--task-signal z --task-thresh 0.40" ;;
  2) TASKARGS="--task-signal x --task-thresh 0.30" ;;
  3) TASKARGS="--task-signal x --task-thresh 0.22" ;;
esac
if [ "$M" = "1" ] && [ "$R" = "1" ]; then TASKARGS="--clip 5.8:37.3"; fi
TRAJ="${HERE}/../iphone_data/${CAP}/b2g/traj_iph_mono.csv"
LOG="${IROS}/block2/replay_M${M}_R${R}_T${T}.csv"

[ -f "$TRAJ" ] || { echo "trajectory not found: $TRAJ" >&2; exit 1; }
[ -e "$LOG" ]  && { echo "log already exists: $LOG  (bump the trial number)" >&2; exit 1; }

echo "=================================================================="
echo " Trial  M${M} (rep ${R})  T${T}   ->  $(basename "$LOG")"
echo " Terminal B (conda rical_unitree), paste this, then ENTER through it:"
echo
echo "   python3 replay_arm.py ${TRAJ} \\"
echo "       --interface ${IFACE} --speed-scale 0.5 --gravity-ff ${TASKARGS}"
echo
echo " Watch for the 'safe-start trim' line (fix is live) and the arm"
echo " lifting on fade-in. Ctrl+C here after 'overlay released'."
echo "=================================================================="

# logger runs until Ctrl+C (--duration 0); aborts if no rt/lowstate in 10 s
exec python3 "${HERE}/sdk_replay_logger.py" --interface "${IFACE}" \
    --motion "$M" --rep "$R" --trial "$T" --duration 0 --out-dir "${IROS}/block2"
