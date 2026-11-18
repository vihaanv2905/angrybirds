#!/bin/bash
#SBATCH --job-name=ab-record
#SBATCH --account=engin1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#
# Record trained policies playing every starter level on the SO-101, offscreen.
#   sbatch slurm/record.sbatch a1_starter_1bird a2_starter_3birds
# Writes docs/img/<run prefix>_<level>.mp4 (prefix = text before the first "_").
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/common.sh
source .venv/bin/activate
module load ffmpeg/6.0.0 2>/dev/null || module load ffmpeg 2>/dev/null || true
command -v ffmpeg >/dev/null || { echo "no ffmpeg"; exit 1; }

# CPU nodes have no GPU: use EGL if it works, else software OSMesa.
for gl in egl osmesa; do
    if MUJOCO_GL=$gl python -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco/>'); mujoco.Renderer(m, 64, 64)" 2>/dev/null; then
        export MUJOCO_GL=$gl; break
    fi
done
echo "MUJOCO_GL=${MUJOCO_GL:-none}"
[ -n "${MUJOCO_GL:-}" ] || exit 1

LEVELS="L1_open_pig L2_wood_shelter L3_reel_tower L4_fortress"
jobs=()
for run in "$@"; do
    for lvl in $LEVELS; do jobs+=("$run $lvl"); done
done
printf '%s\n' "${jobs[@]}" | xargs -P "${SLURM_CPUS_PER_TASK:-4}" -L 1 bash -c '
    out="docs/img/${0%%_*}_$1.mp4"
    python scripts/record_cosim.py --policy "runs/$0" --level "$1" --out "$out" > "logs/record_${0}_$1.log" 2>&1 \
        && echo "ok   $out" || { echo "FAIL $0 $1"; tail -5 "logs/record_${0}_$1.log"; }'
