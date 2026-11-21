#!/bin/bash
#SBATCH --job-name=ab-ghosts
#SBATCH --account=engin1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# "Ghost race" demo: every training run's attempts overlaid on each level, the
# best one fired by the SO-101 (scripts/ghost_demo.py). One process per level,
# then everything is joined into docs/img/ghosts_all.mp4.
#   sbatch slurm/ghosts.sbatch                 # default levels
#   LEVELS="L3_reel_tower" sbatch slurm/ghosts.sbatch --samples 12
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/common.sh
source .venv/bin/activate
module load ffmpeg/6.0.0 2>/dev/null || module load ffmpeg 2>/dev/null || true
command -v ffmpeg >/dev/null || { echo "no ffmpeg"; exit 1; }
for gl in egl osmesa; do
    if MUJOCO_GL=$gl python -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco/>'); mujoco.Renderer(m, 64, 64)" 2>/dev/null; then
        export MUJOCO_GL=$gl; break
    fi
done
echo "MUJOCO_GL=${MUJOCO_GL:-none}"
[ -n "${MUJOCO_GL:-}" ] || exit 1

LEVELS=${LEVELS:-"L1_open_pig L2_wood_shelter L3_reel_tower L4_fortress heldout0 heldout1"}
OUT=${OUT:-docs/img}
EXTRA="$*"
printf '%s\n' $LEVELS | xargs -P "${SLURM_CPUS_PER_TASK:-6}" -I{} bash -c \
    "python scripts/ghost_demo.py --levels {} --out-dir $OUT $EXTRA > logs/ghosts_{}.log 2>&1 && echo 'ok   {}' || { echo 'FAIL {}'; tail -5 logs/ghosts_{}.log; }"

list=$(mktemp)
for l in $LEVELS; do [ -f "$OUT/ghosts_$l.mp4" ] && echo "file '$PWD/$OUT/ghosts_$l.mp4'" >> "$list"; done
ffmpeg -v error -y -f concat -safe 0 -i "$list" -c copy "$OUT/ghosts_all.mp4" && echo "ok   $OUT/ghosts_all.mp4"
rm -f "$list"
