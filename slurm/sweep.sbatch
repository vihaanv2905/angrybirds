#!/bin/bash
# Ground truth for a level: try every shot (no learning).
#   sbatch slurm/sweep.sbatch L3_reel_tower arm
#SBATCH --job-name=ab-sweep
#SBATCH --account=engin1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
LEVEL=${1:-L3_reel_tower}
LAUNCHER=${2:-arm}
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh
source .venv/bin/activate
python scripts/sweep_level.py --level "$LEVEL" --launcher "$LAUNCHER" --angles 361 --draws 200 \
    --workers "$SLURM_CPUS_PER_TASK" --out results/sweeps --best-json "levels/sweep_best_${LAUNCHER}_fine.json"
