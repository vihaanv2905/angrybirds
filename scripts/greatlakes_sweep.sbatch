#!/bin/bash
# Brute-force shot sweep on one Great Lakes CPU node (no GPU needed).
#   sbatch --account=<your_account> scripts/greatlakes_sweep.sbatch L3_reel_tower
#SBATCH --job-name=ab-sweep
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
LEVEL=${1:-L3_reel_tower}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11 2>/dev/null || module load python
source .venv/bin/activate
python scripts/sweep_level.py --level "$LEVEL" --angles 361 --draws 200 \
    --workers "$SLURM_CPUS_PER_TASK" --out results/sweeps
