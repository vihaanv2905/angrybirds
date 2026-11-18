#!/bin/bash
# Quick check that the environment works on a compute node (~5 min).
#   sbatch slurm/test.sbatch
#SBATCH --job-name=ab-test
#SBATCH --account=engin1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh
source .venv/bin/activate
python -m pytest -q tests
python train/aim_ppo.py --run-name "test_${SLURM_JOB_ID}" --n-envs 6 --total-steps 3000 \
    --eval-every 1500 --runs-dir logs/test_runs
