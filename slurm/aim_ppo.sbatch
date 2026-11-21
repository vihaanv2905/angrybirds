#!/bin/bash
# Stage A: train the aiming policy π_aim with PPO on one CPU node.
#
#   sbatch slurm/aim_ppo.sbatch <run_name> [train/aim_ppo.py options...]
#
# Suggested sequence (each is a separate job):
#   sbatch slurm/aim_ppo.sbatch a1_starter_1bird  --birds 1 --total-steps 300000
#   sbatch slurm/aim_ppo.sbatch a2_starter_3birds --birds 3 --total-steps 1000000
#   sbatch --time=08:00:00 slurm/aim_ppo.sbatch a3_mix --level-mix 0.8 --total-steps 5000000
# Afterwards, on the login node:
#   git add runs/<run_name> && git commit -m "run <run_name>" && git push
#SBATCH --job-name=aim-ppo
#SBATCH --account=engin1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
RUN_NAME=${1:?usage: sbatch slurm/aim_ppo.sbatch <run_name> [options]}
shift
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh
source .venv/bin/activate
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK cpus, run $RUN_NAME"
python train/aim_ppo.py --run-name "$RUN_NAME" --n-envs $((SLURM_CPUS_PER_TASK - 2)) \
    --ckpt-dir "$AB_CKPT_DIR" "$@"
