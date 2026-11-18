#!/bin/bash
# One-time setup on a Great Lakes LOGIN node (light work only):
#   bash slurm/setup_greatlakes.sh
# Creates .venv in the repo (keep the repo in /scratch, home is small).
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/common.sh            # loads a Python module
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "stable-baselines3[extra]>=2.3" tensorboard pyyaml
python -c "import Box2D, mujoco, stable_baselines3, torch; print('ok: box2d, mujoco', mujoco.__version__, '| sb3', stable_baselines3.__version__, '| torch', torch.__version__)"
echo "Now run a quick check in a compute job:  sbatch slurm/test.sbatch"
