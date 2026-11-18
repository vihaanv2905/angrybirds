# Sourced by the Slurm scripts. Loads a Python 3.10+ module on Great Lakes.
# Module names change over time: `module avail python` lists what exists;
# set AB_PYTHON_MODULE to pin one, e.g. export AB_PYTHON_MODULE=python/3.11.5
if command -v module >/dev/null 2>&1; then
  if [ -n "${AB_PYTHON_MODULE:-}" ]; then
    module load "$AB_PYTHON_MODULE"
  else
    module load python/3.11.5 2>/dev/null || module load python/3.11 2>/dev/null \
      || module load python/3.10.4 2>/dev/null || module load python 2>/dev/null || true
  fi
fi
python - <<'PY'
import sys
assert sys.version_info >= (3, 10), f"need Python >= 3.10, got {sys.version}"
print("python", sys.version.split()[0])
PY
# keep each env worker single-threaded (36 workers on a 36-core node)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# checkpoints + TensorBoard go to scratch, not the repo
# (Great Lakes scratch is /scratch/<account>_root/<account>/<uniqname>; falls
#  back to $HOME if that does not exist for this account)
export AB_CKPT_DIR="${AB_CKPT_DIR:-/scratch/vihaanv_root/vihaanv/$USER/angrybirds_ckpt}"
mkdir -p "$AB_CKPT_DIR" 2>/dev/null || export AB_CKPT_DIR="$HOME/angrybirds_ckpt"
mkdir -p "$AB_CKPT_DIR" logs
echo "checkpoints -> $AB_CKPT_DIR"
