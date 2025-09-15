#!/bin/bash
#SBATCH --job-name=mgli_all
#SBATCH --output=logs/mgli_all_%j.out
#SBATCH --error=logs/mgli_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# #SBATCH --partition=<VALID_PARTITION>
# #SBATCH --account=<YOUR_ACCOUNT>
# #SBATCH --mail-type=END,FAIL

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

echo "======== SLURM JOB INFO ========"
date
echo "Job ID      : ${SLURM_JOB_ID:-unknown}"
echo "Node list   : ${SLURM_NODELIST:-unknown}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Mem (MB)    : ${SLURM_MEM_PER_NODE:-unknown}"
echo "Submit dir  : $SLURM_SUBMIT_DIR"
echo "==============================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ==== Environment: Python module + your venv ====
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
module load Python/3.12.3-GCCcore-13.3.0

# activate the venv you created earlier
source "$HOME/venvs/mgli-py312/bin/activate"

# threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Using python: $(command -v python)"
python -V

# fail fast if packages are missing
python - << 'PY'
import sys, importlib
for m in ("numpy","Bio","torch"):
    try:
        importlib.import_module(m); print(m, "OK")
    except Exception as e:
        print(m, "MISSING:", e); sys.exit(127)
PY

echo "Starting mGLI directory run…"
time srun -c "${SLURM_CPUS_PER_TASK:-1}" python mGLI.py

echo "All done."
date
