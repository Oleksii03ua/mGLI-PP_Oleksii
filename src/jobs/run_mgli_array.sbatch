#!/bin/bash
#SBATCH --job-name=mgli_arr
#SBATCH --output=logs/mgli_arr_%A_%a.out
#SBATCH --error=logs/mgli_arr_%A_%a.err
#SBATCH --partition=day
#SBATCH --account=gerstein
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# env
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
module load Python/3.12.3-GCCcore-13.3.0
source "$HOME/venvs/mgli-py312/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# pick PDBID from todo list (1-based sed indexing)
PDBID=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ids_todo.txt)
if [ -z "${PDBID:-}" ]; then
  echo "No PDBID for index $SLURM_ARRAY_TASK_ID"; exit 0
fi

OUT_DIR="/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/pt"
OUT_FILE="$OUT_DIR/${PDBID}_mGLI.pt"
if [ -f "$OUT_FILE" ]; then
  echo "[SKIP-EXIST] $OUT_FILE"; exit 0
fi

echo "Task $SLURM_ARRAY_TASK_ID -> $PDBID"

time srun -c "${SLURM_CPUS_PER_TASK:-1}" python - <<PY
from mGLI import generate_mGLI_feature
import torch, os
pdbid = "$PDBID"
out_dir = "$OUT_DIR"
os.makedirs(out_dir, exist_ok=True)
feat = generate_mGLI_feature(pdbid)
torch.save(torch.tensor(feat, dtype=torch.float32), os.path.join(out_dir, f"{pdbid}_mGLI.pt"))
print("[OK]", pdbid)
PY
