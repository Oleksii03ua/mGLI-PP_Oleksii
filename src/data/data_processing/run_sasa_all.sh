#!/bin/bash
#SBATCH --job-name=sasa_all
#SBATCH --output=/home/op98/protein_design/mGLI-pp_Oleksii/src/data/data_processing/logs/sasa_all_%j.out
#SBATCH --error=/home/op98/protein_design/mGLI-pp_Oleksii/src/data/data_processing/logs/sasa_all_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --account=gerstein

set -xeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="/home/op98/protein_design/mGLI-pp_Oleksii/src/data/data_processing"
VENV="/home/op98/venvs/sasa"

mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

source "$VENV/bin/activate"
python -c "import Bio, Bio.PDB, pandas; print('Imports OK')"
python "$SCRIPT_DIR/sasa_all.py"

