#!/bin/bash
#SBATCH --job-name=mgli_debug
#SBATCH --output=logs/mgli_debug_%j.out
#SBATCH --error=logs/mgli_debug_%j.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Be noisy & stop on errors
set -euo pipefail
set -x

echo "==== ENV DUMP ===="
date
echo "SHELL: $SHELL"
echo "WHOAMI: $(whoami)"
echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"
echo "PATH: $PATH"

# Sometimes conda is only on interactive shells; pull in your bashrc
if [ -f ~/.bashrc ]; then
  source ~/.bashrc || true
fi

# Try to enable conda if present
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate protein_design || true
else
  # Try a module as a fallback (adjust to what your cluster has)
  if command -v module >/dev/null 2>&1; then
    module purge || true
    module avail 2>&1 | egrep -i 'conda|anaconda|miniconda|python' | head -n 20 || true
    module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || module load Python 2>/dev/null || true
  fi
fi

echo "Python path: $(command -v python || echo 'NO PYTHON FOUND')"
python -V || true

# Minimal import test (don’t fail the whole script if missing; just tell us)
python - << 'PY' || true
print(">> Import test starting")
try:
    import Bio, torch
    print("BioPython:", Bio.__version__)
    print("Torch:", torch.__version__)
    print("Import test OK")
except Exception as e:
    print("Import test FAILED:", e)
PY

# List the input and CSV to confirm visibility
ls -1 /home/op98/protein_design/dataset/PP/*.ent.pdb 2>/dev/null | head -n 3 || echo "No PDBs visible"
head -n 2 /home/op98/protein_design/dataset/PP/PP_anti/P2P.csv || echo "CSV not readable"

echo "==== DONE DEBUG ===="
