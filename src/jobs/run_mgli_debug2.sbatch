#!/bin/bash
#SBATCH --job-name=mgli_debug2
#SBATCH --output=logs/mgli_debug2_%j.out
#SBATCH --error=logs/mgli_debug2_%j.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
set -x

echo "==== ENV DUMP (pre) ===="
date
echo "SHELL: $SHELL"
echo "WHOAMI: $(whoami)"
echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"
echo "PATH: $PATH"

# 1) Try to enable the module system first (common on clusters)
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
type module || true

# 2) Try to bring in your user init (for conda PATH, etc.)
[ -f ~/.bashrc ] && source ~/.bashrc || true

# 3) Try conda first
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate protein_design || true
else
  # 4) Fallback: load a Python module if available
  module avail 2>&1 | egrep -i 'Anaconda|Miniconda|Python' | head -n 20 || true
  module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || module load Python 2>/dev/null || true
fi

echo "==== ENV DUMP (post) ===="
echo "PATH: $PATH"
echo "which python : $(command -v python || true)"
echo "which python3: $(command -v python3 || true)"

# 5) Pick a python binary that exists
PYBIN="$(command -v python || true)"
[ -z "$PYBIN" ] && PYBIN="$(command -v python3 || true)"
[ -z "$PYBIN" ] && PYBIN="/usr/bin/python3"
echo "Using PYBIN=$PYBIN"
"$PYBIN" -V || true

# 6) Import test (don’t hard-fail here; just print)
"$PYBIN" - << 'PY' || true
print(">> Import test starting")
try:
    import Bio, torch
    import numpy as np
    print("BioPython:", Bio.__version__)
    print("Torch:", torch.__version__)
    print("NumPy:", np.__version__)
    print("Import test OK")
except Exception as e:
    print("Import test FAILED:", e)
PY

# 7) Check inputs/CSV
ls -1 /home/op98/protein_design/dataset/PP/*.ent.pdb 2>/dev/null | head -n 3 || echo "No PDBs visible"
head -n 2 /home/op98/protein_design/dataset/PP/PP_anti/P2P.csv || echo "CSV not readable"

echo "==== DONE DEBUG2 ===="
