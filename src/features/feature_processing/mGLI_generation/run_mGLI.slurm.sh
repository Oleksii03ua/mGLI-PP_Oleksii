#!/bin/bash
#SBATCH --partition=pi_gerstein
#SBATCH --job-name=mGLI
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-2852%10 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=9316361274@vtext.com

module load miniconda         
conda activate protein_design    

cd /home/op98/protein_design/topology/mGLI-PP/src/features/feature_processing/mGLI_generation

PDB_LIST=/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/ids.txt
COMPLETED_LIST=/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/completed_ids.txt

PDBID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PDB_LIST)

# Only run if not already completed
if ! grep -Fxq "$PDBID" "$COMPLETED_LIST"; then
    python run_mGLI.py $PDBID
    if [ $? -eq 0 ]; then
        echo "$PDBID" >> "$COMPLETED_LIST"
        echo "Completed processing $SLURM_ARRAY_TASK_ID PDB ID: $PDBID"
    else
        echo "Failed processing $SLURM_ARRAY_TASK_ID PDB ID: $PDBID"
    fi
else
    echo "Skipping $SLURM_ARRAY_TASK_ID PDB ID: $PDBID (already completed)"
fi
