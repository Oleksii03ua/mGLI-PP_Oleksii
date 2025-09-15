#!/usr/bin/env python3
"""
Filter binding_affinity.tsv to only those PDB_IDs whose complexes contain exactly two proteins,
as determined by counting non-zero ProtX_Count columns in a “complex size” TSV.
"""

import os
import argparse
import pandas as pd

def filter_two_protein_complexes(size_tsv: str,
                                 affinity_tsv: str,
                                 output_tsv: str) -> None:
    # 1) Load complex‐size data
    size_df = pd.read_csv(size_tsv, sep="\t")
    # Identify all Prot?_Count columns
    prot_cols = [c for c in size_df.columns if c.startswith("Prot") and c.endswith("_Count")]
    if not prot_cols:
        raise ValueError(f"No Prot?_Count columns found in {size_tsv}")

    # 2) Count non-zero entries per row
    nonzero_counts = (size_df[prot_cols] != 0).sum(axis=1)
    # Select PDB_IDs where exactly two proteins are present
    two_prot_pdbs = size_df.loc[nonzero_counts == 2, "PDB_ID"].astype(str).tolist()
    print(f"Found {len(two_prot_pdbs)} PDB_IDs with exactly two proteins.")

    # 3) Load binding affinity data
    aff_df = pd.read_csv(affinity_tsv, sep="\t", dtype={"PDB_ID": str})
    if "PDB_ID" not in aff_df.columns:
        raise ValueError(f"'PDB_ID' column not found in {affinity_tsv}")

    # 4) Filter affinity entries to those PDB_IDs
    filtered = aff_df[aff_df["PDB_ID"].isin(two_prot_pdbs)].copy()
    print(f"Retained {len(filtered)} affinity entries after filtering.")

    # 5) Write out filtered TSV
    os.makedirs(os.path.dirname(output_tsv) or ".", exist_ok=True)
    filtered.to_csv(output_tsv, sep="\t", index=False)
    print(f"Wrote filtered affinities to: {output_tsv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter binding_affinity.tsv by complexes with exactly two proteins"
    )
    parser.add_argument(
        "size_tsv",
        help="TSV file listing Prot1_Count…ProtN_Count per PDB_ID"
    )
    parser.add_argument(
        "affinity_tsv",
        help="Original binding_affinity.tsv (must contain 'PDB_ID' column)"
    )
    parser.add_argument(
        "--output",
        default="binding_affinity_two_proteins.tsv",
        help="Path to write filtered binding_affinity TSV"
    )
    args = parser.parse_args()
    filter_two_protein_complexes(args.size_tsv, args.affinity_tsv, args.output)
 