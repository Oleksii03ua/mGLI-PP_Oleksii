#!/usr/bin/env python3
"""
count_residues_summary.py

Parse a PPI index file, count residues per molecule group in each PDB, and produce:
  1) A per-protein counts table (dynamic columns up to the maximum number of proteins).
  2) A complex totals table (sum of all protein residues).

Usage:
    python count_residues_summary.py \
        INDEX_general_PP.2020 \
        /path/to/pdb_dir \
        prot_counts.tsv \
        complex_counts.tsv
"""
import os
import argparse
import pandas as pd

from utils import parse_compound_chains, load_structure


def count_residues(pdb_file, groups):
    """
    Count standard residues in each molecule group (MOL_ID) defined by `groups`.
    Returns a list of counts ordered by MOL_ID index.
    """
    struct = load_structure(pdb_file)
    model = struct[0]
    counts = []
    for chains in groups:
        cnt = 0
        for ch in chains:
            if ch not in model:
                continue
            for res in model[ch]:
                if res.id[0] != " ":  # skip hetatms & waters
                    continue
                cnt += 1
        counts.append(cnt)
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-protein and complex residue counts for PDB complexes."
    )
    parser.add_argument("index_file", help="INDEX_general_PP.2020 file path")
    parser.add_argument("pdb_dir", help="Directory containing .ent.pdb files")
    parser.add_argument("prot_counts_output", help="TSV for per-protein counts")
    parser.add_argument("complex_counts_output", help="TSV for complex total counts")
    args = parser.parse_args()

    # Ensure output dirs exist
    os.makedirs(os.path.dirname(args.prot_counts_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.complex_counts_output), exist_ok=True)

    records = []
    max_groups = 0

    with open(args.index_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            pdb_code = line.split()[0].lower()
            pdb_path = os.path.join(args.pdb_dir, f"{pdb_code}.ent.pdb")
            if not os.path.isfile(pdb_path):
                print(f"[WARN] Missing PDB: {pdb_code}")
                continue

            groups = parse_compound_chains(pdb_path)
            counts = count_residues(pdb_path, groups)
            records.append({"PDB_ID": pdb_code, "counts": counts})
            if len(counts) > max_groups:
                max_groups = len(counts)

    # Build per-protein counts DataFrame
    prot_rows = []
    for rec in records:
        row = {"PDB_ID": rec["PDB_ID"]}
        for i in range(max_groups):
            col = f"Prot{i+1}_Count"
            row[col] = rec["counts"][i] if i < len(rec["counts"]) else 0
        prot_rows.append(row)
    df_prot = pd.DataFrame(prot_rows)
    df_prot.to_csv(args.prot_counts_output, sep="\t", index=False)

    # Build complex totals DataFrame
    comp_rows = []
    for rec in records:
        total = sum(rec["counts"])
        comp_rows.append({"PDB_ID": rec["PDB_ID"], "Complex_Count": total})
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(args.complex_counts_output, sep="\t", index=False)

    print(f"✅ Written per-protein counts → {args.prot_counts_output}")
    print(f"✅ Written complex totals → {args.complex_counts_output}")


if __name__ == "__main__":
    main()
