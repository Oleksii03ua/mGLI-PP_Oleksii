#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd

from utils import (
    find_interface_residues,
    is_hbond,
    POSITIVE,
    NEGATIVE,
    POLAR
)

def compute_interface_counts(pdb_file, cutoff):
    """
    Returns a tuple (charged_count, polar_count, hbond_count)
    for the interface defined by `cutoff` Å.
    """
    interface = find_interface_residues(pdb_file, cutoff=cutoff)

    charged = 0
    polar   = 0
    for res_set in interface.values():
        for _, res in res_set:
            name = res.get_resname()
            if name in POSITIVE or name in NEGATIVE:
                charged += 1
            if name in POLAR:
                polar += 1

    gids = list(interface.keys())
    hbond_ct = 0
    for i in range(len(gids)):
        for j in range(i+1, len(gids)):
            for _, r1 in interface[gids[i]]:
                for _, r2 in interface[gids[j]]:
                    if any(is_hbond(a1, a2) for a1 in r1 for a2 in r2):
                        hbond_ct += 1

    return charged, polar, hbond_ct

def main():
    parser = argparse.ArgumentParser(
        description="Take an existing PPI-summary TSV and append interface counts."
    )
    parser.add_argument("input_tsv",
                       help="TSV with columns including PDB_ID, ΔG_kJ/mol, Avg_Bfactor_Total, Avg_Bfactor_Interface")
    parser.add_argument("pdb_dir",
                       help="Directory where your .ent.pdb files live")
    parser.add_argument("output_tsv",
                       help="Path to write the extended TSV (adds Charged_Count, Polar_Count, Hbond_Count)")
    parser.add_argument("--cutoff", type=float, default=5.0,
                       help="Å cutoff for defining interface (default: 5.0)")
    args = parser.parse_args()

    print("[INFO] Starting PP_database_expanded…")

    # 1) sanity‐check that the input file exists
    if not os.path.isfile(args.input_tsv):
        print(f"[ERROR] Cannot find input TSV: {args.input_tsv}", file=sys.stderr)
        sys.exit(1)

    # 2) peek at the first line to verify it really is your TSV
    with open(args.input_tsv, "r") as fh:
        header = fh.readline().strip().split("\t")
    if "PDB_ID" not in header:
        print(f"[ERROR] Expected 'PDB_ID' in header, got: {header}", file=sys.stderr)
        sys.exit(1)

    # 3) load the true TSV
    df = pd.read_csv(args.input_tsv, sep="\t", header=0, dtype=str)
    total = len(df)
    print(f"[INFO] Loaded {total} rows from {args.input_tsv}")

    # 4) add the three new columns
    df["Charged_Count"] = pd.NA
    df["Polar_Count"]   = pd.NA
    df["Hbond_Count"]   = pd.NA

    # 5) for each row, compute counts from the matching PDB file
    for idx, row in df.iterrows():
        pdb_id = row["PDB_ID"].strip().lower()
        print(f"[INFO] ({idx+1}/{total}) Processing {pdb_id}…", end="", flush=True)

        pdb_path = os.path.join(args.pdb_dir, f"{pdb_id}.ent.pdb")
        if not os.path.isfile(pdb_path):
            print(" MISSING PDB, skipped")
            continue

        charged, polar, hbond = compute_interface_counts(pdb_path, args.cutoff)
        df.at[idx, "Charged_Count"] = charged
        df.at[idx, "Polar_Count"]   = polar
        df.at[idx, "Hbond_Count"]   = hbond

        print(f" ✔ (charged={charged}, polar={polar}, hbond={hbond})")

    # 6) write out the extended table
    os.makedirs(os.path.dirname(args.output_tsv) or ".", exist_ok=True)
    df.to_csv(args.output_tsv, sep="\t", index=False)
    print(f"[INFO] Extended TSV written to {args.output_tsv}")

if __name__ == "__main__":
    main()
