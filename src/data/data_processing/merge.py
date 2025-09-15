#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Merge three tables by pdb_id and keep rows with no missing data; output TSV.")
    ap.add_argument("--features_tsv", required=True, help="TSV with delta_G, complex_count, mean_* columns")
    ap.add_argument("--bfactors_tsv", required=True, help="TSV with PDB_ID, ΔG_kJ/mol, Avg_Bfactor_*")
    ap.add_argument("--charges_csv", required=True, help="CSV with many columns (N, n_pos, n_neg, ...)")
    ap.add_argument("--out_tsv", required=True, help="Output TSV path")
    args = ap.parse_args()

    # 1) Features TSV (keep its delta_G)
    df1 = pd.read_csv(args.features_tsv, sep=",")   # your first file is actually comma-separated
    df1 = df1.rename(columns={"pdb_id": "pdb_id"})

    # 2) B-factors TSV
    df2 = pd.read_csv(args.bfactors_tsv, sep="\t", encoding="utf-8")
    df2 = df2.rename(columns={
        "PDB_ID": "pdb_id",
        "ΔG_kJ/mol": "delta_G_bfactors",  # rename to avoid confusion
        "Avg_Bfactor_Total": "avg_bfactor_total",
        "Avg_Bfactor_Interface": "avg_bfactor_interface",
    })
    # We will drop df2’s delta_G column later

    # 3) Charges CSV (pick subset)
    keep_cols = [
        "pdb_id","N","n_pos","n_neg","net_charge","n_ARG","n_LYS","n_ASP","n_GLU",
        "anisotropy_ratio","min_pos_neg","count_pos_neg_leq_cutoff","mean_nn_dist_all"
    ]
    df3_all = pd.read_csv(args.charges_csv, low_memory=False)
    df3_all.columns = [c.strip() for c in df3_all.columns]
    df3 = df3_all[keep_cols]

    # Merge
    merged = (
        df1.merge(df2[["pdb_id","avg_bfactor_total","avg_bfactor_interface"]], on="pdb_id", how="inner")
           .merge(df3, on="pdb_id", how="inner")
    )

    # Drop rows with missing values
    merged_clean = merged.dropna(how="any").copy()

    # Write TSV
    merged_clean.to_csv(args.out_tsv, sep="\t", index=False)

if __name__ == "__main__":
    main()
