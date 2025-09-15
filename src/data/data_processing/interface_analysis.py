#!/usr/bin/env python3
# interface_analysis.py

import os
import argparse
import numpy as np
from collections import defaultdict
from Bio.PDB import NeighborSearch

# Import utilities from utils.py
from utils import (
    load_structure,
    parse_compound_chains,
    find_interface_residues,
    get_avg_bfactor,
    get_avg_coords,
    is_hbond,
    is_salt_bridge,
    POSITIVE,
    NEGATIVE,
    POLAR
)

def analyze_interface(pdb_file, out_dir, cutoff=5.0):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdb_file))[0]
    out_path = os.path.join(out_dir, f"{base}_interaction_output.tsv")

    # Load structure
    struct = load_structure(pdb_file)

    # Get interface residues via shared function
    interface = find_interface_residues(pdb_file, cutoff=cutoff)

    # Count H-bonds and salt bridges between molecule groups
    hbond_count = saltbridge_count = 0
    gids = list(interface.keys())
    for i in range(len(gids)):
        for j in range(i+1, len(gids)):
            for _, r1 in interface[gids[i]]:
                for _, r2 in interface[gids[j]]:
                    if any(is_hbond(a1, a2) for a1 in r1 for a2 in r2):
                        hbond_count += 1
                    if is_salt_bridge(r1, r2):
                        saltbridge_count += 1

    # Write residue‐level data + summary
    with open(out_path, "w") as out:
        out.write("ProtID\tChain\tRes\tNum\tAvgBfac\tX\tY\tZ\tCharge\tPolar\tIface\n")
        for gid, res_set in interface.items():
            for chain, res in res_set:
                bfac   = get_avg_bfactor(res)
                x, y, z = get_avg_coords(res)
                name   = res.get_resname()
                charge = POSITIVE.get(name, NEGATIVE.get(name, 0.0))
                polar  = 1 if name in POLAR else 0
                out.write(f"{gid}\t{chain}\t{name}\t{res.id[1]}\t"
                          f"{bfac:.2f}\t{x:.2f}\t{y:.2f}\t{z:.2f}\t"
                          f"{charge:.2f}\t{polar}\t1\n")

        out.write("\n=== SUMMARY ===\n")
        for gid, res_set in interface.items():
            mean_b = np.mean([get_avg_bfactor(r) for _,r in res_set])
            total_charge = sum(
                POSITIVE.get(r.get_resname(), 0.0) + NEGATIVE.get(r.get_resname(), 0.0)
                for _, r in res_set
            )
            out.write(f"Protein {gid}:\n")
            out.write(f"  Avg B-factor (interface) = {mean_b:.2f}\n")
            out.write(f"  Total charge            = {total_charge:.2f}\n")
        out.write(f"Total H-bonds      = {hbond_count}\n")
        out.write(f"Total Salt Bridges = {saltbridge_count}\n")

    print(f"✅ Interface analysis written to {out_path}")

def main():
    p = argparse.ArgumentParser(
        description="Analyze PPI interface using shared utilities."
    )
    p.add_argument("pdb_file",  help="Path to input PDB file")
    p.add_argument("output_dir",help="Directory for output TSV")
    p.add_argument("--cutoff",   type=float, default=5.0,
                   help="Å cutoff for interface (default: 5.0)")
    args = p.parse_args()
    analyze_interface(args.pdb_file, args.output_dir, cutoff=args.cutoff)

if __name__ == "__main__":
    main()
