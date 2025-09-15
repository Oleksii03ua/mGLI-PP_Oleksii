# PP_database.py

import os
import re
import argparse
import pandas as pd

from utils import (
    convert_to_molar,
    calculate_dG,
    find_interface_residues,
    load_structure,
    calculate_b_factors as utils_calc_bf
)

def calculate_b_factors(pdb_file, distance_cutoff):
    """
    Wrapper to use utils.calculate_b_factors if available.
    Fallback: Compute averages manually using loaded structure and interfaces.
    """
    if utils_calc_bf:
        return utils_calc_bf(pdb_file, distance_cutoff)
    
    # Manual fallback
    structure = load_structure(pdb_file)
    interface_residues = find_interface_residues(pdb_file, cutoff=distance_cutoff)

    b_total = [atom.get_bfactor() for atom in structure.get_atoms()]
    avg_total = sum(b_total) / len(b_total) if b_total else 0.0

    b_interface = []
    for res in interface_residues:
        for atom in res.get_atoms():
            b_interface.append(atom.get_bfactor())
    avg_interface = sum(b_interface) / len(b_interface) if b_interface else 0.0

    return avg_total, avg_interface

def main():
    parser = argparse.ArgumentParser(
        description="Parse PPI index, compute ΔG from Kd/Ki/IC50, and calculate B-factors."
    )
    parser.add_argument("input_file", help="INDEX_general_PP.2020")
    parser.add_argument("pdb_dir", help="Directory of .ent.pdb files")
    parser.add_argument("output_file", help="TSV output path")
    parser.add_argument("--temperature", type=float, default=298.15,
                        help="Temperature in Kelvin (default: 298.15)")
    parser.add_argument("--cutoff", type=float, default=5.0,
                        help="Å cutoff for interface (default: 5.0)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results = []
    skipped_ltgt = 0

    with open(args.input_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            pdb_code = line.split()[0].lower()

            # Match Kd, Ki, or IC50 and capture <, >, or ~
            match = re.search(r"(Kd|Ki|IC50)([<>=~]?)([\d.]+\s*[fpnumk]?M)", line, re.IGNORECASE)
            if not match:
                continue

            label, symbol, raw_val = match.groups()
            mol, _ = convert_to_molar(raw_val)
            if mol is None:
                continue

            # Skip uncertain entries with < or >
            if symbol in ("<", ">"):
                skipped_ltgt += 1
                continue

            # IC50 is approximated
            if label.upper() == "IC50":
                mol /= 2

            dG = calculate_dG(mol, args.temperature)
            pdb_file = os.path.join(args.pdb_dir, f"{pdb_code}.ent.pdb")
            if not os.path.isfile(pdb_file):
                print(f"[WARN] Missing PDB: {pdb_code}")
                continue

            avg_total, avg_interface = calculate_b_factors(pdb_file, args.cutoff)
            results.append({
                "PDB_ID": pdb_code,
                "ΔG_kJ/mol": round(dG, 2),
                "Avg_Bfactor_Total": round(avg_total, 2),
                "Avg_Bfactor_Interface": round(avg_interface, 2),
            })

    df = pd.DataFrame(results)
    df.to_csv(args.output_file, sep="\t", index=False)

    print(f"✅ Done: {len(df)} entries → {args.output_file}")
    print(f"ℹ️ Skipped entries with '<' or '>' = {skipped_ltgt}")

if __name__ == "__main__":
    main()
