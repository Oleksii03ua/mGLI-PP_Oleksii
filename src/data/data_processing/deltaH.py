#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute ΔH from ΔG using a size-dependent entropy model.

Input  (CSV/TSV, auto-sep):
    pdb_id, delta_G, complex_count, ...

Output (TSV):
    binding_affinity_with_deltaH.txt:
        pdb_id    delta_G    complex_count    delta_H
    binding_affinity_with_deltaH_debug.tsv:
        pdb_id, delta_G_kcal, complex_count, N_if, S0, sigma, deltaS_cal_per_mol_K, TDeltaS_kcal, deltaH_kcal

Notes
- Size model:
    N_if = f_if * N_tot
    S0(N_tot)    = 30 + 4 * tanh((N_tot - 600) / 800)
    sigma(N_if)  = 0.55 - 0.10 * tanh((N_if - 80) / 100)
    ΔS (cal/mol/K) = -S0 - sigma * N_if
    ΔH (kcal/mol)  = ΔG (kcal/mol) + T * ΔS / 1000
- Units:
    If FORCE_UNITS is None, an auto-detector will assume kJ/mol when
    median |ΔG| > 40; otherwise kcal/mol. You can override with "kJ" or "kcal".
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ------------------- USER SETTINGS -------------------
INPUT_FILE  = "/home/op98/protein_design/mGLI-pp_Oleksii/src/data/data_files/binding_affinity_two_protein_size_distance.tsv"
OUT_MAIN    = "/home/op98/protein_design/mGLI-pp_Oleksii/src/data/data_files/binding_affinity_with_deltaH2.txt"
OUT_DEBUG   = "/home/op98/protein_design/mGLI-pp_Oleksii/src/data/data_files/binding_affinity_with_deltaH_debug.tsv"

T_K             = 298.0    # Temperature (K)
f_if_default    = 0.12     # Interface fraction of residues
FORCE_UNITS     = None     # None | "kJ" | "kcal"
WRITE_DEBUG_TSV = True
# -----------------------------------------------------

cal_to_kcal = 1.0 / 1000.0

def S0_of_Ntot(Ntot: float) -> float:
    """Rigid-body entropy (cal/mol/K): small modulation with size."""
    return 30.0 + 4.0 * np.tanh((Ntot - 600.0) / 800.0)

def sigma_of_Nif(Nif: float) -> float:
    """Per-interface-residue penalty (cal/mol/K/res): softens for large interfaces."""
    return 0.55 - 0.10 * np.tanh((Nif - 80.0) / 100.0)

def autodetect_units(dg_values):
    """Return 'kcal' or 'kJ' based on magnitude; override with FORCE_UNITS if set."""
    if FORCE_UNITS in ("kJ", "kcal"):
        return FORCE_UNITS
    vals = pd.to_numeric(dg_values, errors="coerce").dropna().astype(float)
    if len(vals) == 0:
        # fallback to kcal if empty/NaN
        return "kcal"
    med = float(np.median(np.abs(vals)))
    # Heuristic: ΔG for PP binding typically ~5–30 kcal/mol; if median > 40, likely kJ/mol.
    return "kJ" if med > 40.0 else "kcal"

def to_kcal(deltaG, units: str) -> float:
    if pd.isna(deltaG):
        return np.nan
    g = float(deltaG)
    return g * 0.239006 if units == "kJ" else g

def estimate_deltaH_row(deltaG_kcal: float, Ntot: float, T_K: float, f_if: float) -> tuple:
    """
    Returns (deltaH_kcal, N_if, S0, sigma, deltaS_cal_per_mol_K, TDeltaS_kcal).
    """
    N_if  = f_if * float(Ntot)
    S0    = S0_of_Ntot(Ntot)
    sigma = sigma_of_Nif(N_if)
    deltaS_cal_per_mol_K = -S0 - sigma * N_if
    TDeltaS_kcal = deltaS_cal_per_mol_K * T_K * cal_to_kcal
    deltaH_kcal  = float(deltaG_kcal) + TDeltaS_kcal
    return deltaH_kcal, N_if, S0, sigma, deltaS_cal_per_mol_K, TDeltaS_kcal

def main():
    # Load (auto-separator)
    df = pd.read_csv(INPUT_FILE, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Minimal column checks
    for col in ("pdb_id", "delta_G", "complex_count"):
        if col not in df.columns:
            raise ValueError(f"Input must contain '{col}' (found columns: {list(df.columns)})")

    # Decide units
    units = autodetect_units(df["delta_G"])
    print(f"[INFO] Interpreting delta_G units as: {units}")

    # Compute
    out_rows_main = []
    debug_rows = []

    for _, row in df.iterrows():
        pid = str(row["pdb_id"]).strip()
        try:
            Ntot = float(row["complex_count"])
        except Exception:
            Ntot = np.nan

        if pd.isna(row["delta_G"]) or pd.isna(Ntot):
            # skip rows with missing target or size
            continue

        dg_kcal = to_kcal(row["delta_G"], units)
        deltaH_kcal, N_if, S0, sigma, dS_cal, T_dS_kcal = estimate_deltaH_row(
            deltaG_kcal=dg_kcal, Ntot=Ntot, T_K=T_K, f_if=f_if_default
        )

        out_rows_main.append({
            "pdb_id": pid,
            "delta_G": float(row["delta_G"]),          # keep original units as in input
            "complex_count": int(Ntot) if np.isfinite(Ntot) else Ntot,
            "delta_H": deltaH_kcal                    # kcal/mol
        })

        if WRITE_DEBUG_TSV:
            debug_rows.append({
                "pdb_id": pid,
                "delta_G_kcal": dg_kcal,
                "complex_count": Ntot,
                "N_if": N_if,
                "S0": S0,
                "sigma": sigma,
                "deltaS_cal_per_mol_K": dS_cal,
                "TDeltaS_kcal": T_dS_kcal,
                "deltaH_kcal": deltaH_kcal
            })

    # Save main (TSV with exactly the four requested columns)
    Path(OUT_MAIN).parent.mkdir(parents=True, exist_ok=True)
    df_main = pd.DataFrame(out_rows_main, columns=["pdb_id", "delta_G", "complex_count", "delta_H"])
    df_main.to_csv(OUT_MAIN, sep="\t", index=False)
    print(f"[INFO] Wrote: {OUT_MAIN}  (delta_H in kcal/mol)")

    # Save debug (optional)
    if WRITE_DEBUG_TSV:
        df_dbg = pd.DataFrame(debug_rows)
        df_dbg.to_csv(OUT_DEBUG, sep="\t", index=False)
        print(f"[INFO] Wrote: {OUT_DEBUG}")

if __name__ == "__main__":
    main()
