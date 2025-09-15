# utilis.py
# Oleksii Pidlypenets 

import re
import math
import os
from collections import defaultdict

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

HBOND_DISTANCE       = 3.5
SALT_BRIDGE_DISTANCE = 4.0

POSITIVE = {"LYS": 1.0, "ARG": 1.0, "HIS": 0.09}
NEGATIVE = {"ASP": -1.0, "GLU": -1.0}
POLAR    = {"SER", "THR", "ASN", "GLN", "TYR", "HIS"}
WATER_RESNAMES = {"HOH", "WAT"}

# ─── BINDING / THERMODYNAMICS ─────────────────────────────────────────────────

def convert_to_molar(value_str):
    """Convert binding value (e.g. '15 µM') to molar (float) and return any flag (~, <, >)."""
    match = re.match(r"([~<>]?)\s*([\d.]+)\s*([fpnumk]?M)", value_str, re.IGNORECASE)
    if not match:
        return None, None
    flag, val, unit = match.groups()
    val = float(val)
    factor = {"fm":1e-15, "pm":1e-12, "nm":1e-9, "um":1e-6, "mm":1e-3, "m":1.0}
    return val * factor[unit.lower()], flag

def calculate_dG(K_M, temperature=298.15):
    """Compute ΔG (kJ/mol) from equilibrium constant K (in M⁻¹) at T (K)."""
    R = 8.314  # J/(mol·K)
    dG_J = R * temperature * math.log(K_M)
    return dG_J / 1000.0

# ─── PDB PARSING FOR B-FACTORS & INTERFACE ────────────────────────────────────

def parse_compound_chains(pdb_file):
    """
    Read COMPND records to group chains by molecule.
    Returns list of lists of chain IDs.
    """
    groups = defaultdict(list)
    current_id = None
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("COMPND"):
                continue
            parts = line.split(":",1)
            if len(parts)<2:
                continue
            data = parts[1]
            if "MOL_ID" in line:
                try:
                    current_id = int(data.split(";")[0].strip())
                except ValueError:
                    current_id = None
            elif "CHAIN" in line and current_id is not None:
                chains = data.replace(";", "").split(",")
                groups[current_id].extend(c.strip() for c in chains)
    return list(groups.values())

def load_structure(pdb_file):
    """Load a Bio.PDB Structure from file."""
    parser = PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(pdb_file), pdb_file)

def calculate_b_factors(pdb_file, distance_cutoff):
    """
    Compute average B-factor:
      - total: over all standard residues 
      - interface: atoms within distance_cutoff Å of another chain
    Returns (avg_total, avg_interface).
    """
    struct = load_structure(pdb_file)
    groups = parse_compound_chains(pdb_file)

    atoms_by_gid = defaultdict(list)
    for gid, chains in enumerate(groups):
        for ch in chains:
            if ch not in struct[0]:
                continue
            for res in struct[0][ch]:
                if res.id[0] != " ":
                    continue
                for atom in res:
                    atoms_by_gid[gid].append(atom)

    # all atoms for neighbor search
    all_atoms = [atom for lst in atoms_by_gid.values() for atom in lst]
    ns = NeighborSearch(all_atoms)

    # find interface atoms
    interface_atoms = defaultdict(list)
    for gid, atoms in atoms_by_gid.items():
        for atom in atoms:
            for neigh in ns.search(atom.coord, distance_cutoff):
                other_chain = neigh.get_parent().get_parent().id
                for ogid, chlist in enumerate(groups):
                    if ogid != gid and other_chain in chlist:
                        interface_atoms[gid].append(atom)
                        break

    def avg(bs):
        return sum(bs)/len(bs) if bs else 0.0

    total_bs     = [a.get_bfactor() for a in all_atoms]
    interface_bs = [a.get_bfactor() for lst in interface_atoms.values() for a in lst]

    return avg(total_bs), avg(interface_bs)

# ─── GENERIC INTERFACE DETECTION ──────────────────────────────────────────────

def find_interface_residues(pdb_file, cutoff=5.0, ignore_water=True):
    """
    Identify interface residues by atom–atom contacts across chains.

    Returns: dict[gid] → set of (chain_id, Residue) at interface.
    """
    struct = load_structure(pdb_file)
    groups = parse_compound_chains(pdb_file)

    atoms_by_gid = defaultdict(list)
    for gid, chains in enumerate(groups):
        for ch in chains:
            if ch not in struct[0]:
                continue
            for res in struct[0][ch]:
                if res.id[0] != " ":
                    continue
                if ignore_water and res.get_resname() in WATER_RESNAMES:
                    continue
                for atom in res:
                    atoms_by_gid[gid].append(atom)

    all_atoms = [a for lst in atoms_by_gid.values() for a in lst]
    ns = NeighborSearch(all_atoms)

    interface = defaultdict(set)
    for gid, atoms in atoms_by_gid.items():
        for atom in atoms:
            for neigh in ns.search(atom.coord, cutoff):
                other_chain = neigh.get_parent().get_parent().id
                for ogid, chains in enumerate(groups):
                    if ogid != gid and other_chain in chains:
                        res = atom.get_parent()
                        interface[gid].add((other_chain, res))
                        break
    return interface

# ─── ATOMIC / RESIDUE UTILITIES ───────────────────────────────────────────────

def get_avg_bfactor(residue):
    bs = [a.get_bfactor() for a in residue]
    return sum(bs)/len(bs) if bs else 0.0

def get_avg_coords(residue):
    coords = np.array([a.coord for a in residue])
    return coords.mean(axis=0) if len(coords) else np.zeros(3)

def is_hbond(a1, a2):
    if a1.element in ("N","O") and a2.element in ("N","O"):
        return np.linalg.norm(a1.coord - a2.coord) <= HBOND_DISTANCE
    return False

def is_salt_bridge(r1, r2):
    if "CA" in r1 and "CA" in r2:
        n1, n2 = r1.get_resname(), r2.get_resname()
        if ((n1 in POSITIVE and n2 in NEGATIVE) or
            (n2 in POSITIVE and n1 in NEGATIVE)):
            return np.linalg.norm(r1["CA"].coord - r2["CA"].coord) <= SALT_BRIDGE_DISTANCE
    return False
