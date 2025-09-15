#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate mGLI features and save a .pt tensor for EVERY PDB file in a directory.
- PDBs: /home/op98/protein_design/dataset/PP/*.ent.pdb
- CSV : /home/op98/protein_design/dataset/PP/PP_anti/PDBbind_v2020_PP_anti.csv
- Out : /home/op98/protein_design/dataset/PP/PP_anti/mGLI_pt/<PDB_ID>_mGLI.pt
"""

import os
import time
import math
import csv
import numpy as np
import Bio.PDB as PDB
from Bio.PDB.Polypeptide import is_aa

import torch  # for saving .pt tensors


# ========= PATHS / CONFIG =========
PDB_DIR  = "/home/op98/protein_design/dataset/PP"
CSV_PATH = "/home/op98/protein_design/dataset/PP/PP_anti/P2P.csv"
OUT_DIR  = "/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/pt"
os.makedirs(OUT_DIR, exist_ok=True)
# ==================================


# ---------- Core geometry ----------
def Gauss_linking_integral(line1, line2):
    a = [line1.startpoint, line1.endpoint]
    b = [line2.startpoint, line2.endpoint]

    R = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            R[i, j] = a[i] - b[j]

    cprod = [
        np.cross(R[0, 0], R[0, 1]),
        np.cross(R[0, 1], R[1, 1]),
        np.cross(R[1, 1], R[1, 0]),
        np.cross(R[1, 0], R[0, 0]),
    ]
    n = [c / (np.linalg.norm(c) + 1e-6) for c in cprod]

    area1 = np.arcsin(np.dot(n[0], n[1]))
    area2 = np.arcsin(np.dot(n[1], n[2]))
    area3 = np.arcsin(np.dot(n[2], n[3]))
    area4 = np.arcsin(np.dot(n[3], n[0]))

    sign = np.sign(np.cross(a[1] - a[0], b[1] - b[0]).dot(a[0] - b[0]))
    Area = area1 + area2 + area3 + area4
    return float(sign * Area)


class Line:
    def __init__(self, startpoint, endpoint, start_type, end_type):
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.start_type = start_type
        self.end_type = end_type


class Atom:
    def __init__(self, data, etype, eid=None):
        self.data = data
        self.etype = etype  # atom element
        self.eid = eid      # atom full name (e.g., " N  ", " C  ")


class bonded_Atom:
    def __init__(self, lines, atom):
        self.lines = lines
        self.atom = atom

    def add_line(self, line):
        self.lines.append(line)


def get_bond(atoms, atom):
    maximal_bond_lengths = [["C", "H", 1.19],
                            ["C", "C", 1.64],
                            ["C", "S", 1.92],
                            ["C", "N", 1.57],
                            ["C", "O", 1.57],
                            ["N", "H", 1.11],
                            ["N", "O", 1.56],
                            ["N", "N", 1.55],
                            ["N", "S", 2.06],
                            ["O", "H", 1.07],
                            ["O", "S", 1.52],
                            ["O", "O", 1.58],
                            ["S", "H", 1.45],
                            ["S", "S", 2.17],
                            ["H", "H", 0.84]]

    bonds = []
    target_etype = atom.etype
    target_data = atom.data
    for x in atoms:
        etype = x.etype
        data = x.data
        for e1, e2, max_d in maximal_bond_lengths:
            if (etype == e1 and target_etype == e2) or (etype == e2 and target_etype == e1):
                dist = math.dist(data, target_data)
                if dist < max_d:
                    midpoint = (target_data + data) / 2.0
                    bonds.append(Line(target_data, midpoint, target_etype, etype))
    return bonds


def get_bonded_atoms(atoms):
    bonded_atoms = []
    for atom in atoms:
        bonds = get_bond(atoms, atom)
        bonded_atoms.append(bonded_Atom(bonds, atom))
    return bonded_atoms


# ---------- Parse PDB + partners ----------
def get_protein_struct_from_pdb(pdbid, pdb_path, csv_path):
    pid = pdbid.strip()
    partner1_ids = partner2_ids = None

    # find partners in CSV
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_pid = str(row.get("id", row.get("ID", ""))).strip()
            if row_pid != pid:
                continue
            partner1_ids = str(row["partner1"]).strip()
            partner2_ids = str(row["partner2"]).strip()
            break

    if not partner1_ids or not partner2_ids:
        raise ValueError(f"PDB {pid}: not found in CSV or missing partner columns.")

    # split chain lists
    partner1_ids = [c.strip() for c in partner1_ids.replace(";", ",").split(",") if c.strip()]
    partner2_ids = [c.strip() for c in partner2_ids.replace(";", ",").split(",") if c.strip()]

    # parse structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pid, pdb_path)
    model = next(structure.get_models())  # first model
    chain_by_id = {c.id: c for c in model}

    def is_amino_acid_chain(chain):
        return any(is_aa(res, standard=True) for res in chain)

    partner1_ids = [cid for cid in partner1_ids if cid in chain_by_id and is_amino_acid_chain(chain_by_id[cid])]
    partner2_ids = [cid for cid in partner2_ids if cid in chain_by_id and is_amino_acid_chain(chain_by_id[cid])]

    if not partner1_ids or not partner2_ids:
        raise ValueError(f"PDB {pid}: partner1 or partner2 has no amino-acid chain after filtering.")

    print(f"PDB {pid}: partner1:{partner1_ids} / partner2:{partner2_ids}")

    def build_one_chain_struct(chain):
        chain_struct, pre_struc = [], None
        for residue in chain:
            if residue.id[0] != " ":
                continue
            atoms = [Atom(a.get_coord(), a.element, a.fullname) for a in residue]
            cur_struc = get_bonded_atoms(atoms)

            # peptide bond halves between previous C and current N
            if pre_struc is not None:
                for cur in cur_struc:
                    if getattr(cur.atom, "eid", "") == " N  ":
                        for pre in pre_struc:
                            if getattr(pre.atom, "eid", "") == " C  ":
                                start, end = pre.atom.data, cur.atom.data
                                mid = (start + end) / 2.0
                                pre.add_line(Line(start, mid, pre.atom.etype, cur.atom.etype))
                                cur.add_line(Line(end, mid, cur.atom.etype, pre.atom.etype))

            chain_struct += cur_struc
            pre_struc = cur_struc
        return chain_struct

    def build_group_struct(chain_ids):
        struct = []
        for cid in chain_ids:
            struct += build_one_chain_struct(chain_by_id[cid])
        return struct

    struct1 = build_group_struct(partner1_ids)
    struct2 = build_group_struct(partner2_ids)
    return struct1, struct2


# ---------- GLI calculator ----------
class Gli_calculator:
    def __init__(self):
        self.struct1 = []
        self.struct2 = []
        self.result = {}

    def add_pair(self, bonded_atom1, bonded_atom2):
        self.struct1.append(bonded_atom1)
        self.struct2.append(bonded_atom2)

    def calculate_result(self):
        for bonded_atom1 in self.struct1:
            for bonded_atom2 in self.struct2:
                L = []
                for line1 in bonded_atom1.lines:
                    for line2 in bonded_atom2.lines:
                        gli = Gauss_linking_integral(line1, line2)
                        L.append(abs(gli))  # ACN: avoid cancellation
                self.result[bonded_atom1, bonded_atom2] = L
                self.result[bonded_atom2, bonded_atom1] = L

    def get_result(self, bonded_atom1, bonded_atom2):
        return self.result.get((bonded_atom1, bonded_atom2), 0)


# NOTE: this keeps the original behavior of returning at most ONE neighbor (due to break)
def cutoff_struct(protein_struct, query_bonded_atom, cutoff=16):
    new_struct = []
    for bonded_atom1 in protein_struct:
        if math.dist(bonded_atom1.atom.data, query_bonded_atom.atom.data) < cutoff:
            new_struct.append(bonded_atom1)
            break
    return new_struct


def get_mGLI(e1, e2, r1, r2, protein_struct1, protein_struct2, calculator):
    Gli_sum = []
    for bonded_atom1 in protein_struct1:
        if bonded_atom1.atom.etype == e1:
            gli_sum = 0.0
            for bonded_atom2 in protein_struct2:
                if bonded_atom2.atom.etype == e2:
                    dist = math.dist(bonded_atom1.atom.data, bonded_atom2.atom.data)
                    if r1 < dist < r2:
                        val = calculator.get_result(bonded_atom1, bonded_atom2)
                        # calculator.get_result returns a LIST of GLI values; sum it
                        if isinstance(val, (list, tuple, np.ndarray)):
                            val = float(np.sum(val))
                        gli_sum += float(val)
            Gli_sum.append(gli_sum)

    if not Gli_sum:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    arr = np.array(Gli_sum, dtype=float)
    return [
        float(np.sum(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.median(arr)),
    ]


def get_inteval(n=10, minimal=2, maximal=11):
    scales = np.linspace(minimal, maximal, n)
    return [[float(scales[i]), float(scales[i + 1])] for i in range(n - 1)]


def generate_mGLI_feature(pdbid):
    pdb_path = f"{PDB_DIR}/{pdbid}.ent.pdb"
    protein_struct1, protein_struct2 = get_protein_struct_from_pdb(pdbid, pdb_path, CSV_PATH)

    print("struct,done")
    calculator = Gli_calculator()

    for query_bonded_atom in protein_struct2:
        new_struct = cutoff_struct(protein_struct1, query_bonded_atom)
        for bonded_atom1 in new_struct:
            calculator.add_pair(bonded_atom1, query_bonded_atom)

    # deduplicate
    calculator.struct1 = list(set(calculator.struct1))
    print("calculator,ready")
    print(len(calculator.struct1), len(calculator.struct2))
    calculator.calculate_result()
    print("calculation,done")

    el = ['C', 'N', 'O', 'S']
    mGLI_feature = []
    for e1 in el:
        for e2 in el:
            for r1, r2 in get_inteval():
                mGLI_feature += get_mGLI(e1, e2, r1, r2, protein_struct1, protein_struct2, calculator)
                mGLI_feature += get_mGLI(e2, e1, r1, r2, protein_struct2, protein_struct1, calculator)

    return mGLI_feature


# ---------- Batch over directory ----------
def find_all_pdb_ids(pdb_dir):
    pdbids = []
    for entry in os.scandir(pdb_dir):
        if entry.is_file() and entry.name.endswith(".ent.pdb"):
            pdbids.append(entry.name[:-8])  # strip ".ent.pdb"
    pdbids.sort()
    return pdbids


if __name__ == "__main__":
    pdbid_list = find_all_pdb_ids(PDB_DIR)
    print(f"Discovered {len(pdbid_list)} PDB files in {PDB_DIR}")

    ok, fail = 0, 0
    for pdbid in pdbid_list:
        try:
            start = time.time()
            features = generate_mGLI_feature(pdbid)

            tensor = torch.tensor(features, dtype=torch.float32)
            out_path = os.path.join(OUT_DIR, f"{pdbid}_mGLI.pt")
            torch.save(tensor, out_path)

            end = time.time()
            ok += 1
            print(f"[OK]  {pdbid} -> {out_path}  ({end - start:.2f}s)")
        except Exception as e:
            fail += 1
            print(f"[SKIP] {pdbid}: {e}")

    print(f"\nDone. Saved {ok} tensors. Skipped {fail}. Output dir: {OUT_DIR}")

