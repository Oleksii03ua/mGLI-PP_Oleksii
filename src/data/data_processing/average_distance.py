#!/usr/bin/env python3
# add_charge_distances.py
import argparse, csv, os, re, sys
from typing import Dict, Tuple, List, Optional
import torch

# -----------------------
# I/O helpers
# -----------------------
def load_pt(path: str) -> Dict:
    """Safe-ish torch load with fallback for older versions."""
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)

def autodetect_delimiter(header_line: str) -> str:
    return '\t' if '\t' in header_line and ',' not in header_line else ','

def read_table(path: str) -> Tuple[List[Dict[str, str]], List[str], str]:
    """Return rows (as dicts), fieldnames, and delimiter (auto-detected)."""
    with open(path, 'r', newline='') as f:
        first = f.readline()
        if not first:
            raise ValueError(f"{path} is empty.")
        delim = autodetect_delimiter(first)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames, delim

def write_table(path: str, rows: List[Dict[str, str]], fieldnames: List[str], delimiter: str):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def guess_pdb_id_from_filename(path: str) -> str:
    """
    Try to extract a 4-char PDB ID (starting with a digit) from the filename.
    Fallback: use the stem before first '_' or the full stem.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'([0-9][A-Za-z0-9]{3})', stem)
    if m:
        return m.group(1).lower()
    return stem.split('_')[0].lower()

# -----------------------
# Geometry
# -----------------------
def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (Na,3), b: (Nb,3) -> (Na,Nb)
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(dim=2).sqrt()

# -----------------------
# Metrics (existing + new)
# -----------------------
def mean_nearest_opposite_cross_protein(data: Dict) -> float:
    """
    Average (over residues) of the nearest distance to an opposite-charged
    residue on a *different* protein (group_id).
    Expects charged dataset: 'charge' ∈ {+1,-1}.
    """
    coords = torch.tensor(data["coords"], dtype=torch.float32)   # (N,3)
    charges = torch.tensor(data["charge"], dtype=torch.int8)     # (N,)
    groups  = torch.tensor(data["group_id"], dtype=torch.int16)  # (N,)

    uniq_groups = torch.unique(groups).tolist()
    if len(uniq_groups) < 2:
        raise ValueError("Need at least two different group_id values in the .pt file.")

    nearest_chunks: List[torch.Tensor] = []

    for i, gi in enumerate(uniq_groups):
        for gj in uniq_groups[i+1:]:
            mask_i = (groups == gi)
            mask_j = (groups == gj)

            pos_i = mask_i & (charges ==  1)
            neg_i = mask_i & (charges == -1)
            pos_j = mask_j & (charges ==  1)
            neg_j = mask_j & (charges == -1)

            # gi(+) -> nearest gj(-)
            if pos_i.any() and neg_j.any():
                d = pairwise_dist(coords[pos_i], coords[neg_j])
                nearest_chunks.append(d.min(dim=1).values)

            # gi(-) -> nearest gj(+)
            if neg_i.any() and pos_j.any():
                d = pairwise_dist(coords[neg_i], coords[pos_j])
                nearest_chunks.append(d.min(dim=1).values)

            # symmetric directions (gj vs gi)
            if pos_j.any() and neg_i.any():
                d = pairwise_dist(coords[pos_j], coords[neg_i])
                nearest_chunks.append(d.min(dim=1).values)

            if neg_j.any() and pos_i.any():
                d = pairwise_dist(coords[neg_j], coords[pos_i])
                nearest_chunks.append(d.min(dim=1).values)

    if not nearest_chunks:
        raise ValueError("No oppositely charged cross-protein pairs found.")

    nearest = torch.cat(nearest_chunks)
    return float(nearest.mean().item())  # Å

def mean_nonpolar_to_nonpolar_cross(data_nonpolar: Dict) -> float:
    """
    For every non-polar residue, find nearest non-polar residue on a different protein.
    Average all such nearest distances (both directions across group pairs).
    Expects non-polar dataset (all 'charge' == 0 is fine; not used here).
    """
    coords = torch.tensor(data_nonpolar["coords"], dtype=torch.float32)   # (N,3)
    groups = torch.tensor(data_nonpolar["group_id"], dtype=torch.int16)   # (N,)

    uniq_groups = torch.unique(groups).tolist()
    if len(uniq_groups) < 2:
        raise ValueError("Need at least two different group_id values in the non-polar .pt file.")

    nearest_chunks: List[torch.Tensor] = []

    for i, gi in enumerate(uniq_groups):
        for gj in uniq_groups[i+1:]:
            mask_i = (groups == gi)
            mask_j = (groups == gj)

            if mask_i.any() and mask_j.any():
                d1 = pairwise_dist(coords[mask_i], coords[mask_j])
                nearest_chunks.append(d1.min(dim=1).values)
                d2 = pairwise_dist(coords[mask_j], coords[mask_i])
                nearest_chunks.append(d2.min(dim=1).values)

    if not nearest_chunks:
        raise ValueError("No non-polar cross-protein pairs found.")

    nearest = torch.cat(nearest_chunks)
    return float(nearest.mean().item())

def mean_nonpolar_to_charged_cross(data_nonpolar: Dict, data_charged: Dict) -> float:
    """
    For every non-polar residue, find nearest charged residue on a different protein.
    Average all such nearest distances (both directions across group pairs).
    """
    np_coords = torch.tensor(data_nonpolar["coords"], dtype=torch.float32)
    np_groups = torch.tensor(data_nonpolar["group_id"], dtype=torch.int16)

    ch_coords = torch.tensor(data_charged["coords"], dtype=torch.float32)
    ch_groups = torch.tensor(data_charged["group_id"], dtype=torch.int16)

    # Require at least 2 groups on both sides to ensure cross-protein search is meaningful
    if len(torch.unique(np_groups)) < 2 or len(torch.unique(ch_groups)) < 2:
        raise ValueError("Need at least two different group_id values in both non-polar and charged .pt files.")

    nearest_chunks: List[torch.Tensor] = []

    # For each pair of distinct groups across the *same* protein indexing,
    # compute NP(group i) -> Charged(group j) and NP(group j) -> Charged(group i)
    uniq_groups = sorted(set(torch.unique(np_groups).tolist()) | set(torch.unique(ch_groups).tolist()))
    for i, gi in enumerate(uniq_groups):
        for gj in uniq_groups[i+1:]:
            np_mask_i = (np_groups == gi)
            np_mask_j = (np_groups == gj)
            ch_mask_i = (ch_groups == gi)
            ch_mask_j = (ch_groups == gj)

            # NP in gi -> Charged in gj
            if np_mask_i.any() and ch_mask_j.any():
                d = pairwise_dist(np_coords[np_mask_i], ch_coords[ch_mask_j])
                nearest_chunks.append(d.min(dim=1).values)
            # NP in gj -> Charged in gi
            if np_mask_j.any() and ch_mask_i.any():
                d = pairwise_dist(np_coords[np_mask_j], ch_coords[ch_mask_i])
                nearest_chunks.append(d.min(dim=1).values)

    if not nearest_chunks:
        raise ValueError("No non-polar to charged cross-protein pairs found.")

    nearest = torch.cat(nearest_chunks)
    return float(nearest.mean().item())

# -----------------------
# Main
# -----------------------
def main():
    default_table = "/home/op98/protein_design/topology/mGLI-PP/src/data/data_files/binding_affinity_two_protein_size_distance.tsv"

    ap = argparse.ArgumentParser(description="Append cross-protein distance metrics to table by matching pdb_id.")
    ap.add_argument("--pt-dir", required=True, help="Directory with .pt files (expects pairs like 1abc_charged.pt and 1abc_nonpolar.pt)")
    ap.add_argument("--in-table", default=default_table, help=f"Input CSV/TSV (default: {default_table})")
    ap.add_argument("--out-table", default=default_table, help=f"Output CSV/TSV (default: overwrite {default_table})")
    ap.add_argument("--col-opposite", default="mean_nearest_opposite_A",
                    help="Column for mean nearest opposite-charged cross-protein distance (Å).")
    ap.add_argument("--col-np-np", default="mean_nonpolar_to_nonpolar_cross_A",
                    help="Column for mean nearest non-polar→non-polar cross-protein distance (Å).")
    ap.add_argument("--col-np-ch", default="mean_nonpolar_to_charged_cross_A",
                    help="Column for mean nearest non-polar→charged cross-protein distance (Å).")
    args = ap.parse_args()

    if not os.path.isdir(args.pt_dir):
        print(f"ERROR: --pt-dir '{args.pt_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Index .pt files by stem type: { pdb_key: {"charged": path, "nonpolar": path} }
    by_pdb: Dict[str, Dict[str, str]] = {}
    for fname in os.listdir(args.pt_dir):
        if not fname.lower().endswith(".pt"):
            continue
        fpath = os.path.join(args.pt_dir, fname)
        stem = os.path.splitext(fname)[0].lower()
        # classify
        kind: Optional[str] = None
        if stem.endswith("_charged"):
            kind = "charged"
            pdb_key = stem[:-8]  # strip "_charged"
        elif stem.endswith("_nonpolar"):
            kind = "nonpolar"
            pdb_key = stem[:-9]  # strip "_nonpolar"
        else:
            # fallback: we’ll treat as charged-only file (old behavior)
            kind = "charged"
            pdb_key = stem

        d = by_pdb.setdefault(pdb_key, {})
        d[kind] = fpath

    # Compute metrics per pdb_id
    res_opposite: Dict[str, float] = {}
    res_np_np: Dict[str, float] = {}
    res_np_ch: Dict[str, float] = {}
    failures: List[Tuple[str, str]] = []

    for pdb_key, files in sorted(by_pdb.items()):
        charged_path = files.get("charged")
        nonpolar_path = files.get("nonpolar")

        # Opposite-charge metric: requires charged dataset
        if charged_path:
            try:
                ch = load_pt(charged_path)
                # basic key check
                for k in ("coords", "charge", "group_id"):
                    if k not in ch:
                        raise KeyError(f"Missing key '{k}' in {os.path.basename(charged_path)}")
                pdb_id = str(ch.get("pdb_id", guess_pdb_id_from_filename(charged_path))).lower()
                res_opposite[pdb_id] = mean_nearest_opposite_cross_protein(ch)
            except Exception as e:
                failures.append((os.path.basename(charged_path), str(e)))

        # Non-polar → Non-polar metric
        if nonpolar_path:
            try:
                npd = load_pt(nonpolar_path)
                for k in ("coords", "group_id"):
                    if k not in npd:
                        raise KeyError(f"Missing key '{k}' in {os.path.basename(nonpolar_path)}")
                pdb_id_np = str(npd.get("pdb_id", guess_pdb_id_from_filename(nonpolar_path))).lower()
                res_np_np[pdb_id_np] = mean_nonpolar_to_nonpolar_cross(npd)
            except Exception as e:
                failures.append((os.path.basename(nonpolar_path), str(e)))

        # Non-polar → Charged metric (needs both)
        if nonpolar_path and charged_path:
            try:
                # reuse loaded if available
                npd = 'npd' in locals() and isinstance(locals()['npd'], dict) and locals()['npd'] or load_pt(nonpolar_path)
                ch  = 'ch'  in locals() and isinstance(locals()['ch'], dict)  and locals()['ch']  or load_pt(charged_path)
                pdb_id_combo = str(
                    (npd.get("pdb_id", None) or ch.get("pdb_id", None) or guess_pdb_id_from_filename(nonpolar_path))
                ).lower()
                res_np_ch[pdb_id_combo] = mean_nonpolar_to_charged_cross(npd, ch)
            except Exception as e:
                failures.append((f"{os.path.basename(nonpolar_path)} + {os.path.basename(charged_path)}", str(e)))

    # Read input table, append/merge columns
    rows, headers, delim = read_table(args.in_table)

    # Ensure pdb_id presence and keep original header casing
    header_map = {h.lower(): h for h in headers}
    if "pdb_id" not in header_map:
        print("ERROR: Input table must have a 'pdb_id' column.", file=sys.stderr)
        sys.exit(1)

    # Add new columns if not present
    for col in (args.col_opposite, args.col_np_np, args.col_np_ch):
        if col not in headers:
            headers.append(col)

    # Fill values
    missing_op, missing_npnp, missing_npch = 0, 0, 0
    for r in rows:
        pid = (r.get(header_map["pdb_id"], "") or "").strip().lower()

        # Opposite-charged (existing behavior)
        v_op = res_opposite.get(pid, None)
        if v_op is None:
            r[args.col_opposite] = ""
            missing_op += 1
        else:
            r[args.col_opposite] = f"{v_op:.6f}"

        # Non-polar -> Non-polar
        v_npnp = res_np_np.get(pid, None)
        if v_npnp is None:
            r[args.col_np_np] = ""
            missing_npnp += 1
        else:
            r[args.col_np_np] = f"{v_npnp:.6f}"

        # Non-polar -> Charged
        v_npch = res_np_ch.get(pid, None)
        if v_npch is None:
            r[args.col_np_ch] = ""
            missing_npch += 1
        else:
            r[args.col_np_ch] = f"{v_npch:.6f}"

    # Write output (supports in-place overwrite)
    write_table(args.out_table, rows, headers, delim)

    # Summary
    print(f"[OK] Wrote: {args.out_table}", file=sys.stderr)
    print(f"[INFO] Opposite-charged computed for {len(res_opposite)} PDBs; rows without match: {missing_op}", file=sys.stderr)
    print(f"[INFO] Nonpolar→Nonpolar computed for {len(res_np_np)} PDBs; rows without match: {missing_npnp}", file=sys.stderr)
    print(f"[INFO] Nonpolar→Charged computed for {len(res_np_ch)} PDBs; rows without match: {missing_npch}", file=sys.stderr)
    if failures:
        print("[WARN] Some computations failed:", file=sys.stderr)
        for fname, msg in failures[:20]:
            print(f"  - {fname}: {msg}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)

if __name__ == "__main__":
    main()
