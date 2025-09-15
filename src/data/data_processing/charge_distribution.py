# charge_distribution.py

import argparse
import sys
from pathlib import Path
import torch
from utils import find_interface_residues

# Only strictly charged at pH ~7
CHARGED_SET = {"ARG", "LYS", "ASP", "GLU"}

# Canonical non-polar (hydrophobic) side chains (include CYS commonly treated as weakly non-polar)
NONPOLAR_SET = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY", "CYS"}

def _formal_charge(resname: str) -> int:
    """Formal side-chain charge at ~pH 7."""
    if resname in ("ARG", "LYS"):
        return +1
    if resname in ("ASP", "GLU"):
        return -1
    return 0

def _residue_centroid(residue, heavy_only: bool = False):
    """Average (x,y,z) across residue atoms."""
    coords = []
    for atom in residue:
        if heavy_only and atom.element == "H":
            continue
        coords.append(atom.coord)
    if not coords:
        return None
    import numpy as np
    arr = np.vstack(coords)
    return tuple(arr.mean(axis=0).tolist())

def _empty_result():
    return {
        "coords": torch.empty((0, 3), dtype=torch.float32),
        "charge": torch.empty((0,), dtype=torch.long),
        "resname": [],
        "chain": [],
        "resseq": [],
        "icode": [],
        "group_id": [],
    }

def extract_interface_charged(
    pdb_file: str,
    cutoff: float = 5.0,
    ignore_water: bool = True,
    heavy_only: bool = False,
    save_pt: str | None = None,
):
    """
    Finds interface residues across distinct proteins (MOL_ID groups),
    keeps only ARG, LYS, ASP, GLU, computes centroid coords & charges,
    returns PyTorch-friendly dict.
    """
    interface = find_interface_residues(
        pdb_file, cutoff=cutoff, ignore_water=ignore_water
    )

    coords_list, charges_list = [], []
    resname_list, chain_list, resseq_list, icode_list, gid_list = [], [], [], [], []

    for gid, entries in interface.items():
        for (_other_chain, residue) in entries:
            resname = residue.get_resname().upper()
            if resname not in CHARGED_SET:
                continue

            centroid = _residue_centroid(residue, heavy_only=heavy_only)
            if centroid is None:
                continue

            charge = _formal_charge(resname)
            chain_id = residue.get_parent().id
            _, resseq, icode = residue.id
            icode = icode.strip() if isinstance(icode, str) else ""

            coords_list.append(centroid)
            charges_list.append(charge)
            resname_list.append(resname)
            chain_list.append(chain_id)
            resseq_list.append(int(resseq))
            icode_list.append(icode)
            gid_list.append(int(gid))

    result = (
        _empty_result()
        if not coords_list
        else {
            "coords": torch.tensor(coords_list, dtype=torch.float32),
            "charge": torch.tensor(charges_list, dtype=torch.long),
            "resname": resname_list,
            "chain": chain_list,
            "resseq": resseq_list,
            "icode": icode_list,
            "group_id": gid_list,
        }
    )

    if save_pt:
        torch.save(result, save_pt)
    return result

def extract_interface_nonpolar(
    pdb_file: str,
    cutoff: float = 5.0,
    ignore_water: bool = True,
    heavy_only: bool = False,
    save_pt: str | None = None,
):
    """
    Finds interface residues across distinct proteins (MOL_ID groups),
    keeps only non-polar residues, computes centroid coords,
    stores same fields as the charged set. charge=0 for all.
    """
    interface = find_interface_residues(
        pdb_file, cutoff=cutoff, ignore_water=ignore_water
    )

    coords_list, charges_list = [], []
    resname_list, chain_list, resseq_list, icode_list, gid_list = [], [], [], [], []

    for gid, entries in interface.items():
        for (_other_chain, residue) in entries:
            resname = residue.get_resname().upper()
            if resname not in NONPOLAR_SET:
                continue

            centroid = _residue_centroid(residue, heavy_only=heavy_only)
            if centroid is None:
                continue

            chain_id = residue.get_parent().id
            _, resseq, icode = residue.id
            icode = icode.strip() if isinstance(icode, str) else ""

            coords_list.append(centroid)
            charges_list.append(0)  # non-polar → neutral
            resname_list.append(resname)
            chain_list.append(chain_id)
            resseq_list.append(int(resseq))
            icode_list.append(icode)
            gid_list.append(int(gid))

    result = (
        _empty_result()
        if not coords_list
        else {
            "coords": torch.tensor(coords_list, dtype=torch.float32),
            "charge": torch.tensor(charges_list, dtype=torch.long),
            "resname": resname_list,
            "chain": chain_list,
            "resseq": resseq_list,
            "icode": icode_list,
            "group_id": gid_list,
        }
    )

    if save_pt:
        torch.save(result, save_pt)
    return result

def _collect_pdbs(root: Path, patterns: list[str]) -> list[Path]:
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    return sorted(set(files))

def main():
    ap = argparse.ArgumentParser(
        description="Extract charged and non-polar interface residue coordinates for up to the first N PDB files in a directory."
    )
    ap.add_argument("--dir", required=True, help="Directory containing PDB files.")
    ap.add_argument("--out-dir", required=True, help="Directory where .pt files will be written.")
    ap.add_argument(
        "--limit", type=int, default=1000,
        help="Maximum number of files to process (default: 1000).",
    )
    ap.add_argument(
        "--patterns", default="*.pdb,*.ent.pdb",
        help="Comma-separated glob patterns to match PDB files (default: '*.pdb,*.ent.pdb').",
    )
    ap.add_argument(
        "--cutoff", type=float, default=5.0,
        help="Interface distance cutoff in Å (default: 5.0).",
    )
    ap.add_argument(
        "--ignore-water", action="store_true", default=True,
        help="Ignore waters when finding interfaces (default: True).",
    )
    ap.add_argument(
        "--no-ignore-water", dest="ignore_water", action="store_false",
        help="Consider waters when finding interfaces.",
    )
    ap.add_argument(
        "--heavy-only", action="store_true",
        help="Use only heavy atoms for centroid calculation.",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .pt outputs if present.",
    )
    args = ap.parse_args()

    root = Path(args.dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not root.is_dir():
        print(f"[error] --dir does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    pdb_paths = _collect_pdbs(root, patterns)
    if not pdb_paths:
        print(f"[warn] No files found under {root} matching patterns: {patterns}", file=sys.stderr)
        sys.exit(0)

    pdb_paths = pdb_paths[: max(0, args.limit)]

    processed = skipped = failed = 0

    for i, pdb_path in enumerate(pdb_paths, 1):
        stem = pdb_path.stem
        if stem.endswith(".ent"):
            stem = stem[:-4]

        out_pt_charged = out_dir / f"{stem}_charged.pt"
        out_pt_nonpolar = out_dir / f"{stem}_nonpolar.pt"

        if (out_pt_charged.exists() and out_pt_nonpolar.exists()) and not args.overwrite:
            print(f"[skip] ({i}/{len(pdb_paths)}) {pdb_path.name} -> exists: {out_pt_charged.name}, {out_pt_nonpolar.name}")
            skipped += 1
            continue

        try:
            data_ch = extract_interface_charged(
                str(pdb_path),
                cutoff=args.cutoff,
                ignore_water=args.ignore_water,
                heavy_only=args.heavy_only,
                save_pt=str(out_pt_charged),
            )
            n_ch = int(data_ch["coords"].shape[0]) if isinstance(data_ch.get("coords"), torch.Tensor) else 0

            data_np = extract_interface_nonpolar(
                str(pdb_path),
                cutoff=args.cutoff,
                ignore_water=args.ignore_water,
                heavy_only=args.heavy_only,
                save_pt=str(out_pt_nonpolar),
            )
            n_np = int(data_np["coords"].shape[0]) if isinstance(data_np.get("coords"), torch.Tensor) else 0

            print(f"[ok]   ({i}/{len(pdb_paths)}) {pdb_path.name} -> {out_pt_charged.name} (charged={n_ch}), {out_pt_nonpolar.name} (nonpolar={n_np})")
            processed += 1
        except Exception as e:
            print(f"[FAIL] ({i}/{len(pdb_paths)}) {pdb_path.name} -> error: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone. processed={processed}, skipped={skipped}, failed={failed}, total_considered={len(pdb_paths)}")

if __name__ == "__main__":
    main()
