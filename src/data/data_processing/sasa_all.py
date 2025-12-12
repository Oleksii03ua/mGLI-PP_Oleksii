#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute total SASA for all PDBs in a directory and append results to a TSV.
Robust to large structures:
  - dynamic n_points for Shrake–Rupley
  - per-PDB timeout
  - protein-only default (skip waters/ligands)
  - resume/skip from existing output
  - immediate flush per row

Usage (defaults are good):
  python sasa_all.py
  python sasa_all.py --include-hetatm            # include waters/ligands
  python sasa_all.py --timeout 600               # 10 min per PDB
  python sasa_all.py --min-npoints 480           # coarser grid overall
"""

from __future__ import annotations
import os
import re
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa

# --------- PATHS (edit only if needed) ----------
PDB_DIR = Path("/home/op98/protein_design/dataset/PP/PP2")
GLOB    = "*.ent.pdb"

# IMPORTANT: job writes to /vast (as seen in your logs)
OUT_TXT = Path("/vast/palmer/home.mccleary/op98/protein_design/mGLI-pp_Oleksii/src/data/data_processing/all_pdb_sasa.txt")

# --------- Helpers ----------
def norm_id(p: Path) -> str:
    m = re.search(r"([0-9A-Za-z]{4})", p.name)
    return (m.group(1) if m else p.stem).lower()

def load_done_ids(out: Path) -> Set[str]:
    done: Set[str] = set()
    if not out.exists():
        return done
    with out.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("pdb_id"):  # header
                continue
            pid = line.split("\t", 1)[0].strip().lower()
            if pid:
                done.add(pid)
    return done

class Timeout(Exception):
    pass

def time_limit(seconds: int):
    """Context manager: raise Timeout after `seconds`."""
    class _TL:
        def __init__(self, secs: int):
            self.secs = int(max(1, secs))
            self._old = None
        def __enter__(self):
            def _handler(signum, frame):
                raise Timeout()
            self._old = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(self.secs)
        def __exit__(self, exc_type, exc, tb):
            signal.alarm(0)
            if self._old is not None:
                signal.signal(signal.SIGALRM, self._old)
            return False
    return _TL(seconds)

def count_atoms_fast(pdb_path: Path) -> int:
    n = 0
    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                n += 1
    return n

def dynamic_n_points(n_atoms: int, min_npoints: int = 960) -> int:
    """
    Choose Shrake–Rupley n_points based on size. Start from `min_npoints`,
    throttle as structures get large.
    """
    # enforce lower bound
    base = max(120, min_npoints)
    if n_atoms >= 300_000: return max(240, base // 4)
    if n_atoms >= 150_000: return max(360, base // 3)
    if n_atoms >=  80_000: return max(480, base // 2)
    return base

def protein_only_inplace(struct) -> None:
    """Remove non-standard amino acid residues (waters/ligands) in place."""
    # typical PDBs have model 0; Biopython will create it if absent
    model = struct[0]
    for chain in list(model):
        for res in list(chain):
            if not is_aa(res, standard=True):  # strip non-protein
                chain.detach_child(res.id)

def write_row(out_path: Path, pid: str, pdb_path: Path, sasa: float) -> None:
    with out_path.open("a") as f:
        f.write(f"{pid}\t{pdb_path}\t{sasa:.2f}\n")
        f.flush()
        os.fsync(f.fileno())  # make visible immediately
    print(f"[WRITE] {pid}\t{sasa:.2f}", flush=True)

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Compute total SASA for PDB directory (robust).")
    ap.add_argument("--include-hetatm", action="store_true",
                    help="Include waters/ligands (default: protein-only).")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-PDB timeout in seconds (default: 300 = 5 min).")
    ap.add_argument("--min-npoints", type=int, default=960,
                    help="Baseline Shrake–Rupley n_points (default: 960).")
    ap.add_argument("--probe-radius", type=float, default=1.4,
                    help="Probe radius in Å (default: 1.4).")
    args = ap.parse_args()

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT_TXT.exists():
        OUT_TXT.write_text("pdb_id\tfile\ttotal_sasa_A2\n")

    done = load_done_ids(OUT_TXT)
    parser = PDBParser(QUIET=1)
    paths = sorted(PDB_DIR.glob(GLOB))
    total = len(paths)

    print(f"[INFO] Writing to: {OUT_TXT.resolve()}")
    print(f"[INFO] Found {total} PDB files in {PDB_DIR}")
    print(f"[INFO] Already have {len(done)} rows (will skip).")
    print(f"[INFO] Mode: {'INCLUDE HETATM' if args.include_hetatm else 'PROTEIN-ONLY'}; "
          f"timeout={args.timeout}s; min_npoints={args.min_npoints}", flush=True)

    for i, pdb_path in enumerate(paths, 1):
        pid = norm_id(pdb_path)
        if pid in done:
            print(f"[{i}/{total}] {pdb_path.name}  (skip)")
            continue

        print(f"[{i}/{total}] {pdb_path.name}", flush=True)

        # Pick resolution based on size
        try:
            n_atoms = count_atoms_fast(pdb_path)
        except Exception as e:
            print(f"[WARN] {pid}: failed to count atoms: {e} — skipping", flush=True)
            continue
        n_pts = dynamic_n_points(n_atoms, args.min_npoints)

        try:
            with time_limit(args.timeout):
                struct = parser.get_structure(pid, str(pdb_path))
                if not args.include_hetatm:
                    protein_only_inplace(struct)
                sr = ShrakeRupley(probe_radius=args.probe_radius, n_points=n_pts)
                sr.compute(struct, level="S")
                total_sasa = float(getattr(struct, "sasa", float("nan")))
                if not np.isfinite(total_sasa):
                    raise ValueError("NaN/Inf SASA")
                write_row(OUT_TXT, pid, pdb_path, total_sasa)
                done.add(pid)  # update in-memory set so re-entries skip if loop continues
        except Timeout:
            print(f"[WARN] Timeout on {pid} (atoms={n_atoms}, n_points={n_pts}); skipping", flush=True)
        except Exception as e:
            print(f"[WARN] Failed {pid}: {e}", flush=True)

    print("[INFO] Done.", flush=True)

if __name__ == "__main__":
    # Ensure signals work only on POSIX
    if os.name != "posix":
        print("[WARN] Timeout uses POSIX signals; on non-POSIX, it may be ignored.", flush=True)
    main()

