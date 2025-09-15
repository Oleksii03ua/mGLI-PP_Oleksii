#!/usr/bin/env python3
# single_vector.py

import os, sys, re, math, glob, argparse
import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Optional speedup for nearest-neighbor queries
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -------------------------- filename/id helpers --------------------------

def pt_to_pdbid(path: str) -> str:
    """
    Convert a .pt filename to the TSV pdb_id:
      '1fc2_charged.pt' -> '1fc2'
      '1fc2_interface_charged.pt' -> '1fc2'
      '1FC2.pt' -> '1fc2'
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    # safest: keep only the part before the first underscore
    stem = stem.split("_", 1)[0]
    return stem.strip().lower()


# -------------------------- labels reader --------------------------

def read_labels(labels_path: str) -> pd.DataFrame:
    """
    Read a TSV with columns: pdb_id, delta_G (exact headers).
    - Lowercase pdb_id for robust matching.
    """
    df = pd.read_csv(labels_path, sep="\t")
    missing = [c for c in ("pdb_id", "delta_G") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV/TSV missing required columns: {missing}")
    df = df.copy()
    df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
    return df.set_index("pdb_id")


# -------------------------- feature construction --------------------------

def _safe_scale(coords: np.ndarray):
    """Center & robustly scale (translation/size invariance)."""
    mu = coords.mean(axis=0)
    X = coords - mu
    r = np.linalg.norm(X, axis=1)
    s = np.median(r)
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return X / s, s, mu

def _canonicalize(X: np.ndarray, use_pca: bool = True) -> np.ndarray:
    """Rotate to canonical frame using PCA (orientation invariance)."""
    if not use_pca:
        return X
    pca = PCA(n_components=3, random_state=42)
    return pca.fit_transform(X)

def _pairwise_min_dist(A: np.ndarray, B: np.ndarray):
    """Return (global min distance, per-point nearest distances) from A to B."""
    if len(A) == 0 or len(B) == 0:
        return math.inf, np.array([])
    if HAS_SCIPY:
        tree = cKDTree(B)
        dmin, _ = tree.query(A, k=1)
        return float(np.min(dmin)), dmin
    # numpy fallback
    diff = A[:, None, :] - B[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dmin = dist.min(axis=1)
    return float(dmin.min()), dmin

def _counts_from_resnames(rn: np.ndarray):
    return (
        int((rn == "ARG").sum()),
        int((rn == "LYS").sum()),
        int((rn == "ASP").sum()),
        int((rn == "GLU").sum()),
    )

def compute_complex_features(d: dict, sb_cutoff: float = 5.0, use_pca: bool = True):
    """
    Input dict from interface_charged.pt:
      - coords: (N,3) float tensor/ndarray
      - charge: (N,)  int tensor/ndarray (+1/-1)
      - resname: list[str]
      - group_id: list[int]
    Returns: (features np.ndarray (D,), feature_names list[str])
    """
    C = d["coords"].cpu().numpy() if torch.is_tensor(d["coords"]) else np.asarray(d["coords"], dtype=float)
    q = d["charge"].cpu().numpy() if torch.is_tensor(d["charge"]) else np.asarray(d["charge"], dtype=int)
    rn = np.asarray(d["resname"])
    gid = np.asarray(d["group_id"], dtype=int)

    N = C.shape[0]
    names = [
        "N", "n_pos", "n_neg", "net_charge",
        "n_ARG", "n_LYS", "n_ASP", "n_GLU",
        "std_x", "std_y", "std_z",
        "spread_trace", "anisotropy_ratio",
        "min_pos_neg", "count_pos_neg_leq_cutoff",
        "mean_nn_dist_all",
        "n_group0", "n_group1", "group_balance",
    ]
    if N == 0:
        return np.zeros(len(names), dtype=float), names

    # normalize
    X, scale, _ = _safe_scale(C)
    Xp = _canonicalize(X, use_pca=use_pca)

    # composition
    n_pos = int((q == +1).sum())
    n_neg = int((q == -1).sum())
    net_q = int(q.sum())
    n_ARG, n_LYS, n_ASP, n_GLU = _counts_from_resnames(rn)

    # geometry stats
    std_xyz = Xp.std(axis=0)
    cov = np.cov(Xp.T)
    spread_trace = float(np.trace(cov))
    try:
        w = np.linalg.eigvalsh(cov)
        w = np.clip(w, 1e-8, None)
        anisotropy = float(w.max() / w.min())
    except Exception:
        anisotropy = 1.0

    # coarse electrostatics features
    pos = Xp[q == +1]
    neg = Xp[q == -1]
    min_pos_neg, dmin_pos = _pairwise_min_dist(pos, neg)
    count_sb = int((dmin_pos <= (sb_cutoff / scale)).sum()) if dmin_pos.size else 0

    # interface density proxy
    if N >= 2:
        if HAS_SCIPY:
            tree_all = cKDTree(Xp)
            d_nn, _ = tree_all.query(Xp, k=2)  # self at k=1
            mean_nn = float(np.mean(d_nn[:, 1]))
        else:
            diff = Xp[:, None, :] - Xp[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            mean_nn = float(dist.min(axis=1).mean())
    else:
        mean_nn = float("inf")

    # group balance (dominant two groups)
    unique, counts = np.unique(gid, return_counts=True)
    order = np.argsort(-counts)
    g0 = int(counts[order[0]]) if len(counts) >= 1 else 0
    g1 = int(counts[order[1]]) if len(counts) >= 2 else 0
    balance = float(abs(g0 - g1) / max(1, (g0 + g1)))

    feats = np.array([
        float(N), float(n_pos), float(n_neg), float(net_q),
        float(n_ARG), float(n_LYS), float(n_ASP), float(n_GLU),
        float(std_xyz[0]), float(std_xyz[1]), float(std_xyz[2]),
        float(spread_trace), float(anisotropy),
        float(min_pos_neg if np.isfinite(min_pos_neg) else 1e9),
        float(count_sb),
        float(mean_nn),
        float(g0), float(g1), float(balance),
    ], dtype=float)
    return feats, names


# -------------------------- main CLI --------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build complex-level features from interface_charged.pt files and run t-SNE with ΔG."
    )
    ap.add_argument("pt_dir", help="Directory containing .pt files (e.g., 1fc2_charged.pt).")
    ap.add_argument("labels_tsv", help="TSV with columns: pdb_id, delta_G.")
    ap.add_argument("out_dir", help="Directory to write outputs.")
    ap.add_argument("--perplexity", type=float, default=15.0, help="t-SNE perplexity.")
    ap.add_argument("--sb-cutoff", type=float, default=5.0, help="Salt-bridge distance cutoff in Å.")
    ap.add_argument("--no-pca", action="store_true", help="Disable PCA canonicalization.")
    ap.add_argument("--convert-kj", action="store_true",
                    help="Convert delta_G from kJ/mol to kcal/mol (×0.239005736).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect .pt files
    pt_files = sorted(glob.glob(os.path.join(args.pt_dir, "*.pt")))
    if not pt_files:
        print(f"[ERROR] No .pt files found in {args.pt_dir}", file=sys.stderr)
        sys.exit(2)
    print(f"[INFO] Found {len(pt_files)} .pt files in {args.pt_dir}.")

    # Read labels
    df_labels = read_labels(args.labels_tsv)
    if args.convert_kj:
        df_labels["delta_G"] = df_labels["delta_G"] * 0.239005736  # kJ/mol → kcal/mol

    # Build dataset
    feats = []
    ids = []
    feat_names = None
    for p in pt_files:
        try:
            # Safe load if available (PyTorch >= 2.4); fallback to default
            try:
                d = torch.load(p, weights_only=True)
            except TypeError:
                d = torch.load(p)
        except Exception as e:
            print(f"[WARN] Skipping {p}: cannot load ({e})", file=sys.stderr)
            continue

        f, names = compute_complex_features(d, sb_cutoff=args.sb_cutoff, use_pca=not args.no_pca)
        feats.append(f)
        ids.append(pt_to_pdbid(p))
        if feat_names is None:
            feat_names = names

    if not feats:
        print("[ERROR] No feature vectors were built.", file=sys.stderr)
        sys.exit(2)

    X = np.vstack(feats)

    # Align ΔG by pdb_id
    deltaG = np.array([df_labels.loc[i, "delta_G"] if i in df_labels.index else np.nan for i in ids], dtype=float)
    missing = [i for i in ids if i not in df_labels.index]
    matched = len(ids) - len(missing)
    print(f"[INFO] Matched ΔG for {matched}/{len(ids)} complexes.")
    if missing:
        print(f"[WARN] Missing delta_G for {len(missing)} complexes: {', '.join(missing[:15])}" +
              (" ..." if len(missing) > 15 else ""), file=sys.stderr)

    # Save features CSV
    features_path = os.path.join(args.out_dir, "complex_features.csv")
    df_feat = pd.DataFrame(X, columns=feat_names, index=ids)
    df_feat.index.name = "pdb_id"
    df_feat["delta_G"] = deltaG
    df_feat.to_csv(features_path)
    print(f"[INFO] Wrote {features_path}")

    # t-SNE
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", learning_rate="auto", random_state=42)
    Z = tsne.fit_transform(Xs)

    # Save embedding CSV
    emb_path = os.path.join(args.out_dir, "tsne_embedding.csv")
    pd.DataFrame({"pdb_id": ids, "tsne1": Z[:, 0], "tsne2": Z[:, 1], "delta_G": deltaG}).to_csv(emb_path, index=False)
    print(f"[INFO] Wrote {emb_path}")

    # Save plot
    png_path = os.path.join(args.out_dir, "tsne.png")
    plt.figure(figsize=(6, 5))
    if np.all(np.isnan(deltaG)):
        plt.scatter(Z[:, 0], Z[:, 1], s=24, alpha=0.9)
    else:
        sc = plt.scatter(Z[:, 0], Z[:, 1], s=24, alpha=0.9, c=deltaG)
        cb = plt.colorbar(sc); cb.set_label("ΔG")
    plt.title("t-SNE of complex-level charged-interface features")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print(f"[INFO] Saved plot to {png_path}")


if __name__ == "__main__":
    main()
