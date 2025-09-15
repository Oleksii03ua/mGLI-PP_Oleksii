# src/utils/visualization.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne_embeddings(emb_dir, labels_tsv, output_png,
                         n_components=2, perplexity=30):
    # 1) load affinities
    df = pd.read_csv(labels_tsv, sep="\t")
    pdbs = df["PDB_ID"].tolist()
    affinities = df["ΔG_kJ/mol"].values

    # 2) load embeddings
    embs = []
    valid_indices = []  # Track which samples are valid
    
    for i, pdb in enumerate(pdbs):
        path = os.path.join(emb_dir, f"{pdb}_mGLI.pt")
        try:
            emb = torch.load(path)
            emb_flat = emb.flatten().numpy()
            
            # Check for NaN or infinite values
            if np.isfinite(emb_flat).all():
                embs.append(emb_flat)
                valid_indices.append(i)
            else:
                print(f"Warning: Skipping {pdb} due to NaN/infinite values")
                
        except FileNotFoundError:
            print(f"Warning: Embedding file not found for {pdb}")
            continue
    
    if len(embs) == 0:
        raise ValueError("No valid embeddings found!")
    
    X = np.vstack(embs)
    affinities = affinities[valid_indices]  # Filter affinities to match valid embeddings
    
    print(f"Loaded {len(embs)} valid embeddings out of {len(pdbs)} total")
    print(f"Embedding shape: {X.shape}")
    print(f"NaN check: {np.isnan(X).any()}")
    print(f"Infinite check: {np.isinf(X).any()}")

    # 3) run t-SNE
    tsne = TSNE(n_components=n_components,
                perplexity=min(perplexity, len(embs)//4),  # Adjust perplexity if needed
                random_state=42)
    X_emb = tsne.fit_transform(X)

    # 4) plot
    plt.figure(figsize=(8,6))
    sc = plt.scatter(X_emb[:,0], X_emb[:,1], c=affinities, cmap="viridis")
    plt.colorbar(sc, label="Binding affinity (ΔG kJ/mol)")
    plt.title("t-SNE of Topological Embeddings")
    plt.xlabel("TSNE-1"); plt.ylabel("TSNE-2")
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"t-SNE plot saved to: {output_png}")






# # src/utils/visualization.py
# import os
# import torch
# import numpy as np
# import pandas as pd
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# def plot_tsne_embeddings(emb_dir, labels_tsv, output_png,
#                          n_components=2, perplexity=30):
#     # 1) load affinities
#     df = pd.read_csv(labels_tsv, sep="\t")
#     pdbs = df["PDB_ID"].tolist()
#     affinities = df["ΔG_kJ/mol"].values

#     # 2) load embeddings
#     embs = []
#     for pdb in pdbs:
#         path = os.path.join(emb_dir, f"{pdb}_mGLI.pt")
#         emb = torch.load(path)
#         embs.append(emb.flatten().numpy())
#     X = np.vstack(embs)

#     # 3) run t-SNE
#     tsne = TSNE(n_components=n_components,
#                 perplexity=perplexity,
#                 random_state=42)
#     X_emb = tsne.fit_transform(X)

#     # 4) plot
#     plt.figure(figsize=(8,6))
#     sc = plt.scatter(X_emb[:,0], X_emb[:,1], c=affinities, cmap="viridis")
#     plt.colorbar(sc, label="Binding affinity")
#     plt.title("t-SNE of Topological Embeddings")
#     plt.xlabel("TSNE-1"); plt.ylabel("TSNE-2")
#     plt.tight_layout()
#     plt.savefig(output_png, dpi=300)
#     plt.close()
