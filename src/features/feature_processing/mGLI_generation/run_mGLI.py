#!/usr/bin/env python3
import os
import sys
import pickle
import torch
from mGLI import generate_mGLI_feature

if __name__=="__main__":
    pdbid = sys.argv[1]
    # generate outputs
    feats, _ = generate_mGLI_feature(pdbid)
    # create output directories
    outdir = "/home/op98/protein_design/mGLI-pp_Oleksii/src/data/data_files/pt2"
    os.makedirs(outdir, exist_ok=True)
    # save feature vector as tensor
    t = torch.tensor(feats, dtype=torch.float32)
    torch.save(t, os.path.join(outdir, f"{pdbid}_mGLI.pt"))
