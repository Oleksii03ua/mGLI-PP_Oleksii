# src/train/train.py
import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error

from src.models.models import MLP, GBDT
from src.data.data_processing.loader import load_features_labels  

def cv_score(model_cls, model_kwargs, X, y, cv=5, is_torch=False, train_kwargs=None):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if is_torch:
            # --- train PyTorch MLP ---
            model = model_cls(**model_kwargs)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=train_kwargs["lr"])
            criterion = torch.nn.MSELoss()
            X_tr_t = torch.from_numpy(X_tr).float()
            y_tr_t = torch.from_numpy(y_tr).float().unsqueeze(1)
            for epoch in range(train_kwargs["epochs"]):
                optimizer.zero_grad()
                preds = model(X_tr_t)
                loss = criterion(preds, y_tr_t)
                loss.backward()
                optimizer.step()
            # eval
            model.eval()
            preds_va = model(torch.from_numpy(X_va).float()).detach().numpy().squeeze()
        else:
            # --- train sklearn GBDT ---
            model = model_cls(**model_kwargs)
            model.fit(X_tr, y_tr)
            preds_va = model.predict(X_va)

        mses.append(mean_squared_error(y_va, preds_va))
    return np.mean(mses)

def grid_search(model_cls, param_grid, X, y, **cv_kwargs):
    best_score, best_params = float("inf"), None
    for params in ParameterGrid(param_grid):
        score = cv_score(model_cls, params, X, y, **cv_kwargs)
        print(f"Params {params} → MSE {score:.4f}")
        if score < best_score:
            best_score, best_params = score, params
    return best_params, best_score

def main(args):
    X, y = load_features_labels(args.features, args.labels)

    # --- tune MLP ---
    mlp_grid = {
        "input_dim": [X.shape[1]],
        "hidden_dim": [64, 128],
        "output_dim": [1],
    }
    train_kwargs = {"lr": args.lr, "epochs": args.epochs}
    best_mlp_params, mlp_score = grid_search(
        MLP, mlp_grid, X, y,
        cv=args.cv,
        is_torch=True,
        train_kwargs=train_kwargs
    )
    print("BEST MLP:", best_mlp_params, "MSE:", mlp_score)

    # --- tune GBDT ---
    gbdt_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }
    best_gbdt_params, gbdt_score = grid_search(
        GBDT, gbdt_grid, X, y,
        cv=args.cv,
        is_torch=False
    )
    print("BEST GBDT:", best_gbdt_params, "MSE:", gbdt_score)

    # --- train final on all data & save ---
    final_mlp = MLP(**best_mlp_params)
    # ... train final_mlp on full X, y as above ...
    torch.save(final_mlp.state_dict(), os.path.join(args.out_dir, "best_mlp.pt"))

    final_gbdt = GBDT(**best_gbdt_params)
    final_gbdt.fit(X, y)
    # you could pickle it:
    import pickle
    with open(os.path.join(args.out_dir, "best_gbdt.pkl"), "wb") as f:
        pickle.dump(final_gbdt, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True,
                   help="Processed feature embeddings in .pt")
    p.add_argument("--labels",   required=True,
                   help="TSV with columns ['pdb_id','affinity']")
    p.add_argument("--out_dir",      default="results", help="Where to store models")
    p.add_argument("--cv",           type=int, default=5)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--epochs",       type=int,   default=20)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
