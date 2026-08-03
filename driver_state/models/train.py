"""
Training + evaluation for the sequence models.

Trains the LSTM and Transformer on windowed sessions, selects each by best
validation macro-F1, and reports test accuracy / macro-F1 / calibration (ECE).
The better of the two is marked as the serving model. Fully seeded for
reproducibility.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from ..config import Settings, settings as default_settings
from ..data.windows import build_dataset, save_feature_meta
from ..generation import generate_sessions
from ..monitoring.metrics import expected_calibration_error, reliability_bins
from ..signals import CLASS_NAMES
from . import MODELS


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loaders(ds, cfg: Settings):
    def dl(X, y, shuffle):
        t = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(t, batch_size=cfg.batch_size, shuffle=shuffle)
    return (dl(ds.X_train, ds.y_train, True),
            dl(ds.X_val, ds.y_val, False),
            dl(ds.X_test, ds.y_test, False))


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def train_one(name: str, ds, cfg: Settings, verbose: bool = True):
    _seed(cfg.seed)
    device = torch.device(cfg.device)
    model = MODELS[name]().to(device)
    train_dl, val_dl, test_dl = _loaders(ds, cfg)

    # class weights counter the natural imbalance (critical fatigue is rarer)
    counts = np.bincount(ds.y_train, minlength=len(CLASS_NAMES))
    weights = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_f1, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        vp, vy = _predict(model, val_dl, device)
        f1 = f1_score(vy, vp.argmax(1), average="macro")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}
        if verbose:
            print(f"  [{name}] epoch {epoch+1:2d}/{cfg.epochs}  val_macroF1={f1:.3f}")

    model.load_state_dict(best_state)
    tp, ty = _predict(model, test_dl, device)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(ty, tp.argmax(1))),
        "macro_f1": float(f1_score(ty, tp.argmax(1), average="macro")),
        "per_class_f1": {c: float(v) for c, v in zip(
            CLASS_NAMES, f1_score(ty, tp.argmax(1), average=None))},
        "ece": expected_calibration_error(tp, ty),
        "n_test": int(len(ty)),
    }
    return model, metrics, (tp, ty)


def train_all(cfg: Settings | None = None, verbose: bool = True):
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    if verbose:
        print("Generating sessions + windows...")
    sessions = generate_sessions(cfg)
    ds = build_dataset(sessions, cfg)
    ds.scaler.save(cfg.model_root / "scaler.npz")
    save_feature_meta(cfg)
    if verbose:
        dist = np.bincount(ds.y_train) / len(ds.y_train)
        print(f"  train windows={len(ds.y_train)} val={len(ds.y_val)} "
              f"test={len(ds.y_test)} | class mix={np.round(dist,2)}")

    results = {}
    best_name, best_f1 = None, -1.0
    for name in MODELS:
        model, metrics, preds = train_one(name, ds, cfg, verbose)
        torch.save(model.state_dict(), cfg.model_root / f"{name}.pt")
        results[name] = metrics
        if verbose:
            print(f"  [{name}] test acc={metrics['accuracy']:.3f}  "
                  f"macroF1={metrics['macro_f1']:.3f}  ECE={metrics['ece']:.3f}")
        if metrics["macro_f1"] > best_f1:
            best_f1, best_name = metrics["macro_f1"], name
            best_preds = preds

    # persist comparison + reliability of the serving model
    tp, ty = best_preds
    (cfg.model_root / "metrics.json").write_text(json.dumps({
        "models": results,
        "serving_model": best_name,
        "reliability": reliability_bins(tp, ty),
    }, indent=2))
    (cfg.model_root / "serving_model.json").write_text(
        json.dumps({"name": best_name}, indent=2))
    if verbose:
        print(f"Serving model: {best_name} (macro-F1 {best_f1:.3f})")
    return results, best_name
