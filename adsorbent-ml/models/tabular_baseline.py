"""N2-v1 tabular baseline: the mandatory floor GNNs must beat (ROADMAP).

One ``HistGradientBoostingRegressor`` per target (NaN-native — structural
missingness needs no imputation theater):

- ``log_q_sat``, ``log_E``: all rows. ``n_da``: linear, all rows.
- ``log_Q_st``: multi-T subset only (single-T rows carry no isosteric heat).

Honest estimate = out-of-fold predictions under GroupKFold by family
(random splits leak near-duplicate frameworks — ROADMAP data strategy).
Small-data guards: shallow trees, few iterations, low LR. Deterministic
(``random_state`` fixed). Bundle = pickle {models, feature_names,
target_meta}; report dict carries OOF MAE/RMSE/Spearman per target.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

TARGETS: dict[str, dict] = {
    "log_q_sat": {"label_col": "log_q_sat", "back": "exp", "subset": None},
    "log_Q_st": {"label_col": "log_Q_st", "back": "exp", "subset": "q_st_j_kg"},
    "log_E": {"label_col": "log_E", "back": "exp", "subset": None},
    "n_da": {"label_col": "n_da", "back": "linear", "subset": None},
}


@dataclass
class BaselineConfig:
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 5
    n_splits: int = 4
    random_state: int = 0


def _regressor(cfg: BaselineConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.max_leaf_nodes, min_samples_leaf=cfg.min_samples_leaf,
        early_stopping=False, random_state=cfg.random_state)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    dummy = np.median(y_true)
    rho = float(spearmanr(y_true, y_pred).statistic)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "dummy_rmse": float(np.sqrt(np.mean((y_true - dummy) ** 2))),
            "spearman": rho if np.isfinite(rho) else 0.0, "n": int(len(y_true))}


def cross_validate(X: pd.DataFrame, labels: pd.DataFrame,
                   feature_names: list[str],
                   cfg: BaselineConfig) -> tuple[dict[str, np.ndarray], dict]:
    """GroupKFold OOF predictions (natural scale) + metrics per target."""
    groups = labels["family"].to_numpy()
    n_splits = min(cfg.n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("need ≥ 2 families for grouped CV")
    oof: dict[str, np.ndarray] = {}
    report: dict = {}
    for target, spec in TARGETS.items():
        col = spec["label_col"]
        mask = labels[col].notna().to_numpy()
        if spec["subset"] is not None:
            mask &= labels[spec["subset"]].notna().to_numpy()
        idx = np.where(mask)[0]
        pred = np.full(len(labels), np.nan)
        for tr, va in GroupKFold(n_splits=n_splits).split(idx, groups=groups[idx]):
            model = _regressor(cfg)
            model.fit(X.iloc[idx[tr]][feature_names].to_numpy(),
                      labels.iloc[idx[tr]][col].to_numpy())
            pred[idx[va]] = model.predict(X.iloc[idx[va]][feature_names].to_numpy())
        if spec["back"] == "exp":
            yt, yp = np.exp(labels.iloc[idx][col].to_numpy()), np.exp(pred[idx])
        else:
            yt, yp = labels.iloc[idx][col].to_numpy(), pred[idx]
        oof[target] = pred
        report[target] = _metrics(yt, yp)
    return oof, report


def fit_full(X: pd.DataFrame, labels: pd.DataFrame, feature_names: list[str],
             cfg: BaselineConfig) -> dict:
    """Refit one model per target on all labeled rows (post-CV artifact)."""
    models = {}
    for target, spec in TARGETS.items():
        col = spec["label_col"]
        mask = labels[col].notna()
        if spec["subset"] is not None:
            mask &= labels[spec["subset"]].notna()
        model = _regressor(cfg)
        model.fit(X.loc[mask, feature_names].to_numpy(), labels.loc[mask, col].to_numpy())
        models[target] = model
    return {"models": models, "feature_names": feature_names,
            "targets": TARGETS, "config": asdict(cfg), "created_unix": time.time()}


def predict(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Natural-scale predictions (inverse log where trained in log space).

    Columns are reindexed to the training schema (missing → 0, the HGB
    missing-value path) so inference-time topology/family drift degrades
    gracefully instead of raising.
    """
    Xa = X.reindex(columns=bundle["feature_names"], fill_value=0.0)
    out = pd.DataFrame(index=X.index)
    for target, spec in bundle["targets"].items():
        raw = bundle["models"][target].predict(Xa.to_numpy(dtype=float))
        out[target.replace("log_", "")] = np.exp(raw) if spec["back"] == "exp" else raw
    return out


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)


def load_bundle(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


__all__ = ["BaselineConfig", "TARGETS", "cross_validate", "fit_full",
           "load_bundle", "predict", "save_bundle"]
