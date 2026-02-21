from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .models.gbdt import GBDTRegressor


logger = logging.getLogger(__name__)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _as_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df[list(cols)].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _as_numeric_series(df: pd.DataFrame, col: str) -> np.ndarray:
    y = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    return y


@dataclass(frozen=True)
class TrainResult:
    model_ckpt_paths: Dict[str, str]
    training_info: Dict[str, Any]


def train_models(
    *,
    cfg: Dict[str, Any],
    run_dir: Path,
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    train_dates: Sequence,
    val_dates: Sequence,
) -> TrainResult:
    seed = int(cfg.get("run", {}).get("seed", 42))
    set_global_seeds(seed)
    horizon = int((cfg.get("labels") or {}).get("horizon_trading_days", 20))
    target_col = str((cfg.get("labels") or {}).get("target_col", f"future_return_{horizon}d"))

    models_cfg = cfg.get("models", {}) or {}
    out_dir = run_dir / "model_ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = merged[merged["decision_date"].isin(train_dates)]
    val_df = merged[merged["decision_date"].isin(val_dates)]

    X_train_df = _as_numeric_frame(train_df, feature_cols)
    X_val_df = _as_numeric_frame(val_df, feature_cols)
    y_train_raw = _as_numeric_series(train_df, target_col)
    y_val_raw = _as_numeric_series(val_df, target_col)
    y_train_ok = np.isfinite(y_train_raw)
    y_val_ok = np.isfinite(y_val_raw)

    min_finite_ratio = float((models_cfg.get("gbdt") or {}).get("min_finite_ratio", 0.10))
    finite_train = X_train_df[y_train_ok].notna().mean() if y_train_ok.any() else pd.Series(0.0, index=X_train_df.columns)
    finite_val = X_val_df[y_val_ok].notna().mean() if y_val_ok.any() else pd.Series(0.0, index=X_val_df.columns)
    selected_feature_cols = [
        c for c in feature_cols if float(finite_train.get(c, 0.0)) >= min_finite_ratio and float(finite_val.get(c, 0.0)) >= min_finite_ratio
    ]
    dropped_feature_cols = [c for c in feature_cols if c not in selected_feature_cols]
    if not selected_feature_cols:
        worst = (finite_train.fillna(0.0) + finite_val.fillna(0.0)).sort_values().head(8).to_dict()
        raise RuntimeError(
            f"No usable features after finite-ratio filter min_finite_ratio={min_finite_ratio} worst={worst}"
        )

    X_train_df = X_train_df[selected_feature_cols]
    X_val_df = X_val_df[selected_feature_cols]

    fill_values = X_train_df[y_train_ok].median(numeric_only=True).fillna(0.0)
    X_train_arr = X_train_df.fillna(fill_values).to_numpy(dtype=float)
    X_val_arr = X_val_df.fillna(fill_values).to_numpy(dtype=float)
    y_train_arr = y_train_raw
    y_val_arr = y_val_raw

    train_keep = y_train_ok & np.isfinite(X_train_arr).all(axis=1)
    val_keep = y_val_ok & np.isfinite(X_val_arr).all(axis=1)

    X_train, y_train = X_train_arr[train_keep], y_train_arr[train_keep]
    X_val, y_val = X_val_arr[val_keep], y_val_arr[val_keep]
    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError(f"Insufficient train/val rows after filtering finite values train={len(X_train)} val={len(X_val)}")

    if len(X_train) < 1000 or len(X_val) < 200:
        logger.warning(f"small_train_or_val_samples train={len(X_train)} val={len(X_val)}")

    ckpts: Dict[str, str] = {}
    info: Dict[str, Any] = {
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_features": int(len(selected_feature_cols)),
        "feature_cols": list(selected_feature_cols),
        "dropped_feature_cols": dropped_feature_cols,
        "feature_fill_values": {k: float(v) for k, v in fill_values.to_dict().items()},
        "feature_filter": {
            "min_finite_ratio": min_finite_ratio,
            "finite_train": {k: float(v) for k, v in finite_train.to_dict().items()},
            "finite_val": {k: float(v) for k, v in finite_val.to_dict().items()},
        },
        "target_col": target_col,
    }
    if not bool(models_cfg.get("gbdt", {}).get("enabled", True)):
        raise RuntimeError("models.gbdt.enabled must be true for this agent")

    mcfg = models_cfg.get("gbdt", {}) or {}
    framework = str(mcfg.get("framework", "lightgbm"))
    params = mcfg.get("params", {}) or {}
    prefer_gpu = bool(mcfg.get("prefer_gpu", True))
    model = GBDTRegressor(seed=seed, framework=framework, params=params, prefer_gpu=prefer_gpu)
    model.fit(X_train, y_train, X_val, y_val)
    yhat_val = model.predict(X_val)
    val_mse = float(np.mean((yhat_val - y_val) ** 2))
    if model.framework == "lightgbm":
        path = out_dir / "gbdt_model.txt"
    elif model.framework == "xgboost":
        path = out_dir / "gbdt_model.json"
    else:
        path = out_dir / "gbdt_model.pkl"
    model.save(path)
    ckpts["gbdt"] = str(path)
    info["gbdt"] = {
        "framework": model.framework,
        "prefer_gpu": prefer_gpu,
        "gpu_attempted": bool(model.gpu_attempted),
        "accelerator": model.train_accelerator,
        "params": params,
        "effective_params": model.trained_params or params,
        "val_mse": val_mse,
    }

    return TrainResult(model_ckpt_paths=ckpts, training_info=info)
