"""Global LightGBM model — direct multi-step demand forecasting.

Trains one global model over all products and all 12 horizons.

Objective: tweedie (variance power 1.2) — natural fit for zero-inflated
count-like demand; predictions are always non-negative.
Early stopping on VALID RMSE to prevent overfitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.training_table import get_splits, FEATURES_PATH

MODEL_PATH = Path("outputs/models/lgbm_global.txt")
HIST_PATH  = Path("outputs/models/lgbm_eval_history.json")
PREDS_PATH = Path("data/processed/preds_global.parquet")

DEFAULT_PARAMS: dict = {
    "objective":               "tweedie",
    "tweedie_variance_power":  1.2,
    "metric":                  "rmse",
    "learning_rate":           0.05,
    "num_leaves":              127,
    "min_data_in_leaf":        50,
    "feature_fraction":        0.8,
    "bagging_fraction":        0.8,
    "bagging_freq":            5,
    "lambda_l1":               0.1,
    "lambda_l2":               0.1,
    "verbose":                 -1,
    "n_jobs":                  -1,
    "seed":                    42,
}


def train_global(
    train_df:    pd.DataFrame,
    valid_df:    pd.DataFrame,
    test_df:     pd.DataFrame,
    feature_cols: list[str],
    cat_cols:    list[str],
    params:      dict | None = None,
    num_boost_round:       int = 2000,
    early_stopping_rounds: int = 50,
    model_out:   Path | str = MODEL_PATH,
    hist_out:    Path | str = HIST_PATH,
    preds_out:   Path | str = PREDS_PATH,
) -> tuple[lgb.Booster, dict, pd.DataFrame]:
    """Train the global LightGBM, save model + eval history + predictions.

    Returns
    -------
    booster      : trained LightGBM Booster
    eval_history : dict with keys 'train'/'valid', each {'rmse': [float, …]}
    preds_df     : DataFrame with ProductID, target_week, horizon,
                   y_true, y_pred, split  (for VALID and TEST rows)
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    X_train = train_df[feature_cols]
    y_train = train_df["y"].values
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["y"].values

    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid, label=y_valid,
        categorical_feature=cat_cols,
        reference=dtrain,
        free_raw_data=False,
    )

    eval_history: dict = {}
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.record_evaluation(eval_history),
            lgb.log_evaluation(period=50),
        ],
    )

    # ── Persist model ──────────────────────────────────────────────────────────
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_out))

    hist_out = Path(hist_out)
    with open(hist_out, "w") as f:
        json.dump(eval_history, f)

    # ── Predictions on VALID and TEST ─────────────────────────────────────────
    y_pred_valid = booster.predict(X_valid, num_iteration=booster.best_iteration)
    y_pred_test  = booster.predict(test_df[feature_cols],
                                   num_iteration=booster.best_iteration)

    # Tweedie output is already ≥ 0; clip for safety
    y_pred_valid = np.clip(y_pred_valid, 0, None)
    y_pred_test  = np.clip(y_pred_test,  0, None)

    preds_df = pd.DataFrame({
        "ProductID":   np.concatenate([valid_df["ProductID"].values,
                                       test_df["ProductID"].values]),
        "target_week": np.concatenate([valid_df["target_week"].values,
                                       test_df["target_week"].values]),
        "horizon":     np.concatenate([valid_df["horizon"].values,
                                       test_df["horizon"].values]),
        "y_true":      np.concatenate([y_valid, test_df["y"].values]).astype("float32"),
        "y_pred":      np.concatenate([y_pred_valid, y_pred_test]).astype("float32"),
        "split":       np.concatenate([
            np.full(len(valid_df), "valid", dtype=object),
            np.full(len(test_df),  "test",  dtype=object),
        ]),
    })

    preds_out = Path(preds_out)
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_parquet(preds_out, index=False)

    return booster, eval_history, preds_df


def load_global(
    model_path: Path | str = MODEL_PATH,
    hist_path:  Path | str = HIST_PATH,
) -> tuple[lgb.Booster, dict | None]:
    """Load a saved model and eval history (history is None if file missing)."""
    booster = lgb.Booster(model_file=str(model_path))
    hist_path = Path(hist_path)
    eval_history = json.loads(hist_path.read_text()) if hist_path.exists() else None
    return booster, eval_history
