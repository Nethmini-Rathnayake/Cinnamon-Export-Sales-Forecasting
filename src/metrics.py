"""Evaluation metrics for intermittent-demand forecasting.

Metrics
-------
RMSE     : root mean squared error
MAE      : mean absolute error
Bias     : mean signed error (positive = over-forecast)
WMAPE    : sum(|y - ŷ|) / sum(y)  — volume-weighted; returns None when sum(y) == 0
MASE     : mean(|y - ŷ| / d)  where d = per-series naive scaling denominator
RMSSE    : sqrt(mean((y - ŷ)² / d²))

MASE and RMSSE denominators are computed from the in-sample (training) period
only — never from the validation or test windows.  Series with a zero
denominator (all-zero or constant history) are excluded from scaled-error
averages; the count of excluded rows is reported as `n_excl_scaled`.

Plain MAPE is never computed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Denominator ────────────────────────────────────────────────────────────────

def naive_denominators(
    wp: pd.DataFrame,
    train_cutoff: pd.Timestamp,
    qty_col: str = "qty",
) -> pd.Series:
    """Per-product mean |y[t] − y[t−1]| on the training window.

    Parameters
    ----------
    wp           : weekly-product panel (columns: ProductID, week_start, qty_col)
    train_cutoff : exclusive upper bound — only weeks *before* this date are used
    qty_col      : name of the demand column in wp

    Returns
    -------
    pd.Series indexed by ProductID.
    Products with denominator == 0 (constant or all-zero training history) are
    returned as NaN so downstream code can exclude them from scaled metrics.
    """
    wp = wp.copy()
    wp["week_start"] = pd.to_datetime(wp["week_start"])
    train = (
        wp[wp["week_start"] < train_cutoff]
        .sort_values(["ProductID", "week_start"])
    )
    train = train.copy()
    train["_naive_err"] = train.groupby("ProductID")[qty_col].diff().abs()
    denom = train.groupby("ProductID")["_naive_err"].mean()
    return denom.replace(0.0, np.nan).rename("denom")


# ── Core metric functions ──────────────────────────────────────────────────────

def _safe_scaled(
    abs_err: np.ndarray,
    denom: np.ndarray,
    squared: bool = False,
) -> tuple[float, int]:
    """Compute MASE (squared=False) or RMSSE numerator (squared=True) excluding
    rows where denom is NaN or zero.  Returns (value, n_excluded)."""
    valid = np.isfinite(denom) & (denom > 0)
    n_excl = int((~valid).sum())
    if valid.sum() == 0:
        return np.nan, n_excl
    scaled = abs_err[valid] / denom[valid]
    if squared:
        return float(np.sqrt((scaled ** 2).mean())), n_excl
    return float(scaled.mean()), n_excl


def score_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_denoms: np.ndarray | None = None,
) -> dict:
    """Compute all metrics for one model's predictions.

    Parameters
    ----------
    y_true      : 1-D array of actual demand values
    y_pred      : 1-D array of forecast values
    row_denoms  : 1-D array of per-row scaling denominators (one per row,
                  already mapped from the per-product denom series).
                  NaN / 0 entries are excluded from MASE and RMSSE.

    Returns
    -------
    dict with keys: RMSE, MAE, Bias, WMAPE, MASE, RMSSE, n_excl_scaled.
    WMAPE is None when sum(y_true) == 0.
    MASE / RMSSE are None when row_denoms is None or all rows are excluded.
    """
    y = np.asarray(y_true, dtype=float)
    ŷ = np.asarray(y_pred, dtype=float)
    err     = y - ŷ
    abs_err = np.abs(err)

    rmse = float(np.sqrt((err ** 2).mean()))
    mae  = float(abs_err.mean())
    bias = float((-err).mean())          # positive = over-forecast

    y_sum = float(y.sum())
    wmape = float(abs_err.sum() / y_sum) if y_sum > 0 else None

    if row_denoms is not None:
        d = np.asarray(row_denoms, dtype=float)
        mase_val,  n_excl = _safe_scaled(abs_err, d, squared=False)
        rmsse_val, _      = _safe_scaled(abs_err, d, squared=True)
    else:
        mase_val = rmsse_val = None
        n_excl = 0

    return {
        "RMSE":          rmse,
        "MAE":           mae,
        "Bias":          bias,
        "WMAPE":         wmape,
        "MASE":          mase_val,
        "RMSSE":         rmsse_val,
        "n_excl_scaled": n_excl,
    }


# ── Batch scoring ──────────────────────────────────────────────────────────────

def score_all_models(
    df: pd.DataFrame,
    model_cols: list[str],
    denom_col: str = "denom",
    y_true_col: str = "y_true",
) -> pd.DataFrame:
    """Score multiple models on df. Returns a (model × metric) DataFrame.

    Parameters
    ----------
    df         : prediction DataFrame; must contain y_true_col, denom_col,
                 and each column in model_cols
    model_cols : list of model name columns (e.g. ['LightGBM', 'Naive', ...])
    denom_col  : column with per-row scaling denominators (NaN = excluded)
    y_true_col : column with ground-truth demand

    Returns
    -------
    pd.DataFrame  — rows = models, columns = RMSE, MAE, Bias, WMAPE, MASE,
                    RMSSE, n_excl_scaled
    """
    row_denoms = df[denom_col].values if denom_col in df.columns else None
    results: dict[str, dict] = {}
    for m in model_cols:
        if m not in df.columns:
            continue
        results[m] = score_model(
            df[y_true_col].values,
            df[m].values,
            row_denoms=row_denoms,
        )
    return pd.DataFrame(results).T
