"""Direct multi-step training table for the global LightGBM model.

Each origin week in the feature panel expands to N_HORIZONS rows, one per
forecast horizon.  For every (origin_week, h) pair:

  y              = qty at target_week = origin_week + h weeks
  history feats  = all columns from features_product at origin_week
                   (lags / rollings are already shifted, so origin_week
                    never leaks its own qty value)
  calendar feats = recomputed for the *target* week (not origin_week)
  horizon        = h (added as a numeric feature, int 1-12)

Time split is on target_week so there is no future information in training:

  TRAIN : target_week < valid_start
  VALID : valid_start <= target_week < test_start   (last 12+12 weeks)
  TEST  : target_week >= test_start                 (last 12 weeks, sealed)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/processed/features_product.parquet")
N_HORIZONS    = 12

_SKIP_FEAT = {"ProductID", "week_start", "qty"}   # not model features
_CAL_COLS  = ["week_of_year", "month", "quarter",
               "week_of_month", "sin_annual", "cos_annual"]


def get_splits(
    features_path: Path | str = FEATURES_PATH,
    n_horizons: int = N_HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build the direct multi-step table and return (train, valid, test,
    feature_cols, cat_cols).

    Each returned DataFrame contains:
        ProductID, week_start (origin), target_week, y,
        horizon (feature), all other feature columns
    """
    features_path = Path(features_path)
    feat = pd.read_parquet(features_path)
    feat["week_start"] = pd.to_datetime(feat["week_start"])

    panel_end   = feat["week_start"].max()
    test_start  = panel_end - pd.Timedelta(weeks=n_horizons - 1)
    valid_start = test_start - pd.Timedelta(weeks=n_horizons)

    # Fast label lookup: (ProductID, week_start) → qty
    qty_lookup = feat.set_index(["ProductID", "week_start"])["qty"]

    # Columns to keep in each chunk (drop qty; ProductID+week_start kept for ID)
    keep_base = ["ProductID", "week_start"] + [
        c for c in feat.columns if c not in _SKIP_FEAT
    ]
    feat_slim = feat[keep_base]  # view, no copy

    # Feature column names for the model (including horizon, added below)
    feature_cols = [c for c in keep_base if c not in ("ProductID", "week_start")] + ["horizon"]
    cat_cols     = [c for c in feature_cols if c != "horizon" and
                    isinstance(feat[c].dtype if c in feat.columns else pd.CategoricalDtype(),
                               pd.CategoricalDtype)]

    frames: list[pd.DataFrame] = []

    for h in range(1, n_horizons + 1):
        tw_series = feat["week_start"] + pd.Timedelta(weeks=h)
        valid_mask = (tw_series <= panel_end).values

        # Slice and copy — sub and tw share the same index
        sub = feat_slim.iloc[np.where(valid_mask)[0]].copy()
        tw  = tw_series.iloc[np.where(valid_mask)[0]]   # aligned Series

        # Assign label
        idx      = pd.MultiIndex.from_arrays([sub["ProductID"].values, tw.values])
        sub["y"] = qty_lookup.reindex(idx).values.astype("float32")

        # Guard: drop rows where label is missing (shouldn't happen after mask)
        notna = sub["y"].notna().values
        if not notna.all():
            sub = sub.iloc[np.where(notna)[0]].copy()
            tw  = tw.iloc[np.where(notna)[0]]

        # Recompute calendar features for the TARGET week
        dtw = pd.DatetimeIndex(tw.values)
        woy = dtw.isocalendar().week.values.astype(np.int32)

        sub = sub.copy()   # ensure we own the copy before multi-assign
        sub["week_of_year"]  = woy.astype(np.int16)
        sub["month"]         = dtw.month.values.astype(np.int8)
        sub["quarter"]       = dtw.quarter.values.astype(np.int8)
        sub["week_of_month"] = ((dtw.day.values - 1) // 7 + 1).astype(np.int8)
        sub["sin_annual"]    = np.sin(2 * np.pi * woy / 52).astype(np.float32)
        sub["cos_annual"]    = np.cos(2 * np.pi * woy / 52).astype(np.float32)
        sub["horizon"]       = np.int8(h)
        sub["target_week"]   = dtw.values

        frames.append(sub)

    table = pd.concat(frames, ignore_index=True, sort=False)
    tw_col = pd.to_datetime(table["target_week"])

    is_test  = tw_col >= test_start
    is_valid = (tw_col >= valid_start) & ~is_test

    train_df = table[~is_test & ~is_valid].copy()
    valid_df = table[is_valid].copy()
    test_df  = table[is_test].copy()

    return train_df, valid_df, test_df, feature_cols, cat_cols


def split_boundaries(features_path: Path | str = FEATURES_PATH,
                     n_horizons: int = N_HORIZONS) -> dict:
    """Return the panel_end, test_start, valid_start timestamps."""
    feat = pd.read_parquet(features_path, columns=["week_start"])
    panel_end   = pd.to_datetime(feat["week_start"]).max()
    test_start  = panel_end - pd.Timedelta(weeks=n_horizons - 1)
    valid_start = test_start - pd.Timedelta(weeks=n_horizons)
    return {"panel_end": panel_end, "test_start": test_start, "valid_start": valid_start}
