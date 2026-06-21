"""Baseline demand forecasts for all 13,707 products.

Models: Naive, SeasonalNaive(52), WindowAverage(12),
        CrostonClassic, CrostonOptimized, CrostonSBA, TSB(0.3, 0.3).

Two fixed-origin forecast runs, aligned to the same time split used
in training_table.py / model_global.py:

  VALID origin = valid_start − 1 week  (≈ 2025-03-24)
  TEST  origin = test_start  − 1 week  (≈ 2025-06-16)

Each origin produces h = 1 … 12 week-ahead forecasts.  The two runs are
stacked into a single wide output so Phase 6 can join on
(ProductID, target_week, horizon, split).

Output columns
--------------
ProductID, target_week, horizon, split, y_true,
Naive, SeasonalNaive, WindowAverage,
CrostonClassic, CrostonOptimized, CrostonSBA, TSB
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Must be set before statsforecast is imported so unique_id lands as a column.
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

from statsforecast import StatsForecast                  # noqa: E402
from statsforecast.models import (                       # noqa: E402
    Naive,
    SeasonalNaive,
    WindowAverage,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    TSB,
)

WEEKLY_PRODUCT_PATH = Path("data/processed/weekly_product.parquet")
PREDS_PATH          = Path("data/processed/preds_baselines.parquet")
N_HORIZONS          = 12

_MODELS = [
    Naive(),
    SeasonalNaive(season_length=52),
    WindowAverage(window_size=12),
    CrostonClassic(),
    CrostonOptimized(),
    CrostonSBA(),
    TSB(alpha_d=0.3, alpha_p=0.3),
]

MODEL_COLS = [
    "Naive", "SeasonalNaive", "WindowAverage",
    "CrostonClassic", "CrostonOptimized", "CrostonSBA", "TSB",
]


# ── Private helpers ────────────────────────────────────────────────────────────

def _to_sf_format(wp: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return the zero-filled panel truncated at *cutoff* in statsforecast format.

    Products whose first sale is after *cutoff* have no rows and are therefore
    omitted automatically (they would be cold-start / category_fallback anyway).
    """
    return (
        wp[wp["week_start"] <= cutoff][["ProductID", "week_start", "qty"]]
        .rename(columns={"ProductID": "unique_id", "week_start": "ds", "qty": "y"})
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def _run_sf(hist: pd.DataFrame, h: int) -> pd.DataFrame:
    """Fit all baseline models on *hist* and return an h-step-ahead forecast."""
    sf = StatsForecast(models=_MODELS, freq="W-MON", n_jobs=-1, verbose=False)
    raw = sf.forecast(df=hist, h=h, fitted=False)
    # With NIXTLA_ID_AS_COL=1 the result already has unique_id as a plain column.
    if "unique_id" not in raw.columns:           # guard for unexpected format
        raw = raw.reset_index()
    return raw.rename(columns={"unique_id": "ProductID", "ds": "target_week"})


# ── Public API ─────────────────────────────────────────────────────────────────

def build_baselines(
    weekly_product_path: Path | str = WEEKLY_PRODUCT_PATH,
    out_path:            Path | str = PREDS_PATH,
    n_horizons:          int        = N_HORIZONS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit baselines for VALID and TEST origins; save and return (preds_df, timings).

    Parameters
    ----------
    weekly_product_path : zero-filled weekly panel (from Phase 2)
    out_path            : destination parquet
    n_horizons          : forecast horizon (must match training_table.N_HORIZONS)

    Returns
    -------
    preds_df : wide DataFrame (one column per model)
    timings  : {'valid': seconds, 'test': seconds, 'total': seconds}
    """
    wp = pd.read_parquet(weekly_product_path)
    wp["week_start"] = pd.to_datetime(wp["week_start"])

    panel_end   = wp["week_start"].max()
    test_start  = panel_end   - pd.Timedelta(weeks=n_horizons - 1)
    valid_start = test_start  - pd.Timedelta(weeks=n_horizons)

    test_origin  = test_start  - pd.Timedelta(weeks=1)
    valid_origin = valid_start - pd.Timedelta(weeks=1)

    # Fast label lookup: (ProductID, week_start) → qty
    qty_lookup = wp.set_index(["ProductID", "week_start"])["qty"]

    wall_t0 = time.time()
    frames: list[pd.DataFrame] = []
    timings: dict[str, float] = {}

    for split_name, origin in [("valid", valid_origin), ("test", test_origin)]:
        hist = _to_sf_format(wp, origin)
        n_series = hist["unique_id"].nunique()
        print(f"  [{split_name}] {n_series:,} series | "
              f"origin={origin.date()} | h=1..{n_horizons} …", flush=True)

        t0  = time.time()
        raw = _run_sf(hist, n_horizons)
        elapsed = time.time() - t0
        timings[split_name] = elapsed
        print(f"  [{split_name}] done in {elapsed:.1f}s", flush=True)

        raw["target_week"] = pd.to_datetime(raw["target_week"])
        raw["horizon"]     = ((raw["target_week"] - origin).dt.days // 7).astype("int8")
        raw["split"]       = split_name

        # True demand at each (ProductID, target_week)
        idx            = pd.MultiIndex.from_arrays([raw["ProductID"], raw["target_week"]])
        raw["y_true"]  = qty_lookup.reindex(idx).values.astype("float32")

        # Clip negatives and fill NaN (SeasonalNaive may produce NaN for
        # short series where history < season_length=52 weeks).
        for col in MODEL_COLS:
            if col in raw.columns:
                raw[col] = raw[col].clip(lower=0).fillna(0).astype("float32")

        frames.append(raw)

    preds = pd.concat(frames, ignore_index=True)

    # Ensure all model columns exist (in case a future model alias changes)
    for col in MODEL_COLS:
        if col not in preds.columns:
            preds[col] = np.float32(0)

    preds = preds[
        ["ProductID", "target_week", "horizon", "split", "y_true"] + MODEL_COLS
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_path, index=False)

    timings["total"] = time.time() - wall_t0
    return preds, timings
