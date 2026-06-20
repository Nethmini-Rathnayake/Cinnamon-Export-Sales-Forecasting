"""Feature engineering for weekly demand forecasting.

Produces a model-ready table: one row per (ProductID, week_start).
All lag and rolling features are computed from strictly past data —
every series is shifted by at least 1 week within each product group
so that week t never sees its own qty value.

Feature families
----------------
Lags           : qty_lag_{1,2,3,4,8,12,52}w
Rolling        : roll_{4,8,12}w_{mean,sum,std,nonzero_rate}  (window over shifted qty)
Intermittency  : weeks_since_last_sale_t, zero_streak,
                 weeks_since_first_sale, cumulative_active_weeks,
                 expanding_mean_demand
Calendar       : week_of_year, month, quarter, week_of_month,
                 sin_annual, cos_annual  (Fourier, period=52 weeks)
Promo          : promo_sum_{4,12}w, promo_share_{4,12}w
Price          : unit_price_{4,12}w, unit_price_12w_change,
                 usd_per_kg_{4,12}w, usd_per_kg_12w_change
                 (all from non-promo transactions only)
Static         : grade, pack_size, pack_type, region_seg,
                 Brand Category, Product Range, Sales Channel,
                 demand_class, abc_class   (stored as category dtype)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WEEKLY_PRODUCT_PATH = Path("data/processed/weekly_product.parquet")
SEGMENTS_PATH       = Path("data/processed/product_segments.parquet")
TRANSACTIONS_PATH   = Path("data/processed/transactions_clean.parquet")
FEATURES_PATH       = Path("data/processed/features_product.parquet")

_LAG_WEEKS    = [1, 2, 3, 4, 8, 12, 52]
_ROLL_WINDOWS = [4, 8, 12]
_STATIC_CAT_COLS = [
    "grade", "pack_size", "pack_type", "region_seg",
    "Brand Category", "Product Range", "Sales Channel",
]


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_price_weekly(transactions_path: Path | str) -> pd.DataFrame:
    """Aggregate non-promo transactions to weekly unit price and USD/KG per product."""
    tx = pd.read_parquet(
        transactions_path,
        columns=["ProductID", "Order Date", "Sales USD", "Sales Qty", "Sales KG", "is_promo"],
    )
    tx = tx[tx["is_promo"] == 0].copy()
    tx["week_start"] = tx["Order Date"].dt.to_period("W-SUN").dt.start_time

    agg = (
        tx.groupby(["ProductID", "week_start"])
        .agg(
            sales_usd=("Sales USD", "sum"),
            sales_qty=("Sales Qty", "sum"),
            sales_kg=("Sales KG",  "sum"),
        )
        .reset_index()
    )
    agg["unit_price"] = np.where(agg["sales_qty"] > 0,
                                 agg["sales_usd"] / agg["sales_qty"], np.nan)
    agg["usd_per_kg"] = np.where(agg["sales_kg"]  > 0,
                                 agg["sales_usd"] / agg["sales_kg"],  np.nan)
    return agg[["ProductID", "week_start", "unit_price", "usd_per_kg"]]


def _build_statics(transactions_path: Path | str) -> pd.DataFrame:
    """Return first non-null static attribute per product from the transaction table."""
    tx = pd.read_parquet(transactions_path, columns=["ProductID"] + _STATIC_CAT_COLS)
    return tx.groupby("ProductID", sort=False)[_STATIC_CAT_COLS].first().reset_index()


# ── Public API ─────────────────────────────────────────────────────────────────

def build_features(
    weekly_product_path: Path | str = WEEKLY_PRODUCT_PATH,
    segments_path:       Path | str = SEGMENTS_PATH,
    transactions_path:   Path | str = TRANSACTIONS_PATH,
    out_path:            Path | str = FEATURES_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Build, save, and return the model-ready feature table.

    Returns
    -------
    (df, summary)
        df      : feature DataFrame (ProductID, week_start, qty, features …)
        summary : dict with n_rows, n_feature_cols, dropped_cols
    """
    # ── Load and sort base panel ───────────────────────────────────────────────
    wp = pd.read_parquet(weekly_product_path)
    wp["week_start"] = pd.to_datetime(wp["week_start"])

    # Sort ascending by (product, time) — required for all groupby time-series ops.
    df = (wp
          .sort_values(["ProductID", "week_start"])
          .reset_index(drop=True)
          [["ProductID", "week_start", "qty", "promo_units", "weeks_since_first_sale"]]
          .copy())

    # ── 1. Lag features ───────────────────────────────────────────────────────
    g_qty = df.groupby("ProductID", sort=False)["qty"]
    for n in _LAG_WEEKS:
        df[f"qty_lag_{n}w"] = g_qty.shift(n).astype("float32")

    # ── 2. Rolling features (shift by 1 first, then roll within groups) ───────
    # All rolling windows look at weeks [t-w, …, t-1] so week t is never included.
    df["_qty_s1"] = g_qty.shift(1)
    g_s1 = df.groupby("ProductID", sort=False)["_qty_s1"]

    for w in _ROLL_WINDOWS:
        df[f"roll_{w}w_mean"] = g_s1.rolling(w, min_periods=1).mean().values.astype("float32")
        df[f"roll_{w}w_sum"]  = g_s1.rolling(w, min_periods=1).sum().values.astype("float32")
        df[f"roll_{w}w_std"]  = g_s1.rolling(w, min_periods=2).std().values.astype("float32")

    df["_nz_s1"] = (df["_qty_s1"] > 0).astype("float32")
    g_nz = df.groupby("ProductID", sort=False)["_nz_s1"]
    for w in _ROLL_WINDOWS:
        df[f"roll_{w}w_nonzero_rate"] = (
            g_nz.rolling(w, min_periods=1).mean().values.astype("float32")
        )

    # ── 3. Intermittency / recency features ───────────────────────────────────
    # weeks_since_last_sale_t: at week t, how many weeks since the last non-zero demand?
    # Step 1 — mark each non-zero week with its own timestamp, NaN elsewhere.
    df["_ws_nz"] = df["week_start"].where(df["qty"] > 0)
    # Step 2 — forward-fill within each product group: fills NaN zeros with the
    #           most recent non-zero week_start seen so far.
    df["_last_nz_w"] = df.groupby("ProductID", sort=False)["_ws_nz"].transform("ffill")
    # Step 3 — shift by 1: at week t, we can only use info from week t-1 and earlier.
    df["_last_nz_w_s1"] = df.groupby("ProductID", sort=False)["_last_nz_w"].shift(1)
    df["weeks_since_last_sale_t"] = (
        (df["week_start"] - df["_last_nz_w_s1"]).dt.total_seconds() / (7 * 86_400)
    ).round().astype("float32")

    # zero_streak: consecutive zeros immediately before week t  (= wsls - 1, floor 0)
    df["zero_streak"] = (df["weeks_since_last_sale_t"] - 1).clip(lower=0).astype("float32")

    # cumulative_active_weeks: count of weeks with qty>0 in [first_week, t-1]
    df["_nz_flag_s1"] = (
        df.groupby("ProductID", sort=False)["qty"]
        .transform(lambda x: (x > 0).astype(int).shift(1).fillna(0))
    )
    df["cumulative_active_weeks"] = (
        df.groupby("ProductID", sort=False)["_nz_flag_s1"]
        .cumsum().astype("float32")
    )

    # expanding_mean_demand: expanding mean of qty over [first_week, t-1]
    df["expanding_mean_demand"] = (
        df.groupby("ProductID", sort=False)["_qty_s1"]
        .expanding(min_periods=1).mean().values.astype("float32")
    )

    # weeks_since_first_sale already present in panel (0-indexed counter per product).
    # It encodes the "age" of the series at each week and does not leak the target.

    # ── 4. Calendar features ──────────────────────────────────────────────────
    woy = df["week_start"].dt.isocalendar().week.astype(int)
    df["week_of_year"]  = woy.astype("int16").values
    df["month"]         = df["week_start"].dt.month.astype("int8")
    df["quarter"]       = df["week_start"].dt.quarter.astype("int8")
    df["week_of_month"] = ((df["week_start"].dt.day - 1) // 7 + 1).astype("int8")
    df["sin_annual"]    = np.sin(2 * np.pi * woy / 52).astype("float32")
    df["cos_annual"]    = np.cos(2 * np.pi * woy / 52).astype("float32")

    # ── 5. Promo features (trailing windows of shifted promo_units) ───────────
    df["_promo_s1"] = df.groupby("ProductID", sort=False)["promo_units"].shift(1)
    g_promo = df.groupby("ProductID", sort=False)["_promo_s1"]
    for w in [4, 12]:
        df[f"promo_sum_{w}w"] = (
            g_promo.rolling(w, min_periods=1).sum().values.astype("float32")
        )
    # promo share = promo volume / (total demand volume + 1)
    for w in [4, 12]:
        df[f"promo_share_{w}w"] = (
            df[f"promo_sum_{w}w"] / (df[f"roll_{w}w_sum"] + 1)
        ).astype("float32")

    # ── 6. Price features (non-promo only; shifted + rolling to avoid leakage) ─
    price_weekly = _build_price_weekly(transactions_path)
    df = df.merge(price_weekly, on=["ProductID", "week_start"], how="left")

    for pcol in ["unit_price", "usd_per_kg"]:
        df[f"_{pcol}_s1"] = df.groupby("ProductID", sort=False)[pcol].shift(1)
        g_p = df.groupby("ProductID", sort=False)[f"_{pcol}_s1"]
        df[f"{pcol}_4w"]  = g_p.rolling(4,  min_periods=1).mean().values.astype("float32")
        df[f"{pcol}_12w"] = g_p.rolling(12, min_periods=1).mean().values.astype("float32")
        # Price change: trailing 4w avg now vs 12 weeks ago
        df[f"{pcol}_12w_change"] = (
            df[f"{pcol}_4w"]
            - df.groupby("ProductID", sort=False)[f"{pcol}_4w"].shift(12)
        ).astype("float32")
        df.drop(columns=[f"_{pcol}_s1", pcol], inplace=True)

    # ── 7. Static attributes ──────────────────────────────────────────────────
    statics = _build_statics(transactions_path)
    seg = pd.read_parquet(segments_path)[["ProductID", "demand_class", "abc_class"]]
    df = df.merge(statics, on="ProductID", how="left")
    df = df.merge(seg,     on="ProductID", how="left")

    for col in _STATIC_CAT_COLS + ["demand_class", "abc_class"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # ── 8. Drop temporary working columns ─────────────────────────────────────
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=temp_cols + ["promo_units"], inplace=True)

    # ── 9. Drop constant or fully-null numeric features ───────────────────────
    key_target = {"ProductID", "week_start", "qty"}
    dropped_cols: list[str] = []
    for col in list(df.select_dtypes(include="number").columns):
        if col in key_target:
            continue
        if df[col].nunique(dropna=False) <= 1 or df[col].isna().all():
            df.drop(columns=[col], inplace=True)
            dropped_cols.append(col)

    # ── 10. Persist ───────────────────────────────────────────────────────────
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    feat_cols = [c for c in df.columns if c not in key_target]
    return df, {
        "n_rows":         len(df),
        "n_feature_cols": len(feat_cols),
        "dropped_cols":   dropped_cols,
    }
