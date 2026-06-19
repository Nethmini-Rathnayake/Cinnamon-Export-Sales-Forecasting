"""Build ISO-week demand panels from cleaned transactions.

Binding Data Rules applied here (from CLAUDE.md):
  Rule 5 — Net returns within each ISO-week bucket; clip negative weekly totals to 0.
  Rule 6 — ISO weeks, Monday start (Period freq='W-SUN' in pandas = Mon–Sun).
            Each series runs from the key's own first sale to the global last
            complete week; zero-fill inactive weeks; add weeks_since_first_sale.

Outputs
-------
data/processed/weekly_product.parquet
    Long-format product × week panel with qty, promo_units, and per-product
    metadata (first_sale_week, n_active_weeks, total_qty, weeks_since_first_sale).

data/processed/weekly_product_country.parquet
    Same construction keyed on (ProductID, Country) for the country drill-down.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRANSACTIONS_PATH = Path("data/processed/transactions_clean.parquet")
WEEKLY_PRODUCT_PATH = Path("data/processed/weekly_product.parquet")
WEEKLY_PC_PATH = Path("data/processed/weekly_product_country.parquet")

_WEEK_NS = int(7 * 24 * 3600 * 1e9)  # nanoseconds in exactly 7 days


# ── Private helpers ────────────────────────────────────────────────────────────

def _week_start(dates: pd.Series) -> pd.Series:
    """Return the Monday midnight Timestamp for each date's ISO week (W-SUN period)."""
    return dates.dt.to_period("W-SUN").dt.start_time


def _last_complete_week(df: pd.DataFrame) -> tuple[pd.Timestamp, int]:
    """Return (last complete week_start, row-count trimmed from the partial week).

    The final partial week is the most recent ISO week that does not have data
    through its Sunday. We drop it so the panel ends on a complete week boundary.
    """
    max_week = df["week_start"].max()          # Monday of the latest observed week
    week_end_sunday = max_week + pd.Timedelta(days=6)
    max_order_date = df["Order Date"].max().normalize()

    if max_order_date < week_end_sunday:
        # Latest week is partial (data ends before Sunday) → drop it
        last_complete = max_week - pd.Timedelta(weeks=1)
        trimmed = int((df["week_start"] == max_week).sum())
    else:
        last_complete = max_week
        trimmed = 0
    return last_complete, trimmed


def _build_full_panel(
    sparse: pd.DataFrame,
    key_cols: list[str],
    value_cols: list[str],
    global_last: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand sparse weekly data to a dense zero-filled panel.

    For each unique key group the series spans from the group's own first week
    to *global_last*; missing weeks are filled with zeros.

    Parameters
    ----------
    sparse      : DataFrame with columns key_cols + ['week_start'] + value_cols
    key_cols    : group identifier columns (e.g. ['ProductID'] or ['ProductID','Country'])
    value_cols  : numeric columns to zero-fill
    global_last : Monday Timestamp of the last complete ISO week

    Returns
    -------
    (full_panel, first_week_df)
        full_panel     : zero-filled DataFrame, sorted by key_cols + ['week_start']
        first_week_df  : per-key first-sale week_start
    """
    # Per-key first sale week (as int64 nanoseconds for vectorised arithmetic)
    first = (
        sparse.groupby(key_cols)["week_start"]
        .min()
        .reset_index(name="first_week")
    )
    global_last_ns = global_last.value
    first_ns = first["first_week"].astype(np.int64).values

    # Weeks per key: inclusive count from first_week to global_last
    n_weeks_arr = (global_last_ns - first_ns) // _WEEK_NS + 1

    # Build flat (key, week_start) arrays using nanosecond arithmetic
    week_ns_flat = np.concatenate([
        np.arange(fn, fn + nw * _WEEK_NS, _WEEK_NS, dtype=np.int64)
        for fn, nw in zip(first_ns, n_weeks_arr)
    ])
    week_start_flat = pd.to_datetime(week_ns_flat)

    key_data = {col: np.repeat(first[col].values, n_weeks_arr) for col in key_cols}
    full = pd.DataFrame({**key_data, "week_start": week_start_flat})

    # Left-join to pull in aggregated values; fill gaps with 0
    full = full.merge(
        sparse[key_cols + ["week_start"] + value_cols],
        on=key_cols + ["week_start"],
        how="left",
    )
    full[value_cols] = full[value_cols].fillna(0).astype("float32")

    return full, first


# ── Public API ─────────────────────────────────────────────────────────────────

def build_weekly_product(
    parquet_in: Path | str = TRANSACTIONS_PATH,
    parquet_out: Path | str = WEEKLY_PRODUCT_PATH,
) -> dict:
    """Build and save the product-level weekly demand panel.

    Returns a summary-stats dict for the Phase 2 quality report.
    """
    parquet_in, parquet_out = Path(parquet_in), Path(parquet_out)
    df = pd.read_parquet(parquet_in)

    # ── Assign ISO week (Monday start) ────────────────────────────────────────
    df["week_start"] = _week_start(df["Order Date"])

    # ── Identify and drop the final partial week (Rule 6) ─────────────────────
    global_last, rows_trimmed = _last_complete_week(df)
    df = df[df["week_start"] <= global_last].copy()
    global_first = df["week_start"].min()

    # ── Weekly aggregation (Rule 5) ───────────────────────────────────────────
    # promo_units = sum of Sales Qty on is_promo rows only
    df["qty_promo"] = df["Sales Qty"].where(df["is_promo"] == 1, 0.0)

    weekly_raw = (
        df.groupby(["ProductID", "week_start"], observed=True, sort=True)
        .agg(
            net_qty_raw=("Sales Qty", "sum"),
            promo_units=("qty_promo", "sum"),
        )
        .reset_index()
    )
    # Clip negative weekly net totals to 0 (Rule 5)
    weekly_raw["qty"] = weekly_raw["net_qty_raw"].clip(lower=0).astype("float32")
    weekly_raw["promo_units"] = weekly_raw["promo_units"].astype("float32")
    weekly_raw = weekly_raw.drop(columns="net_qty_raw")

    # ── Build zero-filled full panel (Rule 6) ─────────────────────────────────
    full, first_df = _build_full_panel(
        weekly_raw,
        key_cols=["ProductID"],
        value_cols=["qty", "promo_units"],
        global_last=global_last,
    )

    # ── Derived metadata columns ──────────────────────────────────────────────
    full = full.sort_values(["ProductID", "week_start"]).reset_index(drop=True)

    # weeks_since_first_sale: 0-indexed counter per product
    full["weeks_since_first_sale"] = (
        full.groupby("ProductID").cumcount().astype("int32")
    )

    # first_sale_week: constant per product
    pid_to_first = first_df.set_index("ProductID")["first_week"]
    full["first_sale_week"] = full["ProductID"].map(pid_to_first)

    # n_active_weeks and total_qty: computed over the full (zero-filled) span
    prod_meta = (
        full.groupby("ProductID", sort=False)
        .agg(
            n_active_weeks=("qty", lambda x: int((x > 0).sum())),
            total_qty=("qty", "sum"),
        )
    )
    full = full.join(prod_meta, on="ProductID")
    full["n_active_weeks"] = full["n_active_weeks"].astype("int32")

    # ── Persist ───────────────────────────────────────────────────────────────
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(parquet_out, index=False)

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_span = full.groupby("ProductID")["weeks_since_first_sale"].max() + 1
    active_ratio = prod_meta["n_active_weeks"] / total_span

    return {
        "global_first_week": global_first,
        "global_last_week": global_last,
        "rows_trimmed_partial": rows_trimmed,
        "n_product_weeks": len(full),
        "n_products": full["ProductID"].nunique(),
        "zero_week_share": float((full["qty"] == 0).mean()),
        "median_active_week_ratio": float(active_ratio.median()),
    }


def build_weekly_product_country(
    parquet_in: Path | str = TRANSACTIONS_PATH,
    parquet_out: Path | str = WEEKLY_PC_PATH,
    global_last: pd.Timestamp | None = None,
) -> dict:
    """Build and save the product × country weekly demand panel.

    Uses the same global_last week as the product panel (pass it in, or it is
    re-derived from the data). Returns a summary-stats dict.
    """
    parquet_in, parquet_out = Path(parquet_in), Path(parquet_out)
    df = pd.read_parquet(parquet_in)

    df["week_start"] = _week_start(df["Order Date"])

    if global_last is None:
        global_last, rows_trimmed = _last_complete_week(df)
    else:
        rows_trimmed = int((df["week_start"] > global_last).sum())

    df = df[df["week_start"] <= global_last].copy()

    df["qty_promo"] = df["Sales Qty"].where(df["is_promo"] == 1, 0.0)

    weekly_raw = (
        df.groupby(["ProductID", "Country", "week_start"], observed=True, sort=True)
        .agg(
            net_qty_raw=("Sales Qty", "sum"),
            promo_units=("qty_promo", "sum"),
        )
        .reset_index()
    )
    weekly_raw["qty"] = weekly_raw["net_qty_raw"].clip(lower=0).astype("float32")
    weekly_raw["promo_units"] = weekly_raw["promo_units"].astype("float32")
    weekly_raw = weekly_raw.drop(columns="net_qty_raw")

    full, first_df = _build_full_panel(
        weekly_raw,
        key_cols=["ProductID", "Country"],
        value_cols=["qty", "promo_units"],
        global_last=global_last,
    )

    full = full.sort_values(["ProductID", "Country", "week_start"]).reset_index(drop=True)

    full["weeks_since_first_sale"] = (
        full.groupby(["ProductID", "Country"]).cumcount().astype("int32")
    )

    pc_to_first = first_df.set_index(["ProductID", "Country"])["first_week"]
    full["first_sale_week"] = full.set_index(["ProductID", "Country"]).index.map(pc_to_first).values

    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(parquet_out, index=False)

    return {
        "global_last_week": global_last,
        "rows_trimmed_partial": rows_trimmed,
        "n_pc_weeks": len(full),
        "n_product_country_pairs": full.groupby(["ProductID", "Country"]).ngroups,
        "zero_week_share": float((full["qty"] == 0).mean()),
    }
