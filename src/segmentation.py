"""Product segmentation: Syntetos-Boylan demand classes, ABC, and routing.

Computes per-product demand characteristics from the weekly panel and assigns:
  - demand_class  : Smooth / Intermittent / Erratic / Lumpy / Too sparse
                    (Syntetos-Boylan, thresholds ADI=1.32 and CV²=0.49)
  - abc_class     : A (≤80% cumulative qty), B (≤95%), C (rest)
  - is_cold_start : first_sale_week is within the last 12 weeks of the panel
  - route         : global_model | baseline_intermittent | category_fallback

ADI definition (per spec): mean gap *in weeks* between consecutive non-zero
demand weeks.  For a product with demand at weeks [w1, w2, w3] the gaps are
[w2-w1, w3-w2] and ADI = mean of those gaps.
Products with fewer than 2 active weeks receive demand_class = "Too sparse"
(ADI and CV² are undefined for them; both stored as NaN).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WEEKLY_PRODUCT_PATH = Path("data/processed/weekly_product.parquet")
SEGMENTS_PATH = Path("data/processed/product_segments.parquet")

_ADI_THRESH: float = 1.32
_CV2_THRESH: float = 0.49


# ── Feature computation ────────────────────────────────────────────────────────

def _adi_cv2(wp: pd.DataFrame) -> pd.DataFrame:
    """Compute ADI and CV² for every product from non-zero demand weeks.

    ADI  = mean inter-demand interval (weeks) between consecutive non-zero weeks.
           Requires n_active >= 2; NaN otherwise (Too sparse).
    CV²  = (std / mean)² of non-zero weekly quantities (ddof=1).
           0.0 when n_active == 1; NaN when n_active == 0.
    """
    nz = wp[wp["qty"] > 0][["ProductID", "week_start", "qty"]].copy()

    # ADI: mean gap between consecutive non-zero weeks (in weeks)
    nz_sorted = nz.sort_values(["ProductID", "week_start"])
    nz_sorted["gap_w"] = (
        nz_sorted.groupby("ProductID")["week_start"]
        .diff()
        .dt.total_seconds()
        .div(7 * 86_400)        # convert seconds → weeks
    )
    adi = nz_sorted.groupby("ProductID")["gap_w"].mean()   # NaN if n_active == 1
    adi.name = "adi"

    # CV²: vectorised — var/mean² with ddof=1
    grp = nz.groupby("ProductID")["qty"]
    cv2 = (grp.var(ddof=1) / grp.mean() ** 2).fillna(0.0)  # NaN (n=1) → 0
    cv2.name = "cv2"

    return pd.concat([adi, cv2], axis=1)


def _weeks_since_last_sale(wp: pd.DataFrame, global_last: pd.Timestamp) -> pd.Series:
    """Return integer weeks between each product's last non-zero week and global_last."""
    last_nz = wp[wp["qty"] > 0].groupby("ProductID")["week_start"].max()
    wsls = ((global_last - last_nz).dt.total_seconds() / (7 * 86_400)).round().astype("Int64")
    wsls.name = "weeks_since_last_sale"
    return wsls


# ── Classification helpers ─────────────────────────────────────────────────────

def _demand_class(seg: pd.DataFrame) -> pd.Series:
    """Assign Syntetos-Boylan demand class."""
    dc = pd.Series("", index=seg.index, dtype="object")

    too_sparse = seg["n_active_weeks"] < 2
    dc[too_sparse] = "Too sparse"

    ok = ~too_sparse
    dc[ok & (seg["adi"] <  _ADI_THRESH) & (seg["cv2"] <  _CV2_THRESH)] = "Smooth"
    dc[ok & (seg["adi"] >= _ADI_THRESH) & (seg["cv2"] <  _CV2_THRESH)] = "Intermittent"
    dc[ok & (seg["adi"] <  _ADI_THRESH) & (seg["cv2"] >= _CV2_THRESH)] = "Erratic"
    dc[ok & (seg["adi"] >= _ADI_THRESH) & (seg["cv2"] >= _CV2_THRESH)] = "Lumpy"

    return dc


def _abc_class(total_qty: pd.Series) -> pd.Series:
    """Assign ABC class by cumulative volume share, sorted descending."""
    idx_desc = total_qty.sort_values(ascending=False).index
    cum = total_qty.loc[idx_desc].cumsum() / total_qty.sum()

    abc = pd.Series("C", index=total_qty.index, dtype="object")
    abc.loc[cum.index[cum <= 0.80]] = "A"
    abc.loc[cum.index[(cum > 0.80) & (cum <= 0.95)]] = "B"
    return abc


def _route(seg: pd.DataFrame) -> pd.Series:
    """Assign model routing label.

    Routing is driven solely by ABC class and cold-start status.
    demand_class is kept as a model feature and for per-segment reporting
    in Phase 6, but does not gate routing here.

    Rules (in priority order):
      1. category_fallback  — is_cold_start (no history to train on)
      2. global_model       — abc_class in {A, B}  (evaluable volume)
      3. baseline_intermittent — abc_class == C  (long tail, low volume)

    Note: intermittent-demand baselines (Croston, SBA, TSB, …) are computed
    for ALL products regardless of route; final per-segment model selection
    happens in Phase 6 after cross-validation metrics are compared.
    """
    route = pd.Series("baseline_intermittent", index=seg.index, dtype="object")

    cold = seg["is_cold_start"]
    route[cold] = "category_fallback"
    route[~cold & seg["abc_class"].isin({"A", "B"})] = "global_model"
    return route


# ── Public API ─────────────────────────────────────────────────────────────────

def build_segments(
    parquet_in: Path | str = WEEKLY_PRODUCT_PATH,
    parquet_out: Path | str = SEGMENTS_PATH,
) -> pd.DataFrame:
    """Compute product segments, save to parquet, and return the DataFrame.

    Columns in output
    -----------------
    ProductID, total_qty, n_active_weeks, panel_weeks,
    adi, cv2, first_sale_week, weeks_since_last_sale,
    demand_class, abc_class, is_cold_start, route
    """
    parquet_in = Path(parquet_in)
    parquet_out = Path(parquet_out)

    wp = pd.read_parquet(parquet_in)
    wp["week_start"]    = pd.to_datetime(wp["week_start"])
    wp["first_sale_week"] = pd.to_datetime(wp["first_sale_week"])

    global_last = wp["week_start"].max()
    cold_cutoff = global_last - pd.Timedelta(weeks=12)

    # ── Per-product metadata (one row per product, already stored in panel) ───
    seg = (
        wp.groupby("ProductID", sort=False)
        .agg(
            panel_weeks=("week_start", "count"),
            n_active_weeks=("n_active_weeks", "first"),
            total_qty=("total_qty", "first"),
            first_sale_week=("first_sale_week", "first"),
        )
        .reset_index()
    )

    # ── ADI and CV² ───────────────────────────────────────────────────────────
    stats = _adi_cv2(wp)
    seg = seg.join(stats, on="ProductID")

    # ── weeks_since_last_sale ─────────────────────────────────────────────────
    wsls = _weeks_since_last_sale(wp, global_last)
    seg = seg.join(wsls, on="ProductID")

    # ── Classification ────────────────────────────────────────────────────────
    seg["demand_class"]  = _demand_class(seg)
    seg["abc_class"]     = _abc_class(seg["total_qty"])
    seg["is_cold_start"] = seg["first_sale_week"] > cold_cutoff
    seg["route"]         = _route(seg)

    # ── Column order ──────────────────────────────────────────────────────────
    seg = seg[[
        "ProductID", "total_qty", "n_active_weeks", "panel_weeks",
        "adi", "cv2", "first_sale_week", "weeks_since_last_sale",
        "demand_class", "abc_class", "is_cold_start", "route",
    ]]

    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(parquet_out, index=False)
    return seg
