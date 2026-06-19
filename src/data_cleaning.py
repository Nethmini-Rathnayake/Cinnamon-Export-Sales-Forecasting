"""Transaction-level loading and cleaning for cinnamon export sales.

Binding Data Rules applied here (from CLAUDE.md):
  Rule 1 — Drop the corrupt Sales Qty > 1 M row and drop all rows for product 43962-015.
  Rule 2 — ProductID = Product Code[:9]; parse suffix into grade/pack_size/pack_type/region_seg.
  Rule 3 — Parse Order Date and Invoice Date; backfill missing Order Date from Invoice Date;
            drop rows where both are null.
  Rule 4 — Flag is_promo = (Sales USD <= 0); keep promo rows in the dataset.

Rules 5 (net returns within ISO week) and 6 (ISO-week series construction) are deferred
to the weekly aggregation step (Phase 2).

Product code format example:
  44737-053-R09-SQW-WWJ-0334ST-4.4G-PRM
  ^^^^^^^^^                              = ProductID (first 9 chars)
            ^^^                          = region_seg  (R09)
                          ^^^^^^         = pack_type embedded after 4-digit number (0334ST → ST)
                                 ^^^^    = pack_size (4.4G)
                                      ^^^ = grade (PRM)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/Cinnamon_export_sales.xlsx")
PROCESSED_PATH = Path("data/processed/transactions_clean.parquet")

_DROP_PID = "43962-015"

_GRADE_RE = re.compile(r"\b(PRM|STD|ECO)\b")
_PACK_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?G)\b")  # e.g. 4.4G, 5G, 0.9G
_PACK_TYPE_RE = re.compile(r"\d{4}(ST|PT)")         # e.g. 0334ST → "ST"  (no dash between digits and type)
_REGION_SEG_RE = re.compile(r"\b(R\d{2})\b")        # e.g. R09, R01, R00


def _parse_suffix(code: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract (grade, pack_size, pack_type, region_seg) from the suffix of a product code.

    The suffix is everything after the first 9 characters (the ProductID portion).
    """
    if not isinstance(code, str) or len(code) <= 9:
        return None, None, None, None
    s = code[9:]

    def _first(pat: re.Pattern) -> str | None:
        m = pat.search(s)
        return m.group(1) if m else None

    return _first(_GRADE_RE), _first(_PACK_SIZE_RE), _first(_PACK_TYPE_RE), _first(_REGION_SEG_RE)


def load_clean(
    raw_path: Path | str = RAW_PATH,
    out_path: Path | str = PROCESSED_PATH,
) -> dict:
    """Load, clean, and persist cinnamon export transactions as Parquet.

    Returns a dict of summary counts for the Phase 1 data-quality report.
    """
    raw_path, out_path = Path(raw_path), Path(out_path)

    # ── Load ─────────────────────────────────────────────────────────────────
    raw = pd.read_excel(raw_path, sheet_name="Sheet1")
    rows_raw = len(raw)
    products_raw = int(raw["Product Code"].str[:9].nunique())

    # ── Rule 1a: drop corrupt row (Sales Qty > 1 M) ──────────────────────────
    mask_corrupt = raw["Sales Qty"] > 1_000_000
    rows_dropped_corrupt = int(mask_corrupt.sum())
    df = raw[~mask_corrupt].copy()

    # ── Rule 2a: ProductID = first 9 characters ───────────────────────────────
    df["ProductID"] = df["Product Code"].str[:9]

    # ── Rule 1b: drop all rows for product 43962-015 ─────────────────────────
    mask_pid = df["ProductID"] == _DROP_PID
    rows_dropped_product = int(mask_pid.sum())
    df = df[~mask_pid].copy()

    # ── Rule 3: parse dates; backfill Order Date; drop rows with neither ──────
    for col in ("Order Date", "Invoice Date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    n_null_order_before = int(df["Order Date"].isna().sum())
    df["Order Date"] = df["Order Date"].fillna(df["Invoice Date"])
    nulls_backfilled = n_null_order_before - int(df["Order Date"].isna().sum())

    rows_dropped_no_date = int(df["Order Date"].isna().sum())
    df = df[df["Order Date"].notna()].copy()

    # ── Rule 2b: parse product-code suffix into feature columns ──────────────
    parsed = df["Product Code"].map(_parse_suffix)
    df[["grade", "pack_size", "pack_type", "region_seg"]] = pd.DataFrame(
        parsed.tolist(), index=df.index
    )

    # ── Rule 4: promotional-row flag ──────────────────────────────────────────
    # is_promo rows are kept in the Qty target; exclude them from price features later.
    df["is_promo"] = (df["Sales USD"] <= 0).astype("int8")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    returns_count = int((df["Sales Qty"] < 0).sum())
    promo_rows = int(df["is_promo"].sum())
    products_clean = int(df["ProductID"].nunique())

    # ── Persist ───────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "rows_raw": rows_raw,
        "rows_dropped_corrupt": rows_dropped_corrupt,
        "rows_dropped_product_43962_015": rows_dropped_product,
        "nulls_order_date_backfilled": nulls_backfilled,
        "rows_dropped_no_date": rows_dropped_no_date,
        "rows_clean": len(df),
        "returns_count": returns_count,
        "promo_rows": promo_rows,
        "products_raw": products_raw,
        "products_clean": products_clean,
    }
