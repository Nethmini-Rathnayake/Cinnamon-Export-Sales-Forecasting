# Cinnamon Export Sales — Weekly Demand Forecasting

## Project goal
- Forecast **weekly Sales Qty** (the target variable) **12 weeks ahead**, per product, and per product × country.
- A **product = first 9 characters of `Product Code`**.
- Deliverables: well-documented source code, a 12-week forecast file (CSV/Excel), a project report, and a presentation.

## Roles
- Strategy, data decisions, and presentation figures are planned in Claude.ai chat.
- You (Claude Code) implement: clean, modular code in `src/`; EDA and experiments in `notebooks/`.
- Save every presentation figure as a numbered PNG to `outputs/figures/` (e.g. `01_qty_distribution.png`) at 140 dpi, consistent style.
- Work one phase at a time. After finishing a phase, append a line to the Decision Log at the bottom.

## Data
- Single file: `data/raw/Cinnamon_export_sales.xlsx`, sheet `Sheet1`, 60,670 rows.
- Columns: Region, Country, Customer Code, Customer ID, Brand Category, Product Range, Sales Channel, Product Code, Order Date, Invoice Date, Invoice No, Sales USD, Sales Qty, Sales KG.
- Order Date span: 2022-02-28 → 2025-09-17 (~185 weeks).
- 19 regions, 91 countries, ~13,724 products (first-9), 14 brand categories, 55 product ranges, 4 channels (RETAIL / FOOD SERVICE / GLOBAL / BULK).

## THE key fact (defines the whole approach)
Demand is **extremely sparse / intermittent**: 89% of products have ≤10 lifetime transactions, 6,006 sold exactly once, and there are **zero "smooth" series**. This is **intermittent-demand-at-scale**, not classic time series. **Do NOT fit one ARIMA/Prophet/LSTM per product.** Pool series into global models and benchmark against intermittent-demand baselines.

## Binding data rules (apply in every phase)
1. **Drop** the corrupt row where `Sales Qty > 1e6` (all categorical fields null, ~14.1M units) and drop the whole product `43962-015`.
2. `ProductID = Product Code[:9]`. Also parse the full code suffix into feature columns: `grade` (last token PRM/STD/ECO), `pack_size` (e.g. 4.4G), `pack_type` (ST/PT), `region_seg` (R0x).
3. Bucket on **Order Date** (demand signal). Backfill missing Order Date from Invoice Date; drop the rows that have neither.
4. **Keep** promotional / `Sales USD <= 0` rows in the Qty target (real units shipped). Add an `is_promo` flag. **Exclude** these rows from unit-price features (USD/Qty, USD/KG).
5. **Net returns** (negative Sales Qty) within the weekly bucket; clip a negative weekly total to 0.
6. **ISO weeks, Monday start.** Build each product's weekly series from **its own first sale** to the global end date; zero-fill inactive weeks. Add `weeks_since_first_sale`.

## Modeling strategy
- **Segment** products: ABC by volume × Syntetos–Boylan ADI/CV² class; route segments to tracks.
- **PRIMARY:** one **global LightGBM** over all series (lag / rolling / calendar / intermittency features + categorical encodings). Use a **direct** multi-step setup (horizon as a feature or per-horizon models), not recursive.
- **BASELINES:** Naive, Seasonal-Naive, Moving Average, Croston, SBA, TSB. Also the fallback for the ultra-sparse tail.
- **OPTIONAL:** one global neural model (N-BEATS / DeepAR / LSTM via `neuralforecast` or `darts`) **only** to produce the train/val loss curves the brief asks for.
- **Country drill-down:** top-N countries per product → hierarchical forecast + reconciliation (bottom-up / MinT) so product × country sums to the product total.

## Metrics & validation
- Use **RMSE, MAE, RMSSE, MASE, WMAPE**. **Never plain MAPE** (zeros everywhere).
- Validation: **rolling-origin / expanding-window** CV; hold out the **last 12 weeks** as the final test.
- Report metrics **per segment** (ABC, intermittency class), not just one global number.
- Output a 12-week forecast for **all** products, but report accuracy on the **evaluable subset** (active in last 12 weeks / ABC A+B). Cold-start products (first sale <12 weeks ago, ~747) → category-level fallback.

## Repo conventions
- `src/` reusable modules · `notebooks/` EDA + experiments · `data/raw` (gitignored) · `data/processed` · `outputs/figures` (numbered PNGs) · `outputs/forecasts` · `reports/`.
- `requirements.txt` pinned. Python 3.11+. Core libs: pandas, numpy, openpyxl, lightgbm, statsforecast, hierarchicalforecast, scikit-learn, matplotlib; optional: neuralforecast / darts.
- **Never run git commits.** The user makes all commits manually; do not add Claude as a commit author or co-author. Make and leave changes for the user to review and commit — never run `git commit`, `git add ... && commit`, or any commit command.
- Keep raw data, processed parquet, and large model artifacts out of git via `.gitignore`.

## Decision log
- Phase 1 (2026-06-19): Cleaned 60,670 → 60,667 rows (1 corrupt >1M-qty row dropped, 2 rows for product 43962-015 dropped, 139 Order Dates back-filled from Invoice Date, 0 rows lost to missing dates). 13,725 → 13,724 unique products (ProductID = first 9 chars). Parsed grade/pack_size/pack_type/region_seg from code suffix via regex (0 nulls). is_promo flags 6,827 rows (Sales USD ≤ 0). 140 return rows left in place for weekly netting (Phase 2). Pareto: 97 products = 50% of qty, 677 = 80%, 2,674 = 95% — extreme long-tail distribution confirms intermittent-demand-at-scale framing. Saved to data/processed/transactions_clean.parquet. Figures 01–08 saved to outputs/figures/.