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
- Phase 5b (2026-06-20): Fitted 7 baselines (Naive, SeasonalNaive(52), WindowAverage(12), CrostonClassic, CrostonOptimized, CrostonSBA, TSB(0.3,0.3)) on all products using statsforecast 1.7.8 with n_jobs=-1. Two fixed-origin runs: VALID origin=2025-03-24 (12,287 series, 11.3s) and TEST origin=2025-06-16 (12,954 series, 6.5s); total wall-clock 18.6s. Output: 302,892 rows (wide, one col per model). y_true is 98.1% zero in TEST. Model behaviour: Naive/SeasonalNaive predict 96% zeros; WindowAverage 83% zeros; Croston variants and TSB are all non-zero (Croston mean ~40, TSB ~6 vs y_true mean ~9). SeasonalNaive NaN-fills (short series < 52 weeks) clipped to 0. Saved preds_baselines.parquet (738 KB). Figure 26 saved.
- Phase 5a (2026-06-20): Trained global LightGBM (tweedie, variance_power=1.2) on direct multi-step table. Training table: 13,586,560 train / 1,766,662 valid / 1,856,599 test rows (48 features after fix, 7 native categoricals). Split by target_week: VALID=2025-03-31→2025-06-16, TEST=2025-06-23→2025-09-08 (sealed). abc_class and demand_class removed from model inputs (whole-history leakage; kept for routing and segment eval only). Remaining static categoricals: grade, pack_size, pack_type, region_seg, Brand Category, Product Range, Sales Channel. Model converged at best_iteration=243 (early stopping patience=50, learning_rate=0.05, num_leaves=127, min_data_in_leaf=50). VALID RMSE=103.16, MAE=7.96 (vs null-model RMSE≈160 given 96.9% zeros in y). MAE rises gently h=1→12 (7.72→8.27), RMSE flat ≈100–109: expected for direct multi-step on sparse demand. Saved lgbm_global.txt, lgbm_eval_history.json, preds_global.parquet. Figures 22–25 saved.
- Phase 4 (2026-06-20): Built feature table (1,521,270 rows, 52 cols = 3 key/target + 49 features: 40 numeric, 9 categorical). Feature families: 7 lags (1/2/3/4/8/12/52w), rolling 4/8/12w mean/sum/std/nonzero_rate (all shift-1 within group), 5 intermittency/recency cols (weeks_since_last_sale_t, zero_streak, weeks_since_first_sale, cumulative_active_weeks, expanding_mean_demand), 6 calendar cols (woy/month/quarter/week_of_month + sin/cos Fourier period-52), 4 promo cols (sum+share × 4w/12w), 6 price cols (unit_price + usd_per_kg × 4w/12w + 12w_change; non-promo rows only). Static categoricals: grade, pack_size, pack_type, region_seg, Brand Category, Product Range, Sales Channel, demand_class, abc_class. Missingness: short lags ≈0.9% (first week of each product), qty_lag_52w 41.7%, unit_price features 72–97% (sparse non-promo transactions). No features dropped (none constant/all-null). Saved features_product.parquet (347 MB). Figures 18–21 saved. Route logic updated (Phase 3 correction): demand_class does NOT gate routing; routes are category_fallback (cold-start), global_model (ABC A+B), baseline_intermittent (ABC C). Updated route counts: global_model 2,644/94.7% vol, baseline_intermittent 10,310/4.8%, category_fallback 753/0.5%.
- Phase 7 (2026-06-21): Built per-product results table (13,707 rows × 25 cols). Columns: ProductID, abc_class, demand_class, route, champion_model, fcst_w1..w12, fcst_12wk_total, recent_avg_weekly (last 12w before test_origin), test_mase (per-product MASE; 12,223 evaluable, 484 NaN), active_week_rate, top_country. test_mase median: A=0.43, B=0.54, C=0.00 (95%+ are zero-demand). 12wk forecast totals (median): A=195 units, B=41 units, C=0 units. Saved results_by_product.parquet (1.0 MB) and forecast_by_product.csv (2.4 MB). Generated 7 forecast cards (Figs 33–39): top-3 A-class (68877-087 Lumpy/LightGBM, 53064-028 Intermittent/LightGBM, 14898-083 Intermittent/LightGBM), 1 typical-B (41884-086 Lumpy/CrostonOpt), 1 sparse-C (57955-024 Too-sparse/WindowAverage), 2 worst-MASE A/B (53704-013 MASE=1188, 11766-014 MASE=111 — both B×Lumpy served by CrostonOptimized). notebooks/07_product_results.ipynb created.
- Phase 6 (2026-06-20): Scored 8 models (LightGBM + 7 baselines) on fixed-origin TEST (155,448 rows, 12,954 products × 12 horizons) and VALID. Denominators from training window (before 2025-03-31); 64 products excluded from MASE/RMSSE (zero/constant history). Overall TEST RMSSE: LightGBM=38.63, Naive=39.37, WindowAverage=38.67, TSB=38.68, SeasonalNaive=38.74. Croston variants dramatically worse (67–70) due to massive over-forecast bias. Champions by (abc_class × demand_class): LightGBM wins all Intermittent+Lumpy segments (A/B/C); Naive wins Smooth/Erratic/Too-sparse (zero demand → Naive=0 is optimal); CrostonOptimized wins B×Lumpy; WindowAverage wins C×Too-sparse. Served forecast (segment champion mix): RMSSE=38.56, MASE=0.6991, WMAPE=1.80 — +2.1% skill vs Naive, +0.2% vs best single model. Products: LightGBM=6,926, WindowAverage=5,368, CrostonOptimized=450, Naive=210. WMAPE on A-class: LightGBM=1.41 (best). Figs 27–32 saved. src/metrics.py created.
- Phase 3 (2026-06-19): Segmented 13,707 products. ADI = mean inter-demand gap (weeks) between consecutive non-zero weeks; CV² = (std/mean)² with ddof=1, thresholds ADI=1.32/CV²=0.49. Demand classes: Intermittent 5,893 (72.1% vol), Lumpy 1,502 (25.3%), Too sparse 6,204 (2.0%), Smooth 65 (0.6%), Erratic 43 (0.1%). ABC: A=677 products (80.0% vol), B=1,999 (15.0%), C=11,031 (5.0%). Routes: global_model 1,901 products (69.3% vol), baseline_intermittent 11,053 (30.2%), category_fallback 753 cold-starts (0.5%). Cold-start threshold: first_sale_week > 2025-06-16 (12 weeks before panel end). Saved product_segments.parquet. Figures 13–17 saved.
- Phase 2 (2026-06-19): Built product-level weekly panel (1,521,270 rows, 13,707 products × 185 weeks) and product×country panel (1,542,724 rows, 13,949 pairs). ISO weeks Mon–Sun (Period W-SUN). Partial final week 2025-09-15 dropped (120 rows, 17 products existed only in that week). 96.27% of product-weeks are zeros (confirms extreme intermittent demand). Median active-week ratio 2.54%, median n_active_weeks = 2. net_qty clipped at weekly level (Rule 5). Saved weekly_product.parquet and weekly_product_country.parquet. Figures 09–12 saved.
- Phase 1 (2026-06-19): Cleaned 60,670 → 60,667 rows (1 corrupt >1M-qty row dropped, 2 rows for product 43962-015 dropped, 139 Order Dates back-filled from Invoice Date, 0 rows lost to missing dates). 13,725 → 13,724 unique products (ProductID = first 9 chars). Parsed grade/pack_size/pack_type/region_seg from code suffix via regex (0 nulls). is_promo flags 6,827 rows (Sales USD ≤ 0). 140 return rows left in place for weekly netting (Phase 2). Pareto: 97 products = 50% of qty, 677 = 80%, 2,674 = 95% — extreme long-tail distribution confirms intermittent-demand-at-scale framing. Saved to data/processed/transactions_clean.parquet. Figures 01–08 saved to outputs/figures/.