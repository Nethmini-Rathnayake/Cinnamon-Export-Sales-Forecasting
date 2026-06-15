# Cinnamon Export Sales — Weekly Demand Forecasting

Forecasts **weekly Sales Qty 12 weeks ahead** for ~13,725 cinnamon products
(identified by the first 9 characters of `Product Code`), with optional
product × country drill-downs reconciled via hierarchical forecasting.

---

## Directory structure

```
.
├── data/
│   ├── raw/              # Source XLSX — gitignored, never committed
│   └── processed/        # Cleaned parquet / feather files — gitignored
├── notebooks/            # EDA and experiment notebooks
├── outputs/
│   ├── figures/          # Numbered PNGs (01_*.png …) at 140 dpi — gitignored
│   └── forecasts/        # 12-week forecast CSV / Excel — gitignored
├── reports/              # Written report and slide deck
├── src/                  # Reusable Python modules
│   └── __init__.py
├── .gitignore
├── CLAUDE.md             # AI collaboration spec
├── requirements.txt      # Pinned dependencies (Python 3.11+)
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (Python 3.11+ required)
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) install neural forecasting support
pip install neuralforecast==1.7.4
```

---

## Data placement

Place the raw data file at:

```
data/raw/Cinnamon_export_sales.xlsx
```

This path is gitignored — do **not** commit the raw data.

---

## Modelling approach

Demand is extremely sparse / intermittent (89 % of products have ≤ 10
lifetime transactions). The strategy pools all series into a single global
**LightGBM** model with lag / rolling / calendar / intermittency features,
benchmarked against Croston, SBA, TSB, Seasonal-Naive, and Moving Average
baselines. See `CLAUDE.md` for full specification.
