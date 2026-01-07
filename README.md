# Conceptual Proof – Asset Allocation Optimization Program

This repo contains my solution for the First Eagle take-home case study:
a **rules-based asset allocation optimizer** plus a **minimal UI** for
running scenarios and visualizing analytics.

At a high level, the pipeline is:

1. Read portfolio and investable-universe inputs from `case_inputs.xlsx`.
2. Build a **spliced return history** using ETF proxies for short-history sleeves.
3. Run a **mean–variance optimizer** with transparent, rule-based constraints.
4. Visualize the current vs optimized portfolio and historical backtests
   in a small Streamlit app.

---

## Repo layout

- `case_inputs.xlsx` – case file provided by First Eagle
- `build_spliced_dataset.py` – builds daily spliced prices and monthly returns:
  - downloads prices via `yfinance`
  - constructs ETF / blend proxies
  - chooses best proxy by correlation + history coverage
  - splices pre-inception history using a median price ratio
- `case_io.py` – helpers to parse the Excel sheet:
  - current portfolio weights and asset-group bands
  - full investable universe and asset-group labels
- `ticker_meta.py` – metadata for each fund (asset class, liquidity).
- `optimize_portfolio.py` – core optimization engine and backtest helpers:
  - mean–variance moments from monthly returns
  - convex optimization using `cvxpy`
  - quarterly rebalancing backtest on daily NAV series
- `analytics.py` – shared analytics utilities:
  - current portfolio stats
  - risk contribution by asset group
  - max drawdown calculation
- `app.py` – Streamlit UI (Part 2 of the case):
  - sidebar controls for risk profile / objectives / constraints
  - current vs optimized portfolio tables and charts
- `run_case_study.py` – one-command driver:
  - runs `build_spliced_dataset.py` then `optimize_portfolio.py`
  - writes all generated datasets and optimization outputs into `./data/`

The `data/` folder (created on first run) will contain:

- `chosen_proxies_for_splicing.csv`
- `spliced_price_panel_base_funds.csv`
- `spliced_monthly_returns.csv`
- `optimized_weights.csv`

---
## Setup

> **First:** make sure you are in the project folder (the one containing  
> `run_case_study.py` and `app.py`). If you downloaded a ZIP, unzip it and
> open a terminal in that folder (or `cd` into the pathname):
>
> ```bash
> cd First_Eagle_Case_Study
> ```

Create a virtual environment and install dependencies:

### macOS / Linux

```bash
python -m venv fe_case_env   # use python3 here if python is not found
source fe_case_env/bin/activate
pip install -r requirements.txt
```

### Windows

```bat
python -m venv fe_case_env
fe_case_env\Scripts\activate
pip install -r requirements.txt
```

---

## Run the project

### macOS & Windows

#### Build the dataset & run the optimizer

```bash
python run_case_study.py
```

#### Launch the UI

```bash
python -m streamlit run app.py
```

