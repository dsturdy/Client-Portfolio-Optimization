"""
Build spliced price & return history for First Eagle case study

This script:

1) Reads the investable universe from case_inputs.xlsx
2) Downloads daily prices for those funds + proxy ETFs via yfinance
3) Builds proxy blends and computes monthly return correlations
4) Selects a best proxy for each short-history fund
5) Splices proxy history pre-inception using median price ratios
6) Outputs (into ./data):

Script is meant to be run once to produce a dataset that
the optimization engine and UI can load, it's run when run_case_study.py is run
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ================================================================
# 0. PATHS & GLOBAL CONFIG
# ================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CASE_FILE = BASE_DIR / "case_inputs.xlsx"

TARGET_START_DATE = pd.Timestamp("2015-12-01")
TARGET_END_DATE = pd.Timestamp("2025-11-30")   # from case prompt
MIN_YEARS_OVERLAP = 5.0                        # soft floor for splice overlap, otherwise later produces a warning
FERLX_CSV_PATH = DATA_DIR / "FERLX_adj_close.csv"  # manually adjusted its price for dividends so we pass its adj close rather than download it

# Funds that need splicing (shorter histories)
FUNDS_TO_SPLICE: List[str] = [
    "FDUIX", "FECRX", "FEREX", "FERLX", "FESCX", "FESMX", "SGOV"
]

# Benchmark tickers for benchmark comparison, not passed into the investable universe
BENCHMARK_TICKERS: List[str] = ["URTH", "AGG"]



def load_base_tickers_from_case(case_path: Path) -> List[str]:

    """Load the investable universe from the Excel case study file."""

    df_raw = pd.read_excel(case_path, sheet_name=0, header=None)

    # Find the row index where one of the cells is 'Ticker' or 'Symbol'
    header_row = None
    for i, row in df_raw.iterrows():
        vals = row.astype(str).str.strip().str.lower()
        if "ticker" in vals.values or "symbol" in vals.values:
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            f"Could not find a 'Ticker' or 'Symbol' header row in {case_path.name}"
        )

    # Re-read with the proper header row
    df = pd.read_excel(case_path, sheet_name=0, header=header_row)

    # Decide which column name to use
    if "Ticker" in df.columns:
        col = "Ticker"
    elif "Symbol" in df.columns:
        col = "Symbol"
    else:
        raise ValueError(
            f"Could not find a 'Ticker' or 'Symbol' column in {case_path.name}"
        )

    tickers = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )

    return tickers

# ------------------------------------------------
# Proxy specification (fund -> candidate proxies)
# ------------------------------------------------

"""
This includes the economic logic for which ETFs / blends can stand in
as a proxy for each private/short-history fund, and then later spliced
"""

proxy_spec: Dict[str, Dict[str, Dict]] = {
    # ------------------------------------------------
    # FDUIX — Short-Duration HY Muni
    # ------------------------------------------------
    "FDUIX": {
        "HYD":   {"type": "single", "tickers": ["HYD"],   "weights": [1.0], "use_for_splice": True},
        "ISHAX": {"type": "single", "tickers": ["ISHAX"], "weights": [1.0], "use_for_splice": True},
        "HYMB":  {"type": "single", "tickers": ["HYMB"],  "weights": [1.0], "use_for_splice": True},
        "SHYD":  {"type": "single", "tickers": ["SHYD"],  "weights": [1.0], "use_for_splice": True},
        "SMB":   {"type": "single", "tickers": ["SMB"],   "weights": [1.0], "use_for_splice": True},
    },

    # ------------------------------------------------
    # FECRX — Private Credit
    # ------------------------------------------------
    "FECRX": {
        "HYG":            {"type": "single", "tickers": ["HYG"],        "weights": [1.0], "use_for_splice": True},
        "JNK":            {"type": "single", "tickers": ["JNK"],        "weights": [1.0], "use_for_splice": True},
        "BKLN":           {"type": "single", "tickers": ["BKLN"],       "weights": [1.0], "use_for_splice": True},
        "SRLN":           {"type": "single", "tickers": ["SRLN"],       "weights": [1.0], "use_for_splice": True},
        "HYG_60_LQD_40":  {"type": "blend",  "tickers": ["HYG", "LQD"], "weights": [0.6, 0.4], "use_for_splice": True},
        "FTSL":           {"type": "single", "tickers": ["FTSL"],       "weights": [1.0], "use_for_splice": True},
        "BKLN_70_HYG_30": {"type": "blend",  "tickers": ["BKLN", "HYG"], "weights": [0.7, 0.3], "use_for_splice": True},
        "SRLN_50_HYG_50": {"type": "blend",  "tickers": ["SRLN", "HYG"], "weights": [0.5, 0.5], "use_for_splice": True},
    },

    # ------------------------------------------------
    # FEREX — Global Real Assets
    # ------------------------------------------------
    "FEREX": {
        "GNR":                   {"type": "single", "tickers": ["GNR"],          "weights": [1.0], "use_for_splice": True},
        "GUNR":                  {"type": "single", "tickers": ["GUNR"],         "weights": [1.0], "use_for_splice": True},
        "COMT":                  {"type": "single", "tickers": ["COMT"],         "weights": [1.0], "use_for_splice": True},
        "ACWI_50_COMT_50":       {"type": "blend",  "tickers": ["ACWI", "COMT"], "weights": [0.5, 0.5], "use_for_splice": True},
        "RINF":                  {"type": "single", "tickers": ["RINF"],         "weights": [1.0], "use_for_splice": True},
        "DBC":                   {"type": "single", "tickers": ["DBC"],          "weights": [1.0], "use_for_splice": True},
        "ACWI_40_COMT_40_GNR_20": {
            "type": "blend",
            "tickers": ["ACWI", "COMT", "GNR"],
            "weights": [0.4, 0.4, 0.2],
            "use_for_splice": True,
        },
    },

    # ------------------------------------------------
    # FERLX — Real Estate Debt / Private Credit (CRE loan / CLO style)
    # ------------------------------------------------
    "FERLX": {
        "HYG":             {"type": "single", "tickers": ["HYG"],   "weights": [1.0], "use_for_splice": True},
        "BKLN":            {"type": "single", "tickers": ["BKLN"],  "weights": [1.0], "use_for_splice": True},
        "JAAA":            {"type": "single", "tickers": ["JAAA"],  "weights": [1.0], "use_for_splice": True},
        "CLOA":            {"type": "single", "tickers": ["CLOA"],  "weights": [1.0], "use_for_splice": True},
        "CLOZ":            {"type": "single", "tickers": ["CLOZ"],  "weights": [1.0], "use_for_splice": True},
        "CLOI":            {"type": "single", "tickers": ["CLOI"],  "weights": [1.0], "use_for_splice": True},
        "AAA":             {"type": "single", "tickers": ["AAA"],   "weights": [1.0], "use_for_splice": True},
        "SPSB":            {"type": "single", "tickers": ["SPSB"],  "weights": [1.0], "use_for_splice": True},
        "VCSH":            {"type": "single", "tickers": ["VCSH"],  "weights": [1.0], "use_for_splice": True},
        "IGSB":            {"type": "single", "tickers": ["IGSB"],  "weights": [1.0], "use_for_splice": True},
        "USFR":            {"type": "single", "tickers": ["USFR"],  "weights": [1.0], "use_for_splice": True},
        "TFLO":            {"type": "single", "tickers": ["TFLO"],  "weights": [1.0], "use_for_splice": True},
        "JAAA_60_BKLN_40": {
            "type": "blend",
            "tickers": ["JAAA", "BKLN"],
            "weights": [0.6, 0.4],
            "use_for_splice": True,
        },
        "CLOA_50_BKLN_50": {
            "type": "blend",
            "tickers": ["CLOA", "BKLN"],
            "weights": [0.5, 0.5],
            "use_for_splice": True,
        },
        "BKLN_40_VCSH_40_USFR_20": {
            "type": "blend",
            "tickers": ["BKLN", "VCSH", "USFR"],
            "weights": [0.4, 0.4, 0.2],
            "use_for_splice": True,
        },
        "JAAA_70_SPSB_30": {
            "type": "blend",
            "tickers": ["JAAA", "SPSB"],
            "weights": [0.7, 0.3],
            "use_for_splice": True,
        },
    },

    # ------------------------------------------------
    # FESCX — Small Cap Value
    # ------------------------------------------------
    "FESCX": {
        "IWN": {"type": "single", "tickers": ["IWN"], "weights": [1.0], "use_for_splice": True},
        "VBR": {"type": "single", "tickers": ["VBR"], "weights": [1.0], "use_for_splice": True},
        "IJS": {"type": "single", "tickers": ["IJS"], "weights": [1.0], "use_for_splice": True},
    },

    # ------------------------------------------------
    # FESMX — SMID Blend
    # ------------------------------------------------
    "FESMX": {
        "IWM":           {"type": "single", "tickers": ["IWM"], "weights": [1.0], "use_for_splice": True},
        "IJH":           {"type": "single", "tickers": ["IJH"], "weights": [1.0], "use_for_splice": True},
        "IWM_50_IJH_50": {"type": "blend",  "tickers": ["IWM", "IJH"], "weights": [0.5, 0.5], "use_for_splice": True},
    },

    # ------------------------------------------------
    # SGOV — Cash / T-Bills
    # ------------------------------------------------
    "SGOV": {
        "BIL":  {"type": "single", "tickers": ["BIL"],  "weights": [1.0], "use_for_splice": True},
        "SHV":  {"type": "single", "tickers": ["SHV"],  "weights": [1.0], "use_for_splice": True},
        "^IRX": {"type": "single", "tickers": ["^IRX"], "weights": [1.0], "use_for_splice": True},
    },
}


def build_spliced_dataset() -> None:
    # ------------------------------------------------
    # 1. Load base tickers from the Excel case file
    # ------------------------------------------------
    base_tickers: List[str] = load_base_tickers_from_case(CASE_FILE)
    base_tickers = [t for t in base_tickers if t.upper() != "TICKER"]
    print("Base tickers from case file:", base_tickers)

    # Benchmark tickers for comparison
    benchmark_tickers = [t for t in BENCHMARK_TICKERS if t not in base_tickers]
    base_and_benchmark = base_tickers + benchmark_tickers
    print("Benchmark tickers:", benchmark_tickers)
    # ------------------------------------------------
    # 2. Build full ticker universe (base + all proxies/blends)
    # ------------------------------------------------
    all_tickers = set(base_and_benchmark)

    for fund, proxies in proxy_spec.items():
        for spec in proxies.values():
            for t in spec["tickers"]:
                all_tickers.add(t)

    all_tickers = sorted(all_tickers)
    print("Tickers to download:")
    print(all_tickers)

    # ------------------------------------------------
    # 3. Download daily adjusted close for everything
    # ------------------------------------------------
    data = yf.download(
        tickers=all_tickers,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=True,
    )["Adj Close"]

    # Drop days where EVERYTHING is NaN
    data = data.dropna(how="all")
    daily_prices = data.copy()

    # ------------------------------------------------
    # 3b. Override FERLX with local CSV if available because yfinance's adj close is wrong
    # ------------------------------------------------
    if FERLX_CSV_PATH.exists():
        ferlx_df = pd.read_csv(FERLX_CSV_PATH, parse_dates=["Date"])
        ferlx_df = ferlx_df.set_index("Date").sort_index()

        # Try various names to grab the adj close column
        for candidate_col in ["Adj Close", "AdjClose", "Close", "Price"]:
            if candidate_col in ferlx_df.columns:
                ferlx_series = ferlx_df[candidate_col].astype(float)
                break
        else:
            raise ValueError(
                f"Could not find an adjusted close column in {FERLX_CSV_PATH}. "
                "Expected one of: 'Adj Close', 'AdjClose', 'Close', 'Price'."
            )

        # Ensure FERLX column exists in the panel
        if "FERLX" not in daily_prices.columns:
            daily_prices["FERLX"] = np.nan

        # Expand index to include all FERLX dates
        combined_index = daily_prices.index.union(ferlx_series.index)
        daily_prices = daily_prices.reindex(combined_index)

        # Overwrite FERLX with CSV data (CSV wins wherever it has data)
        daily_prices.loc[ferlx_series.index, "FERLX"] = ferlx_series

        print(f"Overrode FERLX prices with CSV from {FERLX_CSV_PATH}")
    else:
        print(f"Warning: {FERLX_CSV_PATH} not found; using yfinance FERLX (if available).")

    # Now compute returns off the final, overridden daily_prices
    daily_returns = daily_prices.pct_change()

    # ------------------------------------------------
    # 4. Restrict to the target window for the case
    # ------------------------------------------------
    daily_prices = daily_prices[
        (daily_prices.index >= TARGET_START_DATE)
        & (daily_prices.index <= TARGET_END_DATE)
    ]
    daily_returns = daily_returns.loc[daily_prices.index]

    # ------------------------------------------------
    # 5. Monthly returns for correlation work
    # ------------------------------------------------
    monthly_prices = daily_prices.resample("ME").last()
    monthly_rets = monthly_prices.pct_change().dropna(how="all")

    # ------------------------------------------------
    # 6. Build blended proxy MONTHLY return series
    # ------------------------------------------------
    proxy_return_cols: Dict[str, Dict[str, str]] = {}
    proxy_meta: Dict[str, Dict[str, Dict]] = {}

    for fund, proxies in proxy_spec.items():
        proxy_return_cols[fund] = {}
        proxy_meta[fund] = {}

        for proxy_name, spec in proxies.items():
            col_name = f"{fund}_proxy_{proxy_name}"
            use_for_splice = spec.get("use_for_splice", True)

            if spec["type"] == "single":
                t = spec["tickers"][0]
                if t not in monthly_rets.columns:
                    continue
                monthly_rets[col_name] = monthly_rets[t]

            elif spec["type"] == "blend":
                tickers = spec["tickers"]
                weights = spec["weights"]
                blend_ret = 0.0
                missing = False
                for w, t in zip(weights, tickers):
                    if t not in monthly_rets.columns:
                        missing = True
                        break
                    blend_ret = blend_ret + w * monthly_rets[t]
                if missing:
                    continue
                monthly_rets[col_name] = blend_ret

            proxy_return_cols[fund][proxy_name] = col_name
            proxy_meta[fund][proxy_name] = {
                "use_for_splice": use_for_splice,
                "type": spec["type"],
                "tickers": spec["tickers"],
                "weights": spec["weights"],
            }

    # ------------------------------------------------
    # 7. Helper: correlations for a given fund vs its proxies
    # ------------------------------------------------
    def correlations_for_fund(fund: str, allowed_only: bool = False) -> pd.Series:
        if fund not in monthly_rets.columns:
            raise ValueError(f"{fund} not found in monthly_returns columns.")

        fund_ret = monthly_rets[fund].dropna()
        results = {}

        for proxy_name, col_name in proxy_return_cols.get(fund, {}).items():
            meta = proxy_meta[fund][proxy_name]
            if allowed_only and not meta["use_for_splice"]:
                continue
            if col_name not in monthly_rets.columns:
                continue

            df = pd.concat(
                [fund_ret, monthly_rets[col_name]],
                axis=1,
                join="inner",
            ).dropna()

            if len(df) > 0:
                results[proxy_name] = df.iloc[:, 0].corr(df.iloc[:, 1])
            else:
                results[proxy_name] = np.nan

        if not results:
            return pd.Series(dtype=float)

        return pd.Series(results).sort_values(ascending=False)

    # ------------------------------------------------
    # 8. Build DAILY proxy price series (for splicing step)
    # ------------------------------------------------
    daily_proxy_prices: Dict[str, Dict[str, pd.Series]] = {}

    for fund, proxies in proxy_spec.items():
        daily_proxy_prices[fund] = {}
        for proxy_name, spec in proxies.items():
            if spec["type"] == "single":
                t = spec["tickers"][0]
                if t not in daily_prices.columns:
                    continue
                s = daily_prices[t].dropna()
                daily_proxy_prices[fund][proxy_name] = s

            elif spec["type"] == "blend":
                tickers = spec["tickers"]
                weights = spec["weights"]
                blend_ret = 0.0
                missing = False
                for w, t in zip(weights, tickers):
                    if t not in daily_returns.columns:
                        missing = True
                        break
                    blend_ret = blend_ret + w * daily_returns[t]
                if missing:
                    continue
                blend_ret = blend_ret.dropna()
                blend_price = 100.0 * (1.0 + blend_ret).cumprod()
                daily_proxy_prices[fund][proxy_name] = blend_price

    # ------------------------------------------------
    # 9. Select best proxy for each fund
    # ------------------------------------------------
    def select_best_proxy_for_splicing(fund: str) -> str:
        corr_series = correlations_for_fund(fund, allowed_only=True)
        if corr_series.empty:
            raise ValueError(f"No allowed proxies for {fund}")

        candidates: List[Tuple[str, float]] = []

        for proxy_name in corr_series.index:
            if proxy_name not in daily_proxy_prices.get(fund, {}):
                continue
            proxy_s = daily_proxy_prices[fund][proxy_name].dropna()
            if len(proxy_s) == 0:
                continue
            first_date = proxy_s.index.min()
            if first_date <= TARGET_START_DATE:
                candidates.append((proxy_name, corr_series[proxy_name]))

        if candidates:
            # choose candidate with highest correlation among those with full history
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        else:
            # fall back to highest-correlation allowed proxy overall
            return corr_series.index[0]

    # ------------------------------------------------
    # 10. Splicing helper: combine proxy+fund prices
    # ------------------------------------------------
    def splice_fund_with_proxy(
        fund: str,
        proxy_name: str,
        start_date: pd.Timestamp,
    ) -> Tuple[pd.Series, float, float]:
        if fund not in daily_prices.columns:
            raise ValueError(f"{fund} not found in daily_prices.")

        fund_price = daily_prices[fund].dropna()
        proxy_price = daily_proxy_prices[fund][proxy_name].dropna()

        fund_price = fund_price[
            (fund_price.index >= start_date)
            & (fund_price.index <= TARGET_END_DATE)
        ]
        proxy_price = proxy_price[
            (proxy_price.index >= start_date)
            & (proxy_price.index <= TARGET_END_DATE)
        ]

        overlap_idx = fund_price.index.intersection(proxy_price.index)
        if len(overlap_idx) < 30:
            raise ValueError(
                f"Not enough overlap between {fund} and {proxy_name} for splicing."
            )

        fund_overlap = fund_price.loc[overlap_idx]
        proxy_overlap = proxy_price.loc[overlap_idx]

        overlap_years = (overlap_idx.max() - overlap_idx.min()).days / 365.25
        if overlap_years < MIN_YEARS_OVERLAP:
            print(
                f"Warning: only {overlap_years:.2f} years of overlap for "
                f"{fund} vs {proxy_name}"
            )

        ratio = fund_overlap / proxy_overlap
        scale = ratio.median()

        proxy_scaled = proxy_price * scale
        fund_start = fund_price.index.min()

        proxy_pre = proxy_scaled[
            (proxy_scaled.index < fund_start) & (proxy_scaled.index >= start_date)
        ]

        combined = pd.concat([proxy_pre, fund_price]).sort_index()
        combined = combined[
            (combined.index >= start_date)
            & (combined.index <= TARGET_END_DATE)
        ]

        return combined, float(scale), float(overlap_years)

    # ------------------------------------------------
    # 11. Perform splicing & collect documentation rows
    # ------------------------------------------------
    spliced_series: Dict[str, pd.Series] = {}
    chosen_proxy_rows: List[Dict] = []

    for fund in FUNDS_TO_SPLICE:
        if fund not in base_tickers:
            # in case the case file ever changes
            print(f"Skipping {fund}: not in case universe.")
            continue

        try:
            best_proxy = select_best_proxy_for_splicing(fund)
            print(f"\nSelected proxy for {fund}: {best_proxy}")
            s_spliced, scale_used, years_overlap = splice_fund_with_proxy(
                fund, best_proxy, TARGET_START_DATE
            )
            spliced_series[fund] = s_spliced

            corr_series = correlations_for_fund(fund, allowed_only=True)
            meta_proxy = proxy_meta[fund][best_proxy]

            chosen_proxy_rows.append(
                {
                    "fund": fund,
                    "proxy": best_proxy,
                    "correlation": corr_series.get(best_proxy, np.nan),
                    "first_proxy_date": daily_proxy_prices[fund][best_proxy]
                    .dropna()
                    .index.min(),
                    "overlap_years": years_overlap,
                    "scale_used": scale_used,
                    "proxy_type": meta_proxy["type"],
                    "proxy_tickers": ",".join(meta_proxy["tickers"]),
                    "proxy_weights": ",".join(str(w) for w in meta_proxy["weights"]),
                }
            )

        except Exception as e:
            print(f"Could not splice {fund}: {e}")

    if chosen_proxy_rows:
        chosen_df = pd.DataFrame(chosen_proxy_rows)
        chosen_out = DATA_DIR / "chosen_proxies_for_splicing.csv"
        chosen_df.to_csv(chosen_out, index=False)
        print(f"\nSaved chosen proxy summary -> {chosen_out}")

    # ------------------------------------------------
    # 12. Build full spliced DAILY price panel for base funds
    # ------------------------------------------------
    all_spliced: Dict[str, pd.Series] = {}

    for t in base_and_benchmark:
        if t in spliced_series:
            s = spliced_series[t]
        else:
            if t not in daily_prices.columns:
                print(f"Missing price history for {t}, skipping.")
                continue
            s = daily_prices[t].dropna()
            s = s[
                (s.index >= TARGET_START_DATE)
                & (s.index <= TARGET_END_DATE)
            ]
        all_spliced[t] = s


    spliced_panel = pd.concat(all_spliced, axis=1).sort_index()
    panel_spliced_path = DATA_DIR / "spliced_price_panel_base_funds.csv"
    spliced_panel.to_csv(panel_spliced_path)
    print(f"Saved spliced price panel for base funds -> {panel_spliced_path}")

    # ------------------------------------------------
    # 13. Build MONTHLY returns from the spliced panel (optimizer input)
    # ------------------------------------------------
    spliced_monthly_prices = spliced_panel.resample("ME").last()
    spliced_monthly_rets = spliced_monthly_prices.pct_change().dropna(how="all")

    spliced_monthly_path = DATA_DIR / "spliced_monthly_returns.csv"
    spliced_monthly_rets.to_csv(spliced_monthly_path)
    print(f"Saved spliced monthly returns -> {spliced_monthly_path}")

    print("\n[OK] Splicing complete — cleaned price & return history saved to /data and ready for the optimizer.")


if __name__ == "__main__":
    build_spliced_dataset()


