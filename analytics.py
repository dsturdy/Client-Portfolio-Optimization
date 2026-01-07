# analytics.py
from __future__ import annotations

import numpy as np
import pandas as pd

from optimize_portfolio import (
    CASE_FILE,
    load_spliced_returns,
    estimate_moments,
)
from case_io import (
    load_current_portfolio_from_case,
    load_investable_universe_from_case,
)
from ticker_meta import TICKER_META


def get_returns_for_window(window: str = "full") -> pd.DataFrame:
    """
    Spliced monthly returns panel for the chosen window.
    Shared by current/optimized analytics.
    """
    rets = load_spliced_returns(CASE_FILE.parent / "data" / "spliced_monthly_returns.csv")
    # ^ if this path is weird, you can just call load_spliced_returns()
    #   without arguments if your original version didn't need ret_path.

    if window == "last5":
        cutoff = rets.index.max() - pd.DateOffset(years=5)
        rets = rets[rets.index >= cutoff]
    return rets


def get_current_portfolio(window: str = "full"):
    """
    Current portfolio weights + stats
    Returns:
        summary: dict of stats
        comp   : DataFrame indexed by ticker
    """
    cp_current = load_current_portfolio_from_case(CASE_FILE)
    universe = load_investable_universe_from_case(CASE_FILE)
    rets = get_returns_for_window(window)

    # Align tickers
    common_tickers = sorted(set(universe["Ticker"]) & set(rets.columns))
    if not common_tickers:
        raise ValueError("No overlap between investable universe and return panel.")

    universe = universe.set_index("Ticker").reindex(common_tickers).reset_index()

    if "Fund Name" in universe.columns and "FundName" not in universe.columns:
        universe = universe.rename(columns={"Fund Name": "FundName"})

    current_w_series = (
        cp_current.set_index("Ticker")["Weight"]
        .reindex(common_tickers)
        .fillna(0.0)
    )
    current_w = current_w_series.values

    mu, cov, rf = estimate_moments(rets[common_tickers])
    mu_vec = mu[common_tickers].values
    cov_mat = cov.loc[common_tickers, common_tickers].values

    port_ret = float(current_w @ mu_vec)
    port_vol = float(np.sqrt(current_w @ cov_mat @ current_w))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan

    summary = {
        "ExpectedReturn": port_ret,
        "Volatility": port_vol,
        "Sharpe": sharpe,
        "RiskFreeAnnual": rf,
    }

    comp = pd.DataFrame({"CurrentWeight": current_w}, index=common_tickers)

    cols = []
    if "AssetGroup" in universe.columns:
        cols.append("AssetGroup")
    if "FundName" in universe.columns:
        cols.append("FundName")

    if cols:
        comp = comp.join(universe.set_index("Ticker")[cols])

    if "FundName" in comp.columns:
        comp["Fund"] = comp["FundName"]
    else:
        comp["Fund"] = comp.index.map(lambda t: TICKER_META.get(t, {}).get("name", t))

    return summary, comp


def compute_group_risk_contrib(
    rets: pd.DataFrame,
    weights: pd.Series,
    comp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Contribution-to-risk by asset group for a given portfolio.
    If you increase its weight, how does it affect portfolio variance?
    Returns: DataFrame with columns:
        Weight (%) | Risk contribution (%)
    """
    tickers = list(comp.index)
    rets_sub = rets[tickers].dropna(how="all")

    _, cov, _ = estimate_moments(rets_sub)
    cov = cov.loc[tickers, tickers]

    w = weights.reindex(tickers).fillna(0.0).values
    Sigma = cov.values

    total_var = float(w @ Sigma @ w)
    if total_var <= 0:
        return pd.DataFrame(columns=["Asset group", "Weight (%)", "Risk contribution (%)"])

    marginal = Sigma @ w
    contrib_var = w * marginal
    contrib_share = contrib_var / total_var

    if "AssetGroup" in comp.columns:
        groups = comp["AssetGroup"]
    else:
        groups = pd.Series("Total", index=tickers)

    tmp = pd.DataFrame(
        {
            "AssetGroup": groups,
            "Weight": w,
            "RiskContribution": contrib_share,
        },
        index=tickers,
    )

    by_grp = tmp.groupby("AssetGroup")[["Weight", "RiskContribution"]].sum()
    by_grp["Weight (%)"] = 100 * by_grp["Weight"]
    by_grp["Risk contribution (%)"] = 100 * by_grp["RiskContribution"]

    out = by_grp[["Weight (%)", "Risk contribution (%)"]].sort_values(
        "Risk contribution (%)", ascending=False
    )
    out.index.name = "Asset group"
    return out


def compute_max_drawdown(
    rets: pd.DataFrame,
    weights: pd.Series,
    tickers: list[str],
) -> float:

    rets_sub = rets[tickers].dropna(how="all")
    if rets_sub.empty:
        return np.nan

    w = weights.reindex(tickers).fillna(0.0).values
    port_rets = rets_sub.values @ w
    if len(port_rets) == 0:
        return np.nan

    cum = np.cumprod(1.0 + port_rets)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(dd.min())  # negative number
