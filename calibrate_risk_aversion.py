from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from optimize_portfolio import (
    load_spliced_returns,
    backtest_quarterly_daily,
    RISK_AVERSION_MINVAR,
    RISK_AVERSION_SHARPE,
)


TARGET_VOL = {
    ("min_var", "Conservative"): 0.07,   # ~7% annual vol
    ("min_var", "Moderate"):     0.10,
    ("min_var", "Growth"):       0.13,
    ("max_sharpe_like", "Conservative"): 0.08,
    ("max_sharpe_like", "Moderate"):     0.11,
    ("max_sharpe_like", "Growth"):       0.14,
}

# Candidate λ values to try
LAMBDA_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]


def realized_annual_vol(backtest_result: dict) -> float:
    """
    Take the output of backtest_quarterly_daily and compute annualized vol
    from the daily NAV path.
    """
    nav = backtest_result["nav"]
    daily_ret = nav.pct_change().dropna()
    if daily_ret.empty:
        return float("nan")
    # Approx 252 trading days
    return float(daily_ret.std() * np.sqrt(252))


def main():
    rets = load_spliced_returns()

    profiles = ["Conservative", "Moderate", "Growth"]
    objectives = ["min_var", "max_sharpe_like"]

    print("\n=== λ calibration sweep (static, quarterly, full history) ===")

    for objective in objectives:
        for profile in profiles:
            target = TARGET_VOL[(objective, profile)]
            print(f"\n>>> Objective={objective}, Profile={profile}, target vol ≈ {target:.2%}")

            best_lam = None
            best_gap = float("inf")
            best_vol = float("nan")

            for lam in LAMBDA_GRID:
                # Temporarily set λ for this run
                if objective == "min_var":
                    RISK_AVERSION_MINVAR[profile] = lam
                else:
                    RISK_AVERSION_SHARPE[profile] = lam

                bt = backtest_quarterly_daily(
                    rets=rets,
                    mode="static",          # static optimized vs SGOV etc.
                    lookback_months=60,     # same as app defaults
                    start_months=None,
                    risk_profile=profile,
                    objective=objective,
                    equity_cap=None,
                    illiquid_cap=0.15,
                    turnover_budget=None,   # ignore turnover cap for calibration
                    include_real_assets=True,
                    include_non_us_equity=True,
                    include_short_duration_credit=True,
                )

                vol = realized_annual_vol(bt)
                gap = abs(vol - target)

                print(f"  λ={lam:5.2f} -> vol ≈ {vol:6.2%} (gap {gap:6.2%})")

                if gap < best_gap:
                    best_gap = gap
                    best_lam = lam
                    best_vol = vol

            print(
                f"*** Best λ for {objective}/{profile}: {best_lam:.2f} "
                f"(vol ≈ {best_vol:.2%}, target {target:.2%})"
            )

    print("\nNow copy the best λ values back into RISK_AVERSION_MINVAR / RISK_AVERSION_SHARPE in optimize_portfolio.py.")


if __name__ == "__main__":
    main()
