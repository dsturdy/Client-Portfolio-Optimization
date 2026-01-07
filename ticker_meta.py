"""
Already read in the ticker and their asset classes from the Excel file using case_io
but just list them here for convenience
"""

TICKER_META = {
    # ---- Equities ----
    "FEVIX": {
        "name": "First Eagle US I",
        "asset_class": "equity_us_large",
        "liquidity": "daily",
    },
    "FESCX": {
        "name": "First Eagle Small Cap Opportunity I",
        "asset_class": "equity_us_smid",
        "liquidity": "daily",
    },
    "SGOIX": {
        "name": "First Eagle Overseas I",
        "asset_class": "equity_non_us",
        "liquidity": "daily",
    },
    "FEGIX": {
        "name": "First Eagle Global I",
        "asset_class": "equity_non_us",
        "liquidity": "daily",
    },
    "FEHIX": {
        "name": "First Eagle High Yield I",
        "asset_class": "equity_non_us",
        "liquidity": "daily",
    },
    "FESMX": {
        "name": "First Eagle SMID Cap Opportunity I",
        "asset_class": "equity_us_smid",
        "liquidity": "daily",
    },
    "FEAIX": {
        "name": "First Eagle Global Income Builder I",
        "asset_class": "equity_us_large",
        "liquidity": "daily",
    },

    # ---- Allocation / balanced ----
    "SGIIX": {
        "name": "First Eagle Global Income Builder I",
        "asset_class": " llocation",
        "liquidity": "daily",
    },
    "FEBIX": {
        "name": "First Eagle Global Income Builder II",
        "asset_class": "allocation",
        "liquidity": "daily",
    },

    # ---- Fixed income sleeves ----
    "FDUIX": {
        "name": "First Eagle Short Dur Hi Yld Muni I",
        "asset_class": "fixed_short_credit",
        "liquidity": "daily",
    },
    "LQD": {
        "name": "iShares iBoxx $ Invmt Grade Corp Bd ETF",
        "asset_class": "fixed_ig",
        "liquidity": "daily",
    },
    "HYG": {
        "name": "iShares iBoxx $ High Yield Corp Bd ETF",
        "asset_class": "fixed_hy",
        "liquidity": "daily",
    },

    # ---- Private credit / illiquid ----
    "FECRX": {
        "name": "First Eagle Credit Opportunities I",
        "asset_class": "private_credit",
        "liquidity": "quarterly",
    },
    "FERLX": {
        "name": "First Eagle Real Estate Debt I",
        "asset_class": "private_credit",
        "liquidity": "quarterly",
    },

    # ---- Real assets ----
    "FEREX": {
        "name": "First Eagle Real Assets I",
        "asset_class": "real_assets",
        "liquidity": "daily",
    },

    # ---- Cash / T-bills ----
    "SGOV": {
        "name": "iShares 0–3 Month Treasury Bond ETF",
        "asset_class": "cash",
        "liquidity": "daily",
    },
    "SHV": {
        "name": "iShares Short Treasury Bond ETF",
        "asset_class": "cash",
        "liquidity": "daily",
    },

    # ---- Benchmark Tickers ----
    "URTH": {
        "name": "iShares MSCI World ETF",
        "asset_class": "benchmark_equity",
        "liquidity": "daily",
    },
    "AGG": {
        "name": "iShares Core US Aggregate Bond ETF",
        "asset_class": "benchmark_core_bond",
        "liquidity": "daily",
    },

}
