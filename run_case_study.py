"""
This script runs the full pipeline:

1) Build spliced datasets (prices + returns) from the case tickers.
2) Run the portfolio optimizer with reasonable default constraints.
3) Save everything into ./data
"""

from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BUILD_SCRIPT = BASE_DIR / "build_spliced_dataset.py"
OPTIMIZER_SCRIPT = BASE_DIR / "optimize_portfolio.py"


def run_step(name: str, script: Path):
    print(f"\n==============================")
    print(f" STEP: {name}")
    print(f"==============================")

    if not script.exists():
        print(f"[ERROR] Script not found: {script}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=BASE_DIR,
    )

    if result.returncode != 0:
        print(f"[ERROR] {name} failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"[OK] {name} completed successfully")


def main():
    print("\n Running full case-study pipeline...\n")

    run_step("Build spliced dataset", BUILD_SCRIPT)
    run_step("Optimize portfolio", OPTIMIZER_SCRIPT)

    print("\n Pipeline finished.")
    print("Artifacts saved in ./data:")
    print("  • chosen_proxies_for_splicing.csv")
    print("  • spliced_price_panel_base_funds.csv")
    print("  • spliced_monthly_returns.csv")
    print("  • optimized_weights.csv\n")


if __name__ == "__main__":
    main()
