"""One-click CLI script to run systematic rolling-window forecasting benchmark."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.download_sample_data import generate_synthetic_dataset
from src.config import settings
from src.models.baselines import (
    GBDTForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from src.models.evaluation import RollingBacktest
from src.models.timesfm_model import TimesFMModel


def run_benchmark(
    target_col: str = "demand_mw",
    horizon: int = 24,
    stride: int = 24,
    min_train_days: int = 14,
    output_dir: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Execute rolling-window benchmark across all candidate forecasting engines."""
    processed_path = settings.processed_data_dir / "kse_hourly_features.parquet"

    if processed_path.exists():
        logger.info("Loading processed dataset from {}", processed_path)
        df = pl.read_parquet(processed_path)
    else:
        logger.warning(
            "Processed dataset not found at {}. Synthesizing 60-day dataset.", processed_path
        )
        df = generate_synthetic_dataset(processed_path, days=60)

    forecasters = [
        TimesFMModel(),
        GBDTForecaster(max_iter=50),
        SeasonalNaiveForecaster(season_length=24),
        PersistenceForecaster(),
    ]

    min_train_hours = min_train_days * 24
    logger.info(
        "Running rolling backtest (target={}, horizon={}h, stride={}h, min_train={}h)...",
        target_col,
        horizon,
        stride,
        min_train_hours,
    )

    backtest = RollingBacktest(
        horizon=horizon,
        stride=stride,
        min_train_hours=min_train_hours,
    )

    preds_df, leaderboard_df = backtest.run(
        df=df,
        forecasters=forecasters,
        target_col=target_col,
    )

    # Display Leaderboard
    print("\n" + "=" * 80)
    print(f"  ⚡ PSE FORECASTING BENCHMARK LEADERBOARD (Target: {target_col}, Horizon: {horizon}h)")
    print("=" * 80)
    print(f"{'Model':<32} | {'WAPE (%)':<10} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
    print("-" * 80)
    for row in leaderboard_df.iter_rows(named=True):
        print(
            f"{row['model']:<32} | "
            f"{row['WAPE']:<10.2f} | "
            f"{row['MAE']:<10.2f} | "
            f"{row['RMSE']:<10.2f} | "
            f"{row['MAPE']:<10.2f}"
        )
    print("=" * 80 + "\n")

    # Save reports if requested
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"benchmark_{target_col}_{horizon}h.csv"
        md_path = output_dir / f"benchmark_{target_col}_{horizon}h.md"

        leaderboard_df.write_csv(csv_path)

        best_model = leaderboard_df["model"][0]
        best_wape = leaderboard_df["WAPE"][0]

        md_content = f"""# 🏆 Forecasting Benchmark Report

- **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Target Variable**: `{target_col}`
- **Forecast Horizon**: `{horizon} hours`
- **Rolling Window Stride**: `{stride} hours`
- **Winner**: 🥇 **{best_model}** (WAPE: `{best_wape:.2f}%`)

## 📊 Leaderboard Summary

| Rank | Model | WAPE (%) | MAE | RMSE | MAPE (%) | Windows Evaluated |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
"""
        for rank, row in enumerate(leaderboard_df.iter_rows(named=True), start=1):
            badge = "🥇 " if rank == 1 else ("🥈 " if rank == 2 else ("🥉 " if rank == 3 else ""))
            md_content += (
                f"| {rank} | {badge}{row['model']} | {row['WAPE']:.2f}% | "
                f"{row['MAE']:,.1f} | {row['RMSE']:,.1f} | {row['MAPE']:.2f}% | "
                f"{row['evaluated_windows']} |\n"
            )

        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Saved benchmark reports to {} and {}", csv_path, md_path)

    return preds_df, leaderboard_df


def main() -> None:
    """CLI parser and entrypoint."""
    parser = argparse.ArgumentParser(description="Run PSE TimesFM Energy Radar benchmark")
    parser.add_argument(
        "--target-col",
        type=str,
        default="demand_mw",
        choices=["demand_mw", "rce_pln_mwh"],
        help="Target column to forecast (default: demand_mw)",
    )
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (default: 24h)")
    parser.add_argument("--stride", type=int, default=24, help="Rolling stride (default: 24h)")
    parser.add_argument(
        "--min-train-days", type=int, default=14, help="Min training days (default: 14)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports"), help="Output directory for reports"
    )
    args = parser.parse_args()

    run_benchmark(
        target_col=args.target_col,
        horizon=args.horizon,
        stride=args.stride,
        min_train_days=args.min_train_days,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
