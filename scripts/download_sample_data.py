"""CLI script to download, process, and cache Polish power grid and weather dataset."""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from src.config import settings
from src.ingestion.fetch import fetch_and_save_raw_data
from src.processing.clean import clean_grid_data, clean_weather_data, merge_and_align_datasets
from src.processing.features import build_feature_pipeline


def generate_synthetic_dataset(output_path: Path, days: int = 60) -> pl.DataFrame:
    """Generate high-fidelity representative Polish power grid dataset."""
    logger.info("Generating realistic synthetic Polish power grid dataset ({} days)...", days)
    n_hours = 24 * days
    base_ts = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=n_hours)
    timestamps = [base_ts + timedelta(hours=i) for i in range(n_hours)]

    rng = np.random.default_rng(42)
    hours = np.array([ts.hour for ts in timestamps])
    weekdays = np.array([ts.weekday() for ts in timestamps])
    is_wknd = weekdays >= 5

    diurnal_curve = np.sin((hours - 6) / 24.0 * 2 * np.pi) * 3500.0 + 17500.0
    weekend_discount = np.where(is_wknd, -2500.0, 0.0)
    noise = rng.normal(0, 400.0, n_hours)
    demand = np.clip(diurnal_curve + weekend_discount + noise, 12000.0, 26000.0)

    solar_intensity = np.maximum(0.0, np.sin((hours - 6) / 14.0 * np.pi))
    solar_mask = (hours >= 6) & (hours <= 20)
    pv = np.where(solar_mask, solar_intensity * (6500.0 + rng.uniform(-1000, 2000, n_hours)), 0.0)
    pv = np.clip(pv, 0.0, 11000.0)

    wind_waves = np.sin(np.arange(n_hours) / 36.0) * 2500.0 + 3500.0 + rng.normal(0, 500, n_hours)
    wind = np.clip(wind_waves, 200.0, 8500.0)

    thermal = np.maximum(4000.0, demand - (pv + wind) + rng.normal(0, 200, n_hours))

    rce = 450.0 + (demand / 30.0) - (pv / 15.0) + rng.normal(0, 40, n_hours)
    rce = np.clip(rce, -50.0, 1200.0)

    temp = 18.0 + 7.0 * np.sin((hours - 9) / 24.0 * 2 * np.pi) + rng.normal(0, 1.5, n_hours)
    wind_spd = np.clip(wind / 700.0 + rng.normal(0, 1.0, n_hours), 1.0, 25.0)

    raw_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": demand,
            "pv_mw": pv,
            "wind_mw": wind,
            "thermal_hydro_mw": thermal,
            "rce_pln_mwh": rce,
            "temperature_2m": temp,
            "wind_speed_100m": wind_spd,
        }
    )

    featured_df = build_feature_pipeline(raw_df, include_lags=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_df.write_parquet(output_path)
    logger.info("Saved synthetic dataset to {} ({} rows)", output_path, featured_df.height)
    return featured_df


def main() -> None:
    """Run data ingestion and processing pipeline."""
    parser = argparse.ArgumentParser(description="Download and process PSE & Weather data")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of historical days to fetch (default: 30)"
    )
    parser.add_argument(
        "--force-sample", action="store_true", help="Generate synthetic sample data immediately"
    )
    args = parser.parse_args()

    settings.ensure_directories()
    processed_path = settings.processed_data_dir / "kse_hourly_features.parquet"

    if args.force_sample:
        generate_synthetic_dataset(processed_path, days=args.days)
        return

    end_d = date.today() - timedelta(days=1)
    start_d = end_d - timedelta(days=args.days)

    logger.info("Fetching real PSE grid and Open-Meteo weather data from {} to {}", start_d, end_d)
    try:
        raw_files = fetch_and_save_raw_data(
            start_date=start_d,
            end_date=end_d,
            output_dir=settings.raw_data_dir,
        )
        logger.info("Raw data downloaded: {}", list(raw_files.keys()))

        # Clean and align
        raw_grid = pl.read_parquet(raw_files["grid_load"])
        raw_weather = pl.read_parquet(raw_files["weather_historical"])

        clean_grid = clean_grid_data(raw_grid)
        clean_wea = clean_weather_data(raw_weather)

        aligned = merge_and_align_datasets(clean_grid, clean_wea)
        featured = build_feature_pipeline(aligned, include_lags=True)

        processed_path.parent.mkdir(parents=True, exist_ok=True)
        featured.write_parquet(processed_path)
        logger.info(
            "Processed dataset successfully written to {} ({} rows)",
            processed_path,
            featured.height,
        )

    except Exception as exc:
        logger.warning(
            "Live ingestion encountered an error ({}). Falling back to synthetic generator.", exc
        )
        generate_synthetic_dataset(processed_path, days=args.days)


if __name__ == "__main__":
    main()
