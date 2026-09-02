"""Ingestion orchestrator and CLI for fetching and storing raw energy and weather datasets."""

import argparse
from datetime import date, timedelta
from pathlib import Path

from loguru import logger

from src.config import settings
from src.ingestion.pse import PSEClient
from src.ingestion.weather import WeatherClient


def fetch_and_save_raw_data(
    start_date: str | date,
    end_date: str | date,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Orchestrate ingestion of PSE grid metrics, prices, and weather data into Parquet files.

    Parameters
    ----------
    start_date : str | date
        Start date in YYYY-MM-DD format
    end_date : str | date
        End date in YYYY-MM-DD format
    output_dir : Path | None
        Target directory to write raw Parquet files (default: settings.raw_data_dir)

    Returns
    -------
    dict[str, Path]
        Paths to generated Parquet files keyed by dataset type
    """
    start_str = str(start_date)
    end_str = str(end_date)
    target_dir = output_dir or settings.raw_data_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initiating raw data ingestion for period {} to {}", start_str, end_str)

    # 1. PSE Grid Load and Generation
    with PSEClient() as pse_client:
        df_grid = pse_client.fetch_grid_load_and_generation(start_str, end_str)
        grid_path = target_dir / f"pse_grid_{start_str}_{end_str}.parquet"
        df_grid.write_parquet(grid_path)
        logger.info(
            "Saved PSE grid data: rows={}, path={}",
            df_grid.height,
            grid_path,
        )

        # 2. PSE Balancing Prices (RCE)
        df_prices = pse_client.fetch_balancing_prices(start_str, end_str)
        prices_path = target_dir / f"pse_prices_{start_str}_{end_str}.parquet"
        df_prices.write_parquet(prices_path)
        logger.info(
            "Saved PSE prices data: rows={}, path={}",
            df_prices.height,
            prices_path,
        )

    # 3. Weather Covariates from Open-Meteo
    with WeatherClient() as weather_client:
        df_weather = weather_client.fetch_historical_weather(start_str, end_str)
        weather_path = target_dir / f"weather_{start_str}_{end_str}.parquet"
        df_weather.write_parquet(weather_path)
        logger.info(
            "Saved weather data: rows={}, path={}",
            df_weather.height,
            weather_path,
        )

    return {
        "grid": grid_path,
        "prices": prices_path,
        "weather": weather_path,
    }


def main() -> None:
    """CLI entrypoint for data ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest PSE National Grid and Open-Meteo weather datasets.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to ingest up to yesterday if dates are not provided (default: 7)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom destination directory for raw Parquet files",
    )

    args = parser.parse_args()

    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        # Default to previous N days up to yesterday
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=args.days - 1)
        start_date = start.isoformat()
        end_date = end.isoformat()

    logger.info("Executing ingestion CLI: {} -> {}", start_date, end_date)
    saved_files = fetch_and_save_raw_data(
        start_date=start_date,
        end_date=end_date,
        output_dir=args.output_dir,
    )

    logger.info("Ingestion completed successfully:")
    for key, path in saved_files.items():
        logger.info("  - {}: {}", key, path)


if __name__ == "__main__":
    main()
