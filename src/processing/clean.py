"""Cleaning, temporal alignment, and resampling pipelines for energy and weather data."""

from datetime import datetime

import polars as pl
from loguru import logger

from src.config import settings


def parse_and_localize_timestamp(
    df: pl.DataFrame,
    col_name: str,
    target_col: str = "timestamp",
    tz: str = settings.timezone,
) -> pl.DataFrame:
    """Parse string timestamps into Datetime with Europe/Warsaw timezone.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    col_name : str
        Source timestamp column name
    target_col : str
        Target column name for parsed datetime
    tz : str
        Target timezone

    Returns
    -------
    pl.DataFrame
        Dataframe with parsed datetime column
    """
    # Open-Meteo format: 'YYYY-MM-DDTHH:MM' (len 16)
    # PSE format: 'YYYY-MM-DD HH:MM:SS' (len 19)
    normalized_str = (
        pl.when(pl.col(col_name).str.len_chars() <= 16)
        .then(pl.col(col_name).str.replace("T", " ") + ":00")
        .otherwise(pl.col(col_name).str.replace("T", " "))
        .str.slice(0, 19)
    )

    return df.with_columns(normalized_str.str.to_datetime("%Y-%m-%d %H:%M:%S").alias(target_col))


def resample_quarter_hourly_to_hourly(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    aggregation_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Aggregate 15-minute intervals to hourly resolution using arithmetic mean.

    Parameters
    ----------
    df : pl.DataFrame
        Quarter-hourly input DataFrame
    timestamp_col : str
        Timestamp column name
    aggregation_cols : list[str] | None
        Numeric columns to aggregate (default: all numeric columns)

    Returns
    -------
    pl.DataFrame
        Hourly aggregated DataFrame indexed by the beginning of each hour
    """
    if df.is_empty():
        return df

    # Truncate timestamp to hour start
    # Note: 15-minute readings in PSE at 00:15, 00:30, 00:45, 01:00 (or 00:00-00:45)
    # If period end is used (e.g. 01:00 belongs to hour 0), we adjust timestamps so
    # each quarter-hour correctly maps to its hour bucket.
    # To do this cleanly: subtract 1 second before truncating if values represent period-end,
    # or if timestamps represent period start, truncate directly.
    # In PSE /his-wlk-cal, readings are 00:15, 00:30, 00:45, 01:00.
    # So 00:15..01:00 belong to the hour starting at 00:00.
    hourly_df = (
        df.with_columns(
            pl.col(timestamp_col).dt.offset_by("-1m").dt.truncate("1h").alias("hour_bucket")
        )
        .group_by("hour_bucket")
        .agg(
            [
                pl.col(col).mean().alias(col)
                for col in (aggregation_cols or df.select(pl.selectors.numeric()).columns)
                if col not in (timestamp_col, "hour_bucket")
            ]
        )
        .sort("hour_bucket")
        .rename({"hour_bucket": timestamp_col})
    )

    return hourly_df


def clean_grid_data(df: pl.DataFrame) -> pl.DataFrame:
    """Clean PSE grid load, PV, and wind generation metrics.

    - Parses timestamps
    - Aggregates 15-min readings to hourly averages
    - Enforces non-negative values for solar and wind generation
    - Validates grid demand

    Parameters
    ----------
    df : pl.DataFrame
        Raw PSE grid DataFrame

    Returns
    -------
    pl.DataFrame
        Cleaned hourly grid DataFrame
    """
    if df.is_empty():
        logger.warning("Empty grid DataFrame provided to clean_grid_data.")
        return df

    parsed = parse_and_localize_timestamp(df, "timestamp_local", "timestamp")

    numeric_cols = [
        col
        for col in [
            "demand_mw",
            "pv_mw",
            "wind_mw",
            "thermal_hydro_mw",
            "cross_border_exchange_mw",
        ]
        if col in parsed.columns
    ]

    hourly = resample_quarter_hourly_to_hourly(parsed, "timestamp", numeric_cols)

    # Clean non-negative generation and validate reasonable limits
    cleaned = hourly.with_columns(
        [
            pl.col("pv_mw").clip(lower_bound=0.0) if "pv_mw" in hourly.columns else pl.lit(0.0),
            pl.col("wind_mw").clip(lower_bound=0.0) if "wind_mw" in hourly.columns else pl.lit(0.0),
        ]
    )

    logger.info("Cleaned grid data: {} hourly rows.", cleaned.height)
    return cleaned


def clean_prices_data(df: pl.DataFrame) -> pl.DataFrame:
    """Clean PSE balancing market prices (RCE).

    Parameters
    ----------
    df : pl.DataFrame
        Raw PSE prices DataFrame

    Returns
    -------
    pl.DataFrame
        Cleaned hourly prices DataFrame
    """
    if df.is_empty():
        logger.warning("Empty prices DataFrame provided to clean_prices_data.")
        return df

    parsed = parse_and_localize_timestamp(df, "timestamp_local", "timestamp")
    numeric_cols = [col for col in ["rce_pln_mwh"] if col in parsed.columns]

    hourly = resample_quarter_hourly_to_hourly(parsed, "timestamp", numeric_cols)
    logger.info("Cleaned prices data: {} hourly rows.", hourly.height)
    return hourly


def clean_weather_data(df: pl.DataFrame) -> pl.DataFrame:
    """Clean Open-Meteo weather covariates.

    Parameters
    ----------
    df : pl.DataFrame
        Raw weather DataFrame

    Returns
    -------
    pl.DataFrame
        Cleaned hourly weather DataFrame
    """
    if df.is_empty():
        logger.warning("Empty weather DataFrame provided to clean_weather_data.")
        return df

    parsed = parse_and_localize_timestamp(df, "timestamp_local", "timestamp")

    # Clip solar radiation to non-negative values
    cleaned = parsed.with_columns(
        [
            pl.col(col).clip(lower_bound=0.0)
            for col in [
                "direct_normal_irradiance",
                "global_tilted_irradiance",
                "cloud_cover",
                "wind_speed_10m",
                "wind_speed_100m",
            ]
            if col in parsed.columns
        ]
    ).drop("timestamp_local")

    logger.info("Cleaned weather data: {} hourly rows.", cleaned.height)
    return cleaned


def build_continuous_hourly_index(start_dt: datetime, end_dt: datetime) -> pl.DataFrame:
    """Construct a full continuous sequence of hourly timestamps."""
    return pl.datetime_range(start_dt, end_dt, interval="1h", eager=True).to_frame("timestamp")


def merge_and_align_datasets(
    grid_df: pl.DataFrame,
    prices_df: pl.DataFrame,
    weather_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge grid, balancing price, and weather datasets along an aligned hourly timeline.

    Parameters
    ----------
    grid_df : pl.DataFrame
        Cleaned hourly grid DataFrame
    prices_df : pl.DataFrame
        Cleaned hourly prices DataFrame
    weather_df : pl.DataFrame
        Cleaned hourly weather DataFrame

    Returns
    -------
    pl.DataFrame
        Merged and interpolated dataset ready for feature engineering
    """
    cleaned_grid = clean_grid_data(grid_df)
    cleaned_prices = clean_prices_data(prices_df)
    cleaned_weather = clean_weather_data(weather_df)

    # Determine overlapping temporal bounds
    all_timestamps: list[datetime] = []
    for d in (cleaned_grid, cleaned_prices, cleaned_weather):
        if not d.is_empty() and "timestamp" in d.columns:
            all_timestamps.extend(d["timestamp"].to_list())

    if not all_timestamps:
        raise ValueError("Cannot merge datasets: all input DataFrames are empty.")

    min_ts = min(all_timestamps)
    max_ts = max(all_timestamps)

    logger.info(
        "Merging datasets on continuous timeline: {} to {} (expected {} hours)",
        min_ts,
        max_ts,
        int((max_ts - min_ts).total_seconds() // 3600) + 1,
    )

    timeline = build_continuous_hourly_index(min_ts, max_ts)

    # Progressive left joins onto continuous hourly index
    merged = timeline.join(cleaned_grid, on="timestamp", how="left")
    merged = merged.join(cleaned_prices, on="timestamp", how="left")
    merged = merged.join(cleaned_weather, on="timestamp", how="left")

    # Forward-fill / linear interpolate short gaps (e.g. DST adjustments or single missing hours)
    numeric_cols = [c for c in merged.select(pl.selectors.numeric()).columns if c != "timestamp"]

    interpolated = merged.with_columns(
        [
            pl.col(c).interpolate().fill_null(strategy="forward").fill_null(strategy="backward")
            for c in numeric_cols
        ]
    )

    logger.info(
        "Merged dataset created with {} rows and {} columns.",
        interpolated.height,
        len(interpolated.columns),
    )
    return interpolated
