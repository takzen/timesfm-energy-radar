"""Unit tests for cleaning, feature engineering, and dataset management."""

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.processing.clean import (
    clean_grid_data,
    clean_weather_data,
    merge_and_align_datasets,
)
from src.processing.dataset import (
    load_processed_dataset,
    query_duckdb,
    save_processed_dataset,
    time_series_split,
)
from src.processing.features import (
    add_calendar_features,
    add_lag_and_rolling_features,
    add_renewable_features,
    build_feature_pipeline,
    get_easter_sunday,
    is_polish_holiday,
)

# --- Holiday & Calendar Tests ---


def test_easter_computus() -> None:
    """Test Easter Sunday Computus calculation for known years."""
    assert get_easter_sunday(2024) == date(2024, 3, 31)
    assert get_easter_sunday(2025) == date(2025, 4, 20)
    assert get_easter_sunday(2026) == date(2026, 4, 5)


def test_polish_holidays() -> None:
    """Test Polish statutory holiday recognition."""
    assert is_polish_holiday(date(2024, 1, 1))  # Nowy Rok
    assert is_polish_holiday(date(2024, 5, 3))  # Święto Konstytucji 3 Maja
    assert is_polish_holiday(date(2024, 11, 11))  # Święto Niepodległości
    assert not is_polish_holiday(date(2024, 5, 4))  # Regular day


def test_calendar_features() -> None:
    """Test generation of calendar and cyclical features."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 5, 3, 12, 0, 0),  # Constitution day holiday at 12:00
                datetime(2024, 5, 4, 18, 0, 0),  # Saturday regular day at 18:00
            ]
        }
    )

    feat_df = add_calendar_features(df)
    assert "hour" in feat_df.columns
    assert "is_weekend" in feat_df.columns
    assert "is_holiday" in feat_df.columns
    assert "sin_hour" in feat_df.columns
    assert "cos_hour" in feat_df.columns

    assert feat_df["hour"].to_list() == [12, 18]
    assert feat_df["is_holiday"][0] == 1
    assert feat_df["is_holiday"][1] == 0
    assert feat_df["is_weekend"][1] == 1


# --- Cleaning & Alignment Tests ---


def test_clean_grid_data() -> None:
    """Test quarter-hourly to hourly resampling and clipping in clean_grid_data."""
    raw_grid = pl.DataFrame(
        {
            "timestamp_local": [
                "2024-06-14 00:15:00",
                "2024-06-14 00:30:00",
                "2024-06-14 00:45:00",
                "2024-06-14 01:00:00",
            ],
            "demand_mw": [16000.0, 16200.0, 16400.0, 16600.0],
            "pv_mw": [-5.0, 0.0, 0.0, 0.0],  # Negative reading to clip
            "wind_mw": [1000.0, 1000.0, 1200.0, 1200.0],
            "thermal_hydro_mw": [14000.0, 14000.0, 14000.0, 14000.0],
            "cross_border_exchange_mw": [200.0, 200.0, 200.0, 200.0],
        }
    )

    cleaned = clean_grid_data(raw_grid)
    assert cleaned.height == 1
    assert cleaned["demand_mw"][0] == 16300.0  # Mean of 4 quarters
    assert cleaned["pv_mw"][0] == 0.0  # Clipped negative value
    assert cleaned["wind_mw"][0] == 1100.0


def test_clean_weather_data() -> None:
    """Test weather cleaning and clipping."""
    raw_weather = pl.DataFrame(
        {
            "timestamp_local": ["2024-06-14T00:00", "2024-06-14T01:00"],
            "temperature_2m": [15.0, 14.5],
            "direct_normal_irradiance": [-2.0, 0.0],
            "wind_speed_10m": [3.0, 3.5],
        }
    )

    cleaned = clean_weather_data(raw_weather)
    assert cleaned.height == 2
    assert "timestamp" in cleaned.columns
    assert cleaned["direct_normal_irradiance"][0] == 0.0


def test_merge_and_align_datasets() -> None:
    """Test full alignment of grid, price, and weather datasets."""
    grid = pl.DataFrame(
        {
            "timestamp_local": ["2024-06-14 00:15:00", "2024-06-14 01:00:00"],
            "demand_mw": [16000.0, 16000.0],
            "pv_mw": [0.0, 0.0],
            "wind_mw": [1000.0, 1000.0],
        }
    )
    prices = pl.DataFrame(
        {
            "timestamp_local": ["2024-06-14 00:15:00", "2024-06-14 01:00:00"],
            "rce_pln_mwh": [350.0, 350.0],
        }
    )
    weather = pl.DataFrame(
        {
            "timestamp_local": ["2024-06-14T00:00"],
            "temperature_2m": [15.0],
        }
    )

    merged = merge_and_align_datasets(grid, prices, weather)
    assert merged.height >= 1
    assert "demand_mw" in merged.columns
    assert "rce_pln_mwh" in merged.columns
    assert "temperature_2m" in merged.columns


# --- Renewable & Lag Feature Tests ---


def test_renewable_features() -> None:
    """Test calculation of renewable totals, net load, and curtailment risk."""
    df = pl.DataFrame(
        {
            "demand_mw": [20000.0, 10000.0],
            "pv_mw": [2000.0, 6000.0],
            "wind_mw": [3000.0, 3000.0],
        }
    )

    featured = add_renewable_features(df)
    assert "total_renewable_mw" in featured.columns
    assert "net_load_mw" in featured.columns
    assert "renewable_penetration" in featured.columns
    assert "curtailment_risk_flag" in featured.columns

    # Row 0: 5000 / 20000 = 0.25 (no curtailment risk)
    assert featured["net_load_mw"][0] == 15000.0
    assert featured["renewable_penetration"][0] == 0.25
    assert featured["curtailment_risk_flag"][0] == 0

    # Row 1: 9000 / 10000 = 0.90 (curtailment risk >= 0.75)
    assert featured["net_load_mw"][1] == 1000.0
    assert featured["curtailment_risk_flag"][1] == 1


def test_lag_features() -> None:
    """Test lag and rolling feature creation."""
    # Create 30 rows
    df = pl.DataFrame(
        {
            "demand_mw": [10000.0 + i * 100 for i in range(30)],
        }
    )

    featured = add_lag_and_rolling_features(
        df, target_cols=["demand_mw"], lags=[24], rolling_windows=[24]
    )
    assert "demand_mw_lag_24h" in featured.columns
    assert "demand_mw_roll_mean_24h" in featured.columns
    assert featured["demand_mw_lag_24h"][24] == 10000.0


def test_build_feature_pipeline() -> None:
    """Test complete end-to-end pipeline execution."""
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 6, 14, 0, 0) + timedelta(hours=h) for h in range(25)],
            "demand_mw": [16000.0] * 25,
            "pv_mw": [500.0] * 25,
            "wind_mw": [1000.0] * 25,
            "rce_pln_mwh": [300.0] * 25,
        }
    )

    result = build_feature_pipeline(df, include_lags=True)
    assert "hour" in result.columns
    assert "net_load_mw" in result.columns
    assert "demand_mw_lag_24h" in result.columns


# --- Dataset Storage & Splitting Tests ---


def test_dataset_save_and_load(tmp_path: Path) -> None:
    """Test saving and loading processed datasets."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    file_path = tmp_path / "test_features.parquet"

    saved_path = save_processed_dataset(df, file_path)
    assert saved_path.exists()

    loaded_df = load_processed_dataset(saved_path)
    assert loaded_df.equals(df)


def test_query_duckdb(tmp_path: Path) -> None:
    """Test analytical SQL query over Parquet dataset via DuckDB."""
    df = pl.DataFrame({"metric": ["load", "load", "pv"], "val": [10.0, 20.0, 5.0]})
    file_path = tmp_path / "data.parquet"
    df.write_parquet(file_path)

    query = "SELECT metric, AVG(val) as avg_val FROM dataset GROUP BY metric ORDER BY metric"
    res = query_duckdb(query, parquet_path=file_path)

    assert res.height == 2
    assert res["metric"].to_list() == ["load", "pv"]
    assert res["avg_val"].to_list() == [15.0, 5.0]


def test_time_series_split() -> None:
    """Test chronological Train/Val/Test temporal splitting."""
    total_hours = 24 * 30  # 30 days = 720 hours
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0) + timedelta(hours=i) for i in range(total_hours)
            ],
            "demand": [15000.0] * total_hours,
        }
    )

    train_df, val_df, test_df = time_series_split(df, test_hours=168, val_hours=168)

    assert train_df.height == total_hours - 336
    assert val_df.height == 168
    assert test_df.height == 168

    # Ensure strictly monotonic chronological boundaries (no overlap or leakage)
    assert train_df["timestamp"].max() < val_df["timestamp"].min()  # type: ignore[operator]
    assert val_df["timestamp"].max() < test_df["timestamp"].min()  # type: ignore[operator]


def test_time_series_split_too_short() -> None:
    """Test that time_series_split raises error when dataset is too short."""
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 0)],
            "demand": [15000.0],
        }
    )
    with pytest.raises(ValueError, match="requires at least"):
        time_series_split(df, test_hours=168, val_hours=168)
