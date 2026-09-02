"""Unit tests for PSE and Open-Meteo ingestion pipelines with mocked HTTP responses."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest

from src.ingestion.fetch import fetch_and_save_raw_data
from src.ingestion.pse import PSEAPIError, PSEClient
from src.ingestion.weather import WeatherAPIError, WeatherClient

# --- PSE Client Tests ---


def test_pse_fetch_grid_load_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful fetching and formatting of PSE grid load and generation."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "value": [
            {
                "dtime_utc": "2024-06-14 00:00:00",
                "dtime": "2024-06-14 02:00:00",
                "business_date": "2024-06-14",
                "demand": 16500.5,
                "pv": 0.0,
                "wi": 1250.2,
                "jg": 14000.0,
                "swm_p": 500.0,
                "swm_np": 100.0,
            },
            {
                "dtime_utc": "2024-06-14 01:00:00",
                "dtime": "2024-06-14 03:00:00",
                "business_date": "2024-06-14",
                "demand": 15800.0,
                "pv": 10.5,
                "wi": 1300.0,
                "jg": 13500.0,
                "swm_p": 400.0,
                "swm_np": 150.0,
            },
        ],
        "nextLink": None,
    }

    client = PSEClient(max_retries=2)
    monkeypatch.setattr(client._client, "get", lambda *args, **kwargs: mock_response)

    df = client.fetch_grid_load_and_generation("2024-06-14", "2024-06-14")

    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert "demand_mw" in df.columns
    assert "pv_mw" in df.columns
    assert "wind_mw" in df.columns
    assert "cross_border_exchange_mw" in df.columns
    assert df["demand_mw"][0] == 16500.5
    assert df["cross_border_exchange_mw"][0] == 400.0  # 500 - 100


def test_pse_fetch_balancing_prices_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful fetching of PSE balancing prices."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "value": [
            {
                "dtime_utc": "2024-06-14 00:00:00",
                "dtime": "2024-06-14 02:00:00",
                "business_date": "2024-06-14",
                "rce_pln": 345.50,
            }
        ],
        "nextLink": None,
    }

    client = PSEClient(max_retries=1)
    monkeypatch.setattr(client._client, "get", lambda *args, **kwargs: mock_response)

    df = client.fetch_balancing_prices("2024-06-14", "2024-06-14")
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    assert "rce_pln_mwh" in df.columns
    assert df["rce_pln_mwh"][0] == 345.50


def test_pse_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior when PSE returns empty value list."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"value": [], "nextLink": None}

    client = PSEClient(max_retries=1)
    monkeypatch.setattr(client._client, "get", lambda *args, **kwargs: mock_response)

    df = client.fetch_grid_load_and_generation("2024-01-01", "2024-01-01")
    assert isinstance(df, pl.DataFrame)
    assert df.height == 0
    assert "demand_mw" in df.columns


def test_pse_retry_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test retry behavior and eventual failure when PSE API returns error."""
    mock_response = MagicMock()
    mock_response.status_code = 503

    client = PSEClient(max_retries=2)
    monkeypatch.setattr(client._client, "get", lambda *args, **kwargs: mock_response)

    with pytest.raises(PSEAPIError, match="Failed to fetch data from PSE API"):
        client.fetch_grid_load_and_generation("2024-06-14", "2024-06-14")


# --- Weather Client Tests ---


def test_weather_fetch_historical_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful querying of historical weather covariates."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2024-06-14T00:00", "2024-06-14T01:00"],
            "temperature_2m": [15.2, 14.8],
            "relative_humidity_2m": [80.0, 82.0],
            "cloud_cover": [20.0, 30.0],
            "direct_normal_irradiance": [0.0, 0.0],
            "global_tilted_irradiance": [0.0, 0.0],
            "wind_speed_10m": [3.5, 4.1],
            "wind_speed_100m": [6.2, 7.0],
        }
    }

    client = WeatherClient(max_retries=1)
    monkeypatch.setattr(client._client, "get", lambda *args, **kwargs: mock_response)

    df = client.fetch_historical_weather("2024-06-14", "2024-06-14")
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert "temperature_2m" in df.columns
    assert "wind_speed_100m" in df.columns
    assert df["temperature_2m"][0] == 15.2


def test_weather_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that failed Open-Meteo query raises WeatherAPIError."""
    client = WeatherClient(max_retries=1)

    def raise_transport(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(client._client, "get", raise_transport)

    with pytest.raises(WeatherAPIError):
        client.fetch_historical_weather("2024-06-14", "2024-06-14")


# --- Orchestrator Tests ---


def test_fetch_and_save_raw_data(tmp_path: Path) -> None:
    """Test complete ingestion orchestrator pipeline writing Parquet files."""
    dummy_grid = pl.DataFrame({"timestamp_utc": ["2024-06-14 00:00:00"], "demand_mw": [16000.0]})
    dummy_prices = pl.DataFrame({"timestamp_utc": ["2024-06-14 00:00:00"], "rce_pln_mwh": [320.0]})
    dummy_weather = pl.DataFrame(
        {"timestamp_local": ["2024-06-14T00:00"], "temperature_2m": [15.0]}
    )

    with (
        patch.object(PSEClient, "fetch_grid_load_and_generation", return_value=dummy_grid),
        patch.object(PSEClient, "fetch_balancing_prices", return_value=dummy_prices),
        patch.object(WeatherClient, "fetch_historical_weather", return_value=dummy_weather),
    ):
        result = fetch_and_save_raw_data(
            start_date="2024-06-14",
            end_date="2024-06-14",
            output_dir=tmp_path,
        )

        assert "grid" in result
        assert "prices" in result
        assert "weather" in result

        for path in result.values():
            assert path.exists()
            loaded_df = pl.read_parquet(path)
            assert loaded_df.height == 1
