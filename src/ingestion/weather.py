"""Open-Meteo weather API client for grid-scale solar, wind, and temperature covariates."""

import time
from datetime import date
from typing import Any

import httpx
import polars as pl
from loguru import logger

from src.config import settings


class WeatherAPIError(Exception):
    """Raised when weather API requests fail or return invalid data."""


class WeatherClient:
    """Client for querying historical reanalysis and forecast weather data from Open-Meteo."""

    HOURLY_VARIABLES: list[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "cloud_cover",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
        "wind_speed_10m",
        "wind_speed_100m",
    ]

    def __init__(
        self,
        archive_url: str = settings.open_meteo_archive_url,
        forecast_url: str = settings.open_meteo_forecast_url,
        timeout: int = settings.pse_timeout_seconds,
        max_retries: int = settings.pse_max_retries,
    ) -> None:
        self.archive_url = archive_url.rstrip("/")
        self.forecast_url = forecast_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={"Accept": "application/json", "User-Agent": "TimesFM-Energy-Radar/0.1.0"},
        )

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        self._client.close()

    def __enter__(self) -> "WeatherClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _get_with_retry(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send GET request with exponential backoff on network/server errors."""
        last_exception: Exception | None = None
        backoff_delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("Requesting Open-Meteo API: url={}, attempt={}", url, attempt)
                response = self._client.get(url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    if not isinstance(data, dict):
                        raise WeatherAPIError(f"Unexpected JSON payload type: {type(data)}")
                    return data

                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "Transient HTTP {} from Open-Meteo API on attempt {}/{}",
                        response.status_code,
                        attempt,
                        self.max_retries,
                    )
                else:
                    response.raise_for_status()

            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exception = exc
                logger.warning(
                    "Network/HTTP error on attempt {}/{}: {}",
                    attempt,
                    self.max_retries,
                    str(exc),
                )

            if attempt < self.max_retries:
                time.sleep(backoff_delay)
                backoff_delay *= 2.0

        raise WeatherAPIError(
            f"Failed to fetch weather data after {self.max_retries} attempts: {last_exception}"
        )

    def fetch_historical_weather(
        self,
        start_date: str | date,
        end_date: str | date,
        latitude: float = settings.poland_centroid_lat,
        longitude: float = settings.poland_centroid_lon,
        timezone: str = settings.timezone,
    ) -> pl.DataFrame:
        """Fetch historical weather reanalysis covariates.

        Parameters
        ----------
        start_date : str | date
            Start date (YYYY-MM-DD)
        end_date : str | date
            End date (YYYY-MM-DD)
        latitude : float
            Geographical latitude (default: Poland centroid)
        longitude : float
            Geographical longitude (default: Poland centroid)
        timezone : str
            Timezone identifier (default: Europe/Warsaw)

        Returns
        -------
        pl.DataFrame
            Hourly weather covariates
        """
        start_str = str(start_date)
        end_str = str(end_date)
        logger.info(
            "Fetching historical weather from Open-Meteo: {} to {} at ({}, {})",
            start_str,
            end_str,
            latitude,
            longitude,
        )

        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": ",".join(self.HOURLY_VARIABLES),
            "timezone": timezone,
        }

        data = self._get_with_retry(self.archive_url, params=params)
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            raise WeatherAPIError("Open-Meteo response missing 'hourly' time-series object.")

        series_dict: dict[str, list[Any]] = {"timestamp_local": hourly["time"]}
        for var_name in self.HOURLY_VARIABLES:
            series_dict[var_name] = hourly.get(var_name, [None] * len(hourly["time"]))

        df = pl.DataFrame(series_dict)
        return df.sort("timestamp_local")

    def fetch_forecast_weather(
        self,
        forecast_days: int = 7,
        latitude: float = settings.poland_centroid_lat,
        longitude: float = settings.poland_centroid_lon,
        timezone: str = settings.timezone,
    ) -> pl.DataFrame:
        """Fetch forecast weather covariates for upcoming horizon.

        Parameters
        ----------
        forecast_days : int
            Number of forecast days (1-16)
        latitude : float
            Geographical latitude
        longitude : float
            Geographical longitude
        timezone : str
            Timezone identifier

        Returns
        -------
        pl.DataFrame
            Hourly forecast covariates
        """
        logger.info(
            "Fetching forecast weather from Open-Meteo: {} days at ({}, {})",
            forecast_days,
            latitude,
            longitude,
        )

        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "forecast_days": forecast_days,
            "hourly": ",".join(self.HOURLY_VARIABLES),
            "timezone": timezone,
        }

        data = self._get_with_retry(self.forecast_url, params=params)
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            raise WeatherAPIError("Open-Meteo forecast missing 'hourly' time-series object.")

        series_dict: dict[str, list[Any]] = {"timestamp_local": hourly["time"]}
        for var_name in self.HOURLY_VARIABLES:
            series_dict[var_name] = hourly.get(var_name, [None] * len(hourly["time"]))

        df = pl.DataFrame(series_dict)
        return df.sort("timestamp_local")
