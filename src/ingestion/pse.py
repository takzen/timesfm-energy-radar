"""PSE (Polskie Sieci Elektroenergetyczne) API client for grid metrics and prices."""

import time
from datetime import date
from typing import Any

import httpx
import polars as pl
from loguru import logger

from src.config import settings


class PSEAPIError(Exception):
    """Raised when PSE API requests fail or return invalid responses."""


class PSEClient:
    """Client for interacting with the official PSE Open Data API (v2)."""

    def __init__(
        self,
        base_url: str = settings.pse_api_base_url,
        timeout: int = settings.pse_timeout_seconds,
        max_retries: int = settings.pse_max_retries,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={"Accept": "application/json", "User-Agent": "TimesFM-Energy-Radar/0.1.0"},
        )

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        self._client.close()

    def __enter__(self) -> "PSEClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _get_with_retry(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform GET request with exponential backoff on server/network errors."""
        last_exception: Exception | None = None
        backoff_delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("Requesting PSE API: url={}, attempt={}", url, attempt)
                response = self._client.get(url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    if not isinstance(data, dict):
                        raise PSEAPIError(f"Unexpected JSON payload type: {type(data)}")
                    return data

                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "Transient HTTP {} from PSE API on attempt {}/{}",
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

        raise PSEAPIError(
            f"Failed to fetch data from PSE API after {self.max_retries} attempts: {last_exception}"
        )

    def _fetch_all_pages(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch all pages from an OData-style endpoint following nextLink."""
        url: str | None = f"{self.base_url}/{endpoint}"
        results: list[dict[str, Any]] = []
        current_params: dict[str, Any] | None = params

        while url:
            data = self._get_with_retry(url, params=current_params)
            items = data.get("value", [])
            if not isinstance(items, list):
                raise PSEAPIError(
                    f"Expected list in 'value' field from endpoint {endpoint}, got {type(items)}"
                )

            results.extend(items)
            url = data.get("nextLink")
            # Clear params for nextLink as it already includes pagination query params
            current_params = None

        return results

    def fetch_grid_load_and_generation(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """Fetch national demand (KSE load) and generation breakdown from /his-wlk-cal endpoint.

        Parameters
        ----------
        start_date : str | date
            Start date (YYYY-MM-DD)
        end_date : str | date
            End date (YYYY-MM-DD)

        Returns
        -------
        pl.DataFrame
            Columns: timestamp_utc, timestamp_local, business_date, demand_mw, pv_mw,
            wind_mw, thermal_hydro_mw, cross_border_exchange_mw
        """
        start_str = str(start_date)
        end_str = str(end_date)
        logger.info(
            "Fetching PSE grid load and generation from {} to {}",
            start_str,
            end_str,
        )

        filter_expr = f"business_date ge '{start_str}' and business_date le '{end_str}'"
        records = self._fetch_all_pages(
            endpoint="his-wlk-cal",
            params={"$filter": filter_expr, "$first": 100},
        )

        if not records:
            logger.warning(
                "No PSE grid records found for date range {} to {}",
                start_str,
                end_str,
            )
            return pl.DataFrame(
                schema={
                    "timestamp_utc": pl.String,
                    "timestamp_local": pl.String,
                    "business_date": pl.String,
                    "demand_mw": pl.Float64,
                    "pv_mw": pl.Float64,
                    "wind_mw": pl.Float64,
                    "thermal_hydro_mw": pl.Float64,
                    "cross_border_exchange_mw": pl.Float64,
                }
            )

        df = pl.DataFrame(records)

        # Standardize and project columns
        selected = df.select(
            pl.col("dtime_utc").alias("timestamp_utc"),
            pl.col("dtime").alias("timestamp_local"),
            pl.col("business_date"),
            pl.col("demand").cast(pl.Float64).alias("demand_mw"),
            pl.col("pv").fill_null(0.0).cast(pl.Float64).alias("pv_mw"),
            pl.col("wi").fill_null(0.0).cast(pl.Float64).alias("wind_mw"),
            pl.col("jg").fill_null(0.0).cast(pl.Float64).alias("thermal_hydro_mw"),
            (pl.col("swm_p").fill_null(0.0) - pl.col("swm_np").fill_null(0.0))
            .cast(pl.Float64)
            .alias("cross_border_exchange_mw"),
        ).sort("timestamp_utc")

        return selected

    def fetch_balancing_prices(
        self,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """Fetch market clearing balancing prices (RCE) from /rce-pln endpoint.

        Parameters
        ----------
        start_date : str | date
            Start date (YYYY-MM-DD)
        end_date : str | date
            End date (YYYY-MM-DD)

        Returns
        -------
        pl.DataFrame
            Columns: timestamp_utc, timestamp_local, business_date, rce_pln_mwh
        """
        start_str = str(start_date)
        end_str = str(end_date)
        logger.info(
            "Fetching PSE balancing market prices (RCE) from {} to {}",
            start_str,
            end_str,
        )

        filter_expr = f"business_date ge '{start_str}' and business_date le '{end_str}'"
        records = self._fetch_all_pages(
            endpoint="rce-pln",
            params={"$filter": filter_expr, "$first": 100},
        )

        if not records:
            logger.warning(
                "No PSE price records found for date range {} to {}",
                start_str,
                end_str,
            )
            return pl.DataFrame(
                schema={
                    "timestamp_utc": pl.String,
                    "timestamp_local": pl.String,
                    "business_date": pl.String,
                    "rce_pln_mwh": pl.Float64,
                }
            )

        df = pl.DataFrame(records)

        selected = df.select(
            pl.col("dtime_utc").alias("timestamp_utc"),
            pl.col("dtime").alias("timestamp_local"),
            pl.col("business_date"),
            pl.col("rce_pln").cast(pl.Float64).alias("rce_pln_mwh"),
        ).sort("timestamp_utc")

        return selected
