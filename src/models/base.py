"""Abstract base classes and schemas for time-series forecasters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl


@dataclass(slots=True, kw_only=True)
class ForecastOutput:
    """Standardized output container for time-series model forecasts."""

    model_name: str
    target_name: str
    timestamps: list[datetime]
    point_forecast: list[float]
    quantiles: dict[float, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pl.DataFrame:
        """Convert forecast output into a clean Polars DataFrame."""
        data: dict[str, Any] = {
            "timestamp": self.timestamps,
            f"{self.target_name}_pred": self.point_forecast,
        }

        # Include available quantiles sorted by alpha
        for alpha in sorted(self.quantiles.keys()):
            pct = int(alpha * 100)
            data[f"{self.target_name}_q{pct}"] = self.quantiles[alpha]

        return pl.DataFrame(data)


class Forecaster(ABC):
    """Abstract Base Class defining the contract for all forecasters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique human-readable name of the forecaster."""

    @abstractmethod
    def fit(
        self,
        train_df: pl.DataFrame,
        target_col: str = "demand_mw",
        feature_cols: list[str] | None = None,
    ) -> "Forecaster":
        """Fit the forecaster on historical time-series data.

        For zero-shot foundation models (like TimesFM), this may be a no-op or set context.
        """

    @abstractmethod
    def forecast(
        self,
        context_df: pl.DataFrame,
        horizon: int = 24,
        target_col: str = "demand_mw",
        future_covariates_df: pl.DataFrame | None = None,
    ) -> ForecastOutput:
        """Generate forecasts for the specified horizon using historical context and covariates."""
