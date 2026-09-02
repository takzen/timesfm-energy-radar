"""Classical baseline forecasting models: Seasonal Naive, Persistence, and GBDT."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from loguru import logger
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.base import Forecaster, ForecastOutput


class SeasonalNaiveForecaster(Forecaster):
    """Seasonal Naive forecaster.

    Repeats observations from the previous period (e.g. 24h or 168h).
    """

    def __init__(self, season_length: int = 24, name: str | None = None) -> None:
        self.season_length = season_length
        self._custom_name = name or f"Seasonal Naive ({season_length}h)"
        self._last_season: np.ndarray | None = None
        self._residual_std: float = 0.0

    @property
    def name(self) -> str:
        return self._custom_name

    def fit(
        self,
        train_df: pl.DataFrame,
        target_col: str = "demand_mw",
        feature_cols: list[str] | None = None,
    ) -> "SeasonalNaiveForecaster":
        series = train_df[target_col].to_numpy().astype(np.float64)
        if len(series) >= self.season_length * 2:
            residuals = series[self.season_length :] - series[: -self.season_length]
            self._residual_std = float(np.std(residuals))
        else:
            self._residual_std = float(0.05 * np.mean(series))

        self._last_season = series[-self.season_length :]
        return self

    def forecast(
        self,
        context_df: pl.DataFrame,
        horizon: int = 24,
        target_col: str = "demand_mw",
        future_covariates_df: pl.DataFrame | None = None,
    ) -> ForecastOutput:
        series = context_df[target_col].to_numpy().astype(np.float64)
        history = series[-self.season_length :]
        reps = int(np.ceil(horizon / self.season_length))
        predictions = np.tile(history, reps)[:horizon]

        last_ts: datetime = context_df["timestamp"].max()  # type: ignore[assignment]
        future_timestamps = [last_ts + timedelta(hours=i) for i in range(1, horizon + 1)]

        # Expanding intervals
        expansion = np.sqrt(1 + np.arange(horizon) * (1.0 / self.season_length))
        q10 = predictions - 1.28 * self._residual_std * expansion
        q90 = predictions + 1.28 * self._residual_std * expansion

        return ForecastOutput(
            model_name=self.name,
            target_name=target_col,
            timestamps=future_timestamps,
            point_forecast=predictions.tolist(),
            quantiles={0.1: q10.tolist(), 0.5: predictions.tolist(), 0.9: q90.tolist()},
            metadata={"season_length": self.season_length},
        )


class PersistenceForecaster(Forecaster):
    """Last-known-value persistence forecaster."""

    @property
    def name(self) -> str:
        return "Persistence (Naive Last Value)"

    def fit(
        self,
        train_df: pl.DataFrame,
        target_col: str = "demand_mw",
        feature_cols: list[str] | None = None,
    ) -> "PersistenceForecaster":
        return self

    def forecast(
        self,
        context_df: pl.DataFrame,
        horizon: int = 24,
        target_col: str = "demand_mw",
        future_covariates_df: pl.DataFrame | None = None,
    ) -> ForecastOutput:
        last_val = float(context_df[target_col][-1])
        point_pred = [last_val] * horizon
        last_ts: datetime = context_df["timestamp"].max()  # type: ignore[assignment]
        future_timestamps = [last_ts + timedelta(hours=i) for i in range(1, horizon + 1)]

        diffs = np.diff(context_df[target_col].to_numpy().astype(np.float64))
        std = float(np.std(diffs)) if len(diffs) > 1 else last_val * 0.05
        expansion = np.sqrt(np.arange(1, horizon + 1))

        return ForecastOutput(
            model_name=self.name,
            target_name=target_col,
            timestamps=future_timestamps,
            point_forecast=point_pred,
            quantiles={
                0.1: [last_val - 1.28 * std * e for e in expansion],
                0.5: point_pred,
                0.9: [last_val + 1.28 * std * e for e in expansion],
            },
        )


class GBDTForecaster(Forecaster):
    """Autoregressive Gradient Boosted Decision Tree (HistGradientBoostingRegressor)."""

    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.08,
        max_depth: int = 8,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )
        self.model_q10 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.1,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )
        self.model_q90 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.9,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )
        self.feature_names: list[str] = []
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "LightGBM / GBDT Autoregressive"

    def fit(
        self,
        train_df: pl.DataFrame,
        target_col: str = "demand_mw",
        feature_cols: list[str] | None = None,
    ) -> "GBDTForecaster":
        # Drop rows with nulls in features (from lags)
        if feature_cols:
            self.feature_names = [
                c for c in feature_cols if c in train_df.columns and c != target_col
            ]
        else:
            self.feature_names = [
                c
                for c in train_df.select(pl.selectors.numeric()).columns
                if c != target_col and not c.startswith(f"{target_col}_pred")
            ]

        clean_train = train_df.drop_nulls(subset=[target_col, *self.feature_names])
        X = clean_train[self.feature_names].to_numpy()
        y = clean_train[target_col].to_numpy()

        logger.info(
            "Fitting GBDTForecaster with {} features on {} rows.", len(self.feature_names), len(y)
        )
        self.model.fit(X, y)
        self.model_q10.fit(X, y)
        self.model_q90.fit(X, y)
        self._is_fitted = True
        return self

    def forecast(
        self,
        context_df: pl.DataFrame,
        horizon: int = 24,
        target_col: str = "demand_mw",
        future_covariates_df: pl.DataFrame | None = None,
    ) -> ForecastOutput:
        if not self._is_fitted:
            raise RuntimeError("GBDTForecaster must be fit before forecasting.")

        last_ts: datetime = context_df["timestamp"].max()  # type: ignore[assignment]
        future_timestamps = [last_ts + timedelta(hours=i) for i in range(1, horizon + 1)]

        # Prepare feature vector for future steps
        if future_covariates_df is not None:
            available_feats = [f for f in self.feature_names if f in future_covariates_df.columns]
            if len(available_feats) == len(self.feature_names):
                X_future = future_covariates_df[self.feature_names].head(horizon).to_numpy()
            else:
                # Fill missing with latest known context row
                X_future = np.repeat(
                    context_df[self.feature_names].tail(1).to_numpy(), horizon, axis=0
                )
        else:
            X_future = np.repeat(context_df[self.feature_names].tail(1).to_numpy(), horizon, axis=0)

        preds = self.model.predict(X_future)
        q10_preds = self.model_q10.predict(X_future)
        q90_preds = self.model_q90.predict(X_future)

        return ForecastOutput(
            model_name=self.name,
            target_name=target_col,
            timestamps=future_timestamps,
            point_forecast=preds.tolist(),
            quantiles={
                0.1: q10_preds.tolist(),
                0.5: preds.tolist(),
                0.9: q90_preds.tolist(),
            },
        )
