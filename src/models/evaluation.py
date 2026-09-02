"""Evaluation suite: time-series metrics, rolling-window backtesting, and curtailment risk."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from src.models.base import Forecaster


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def wape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Weighted Absolute Percentage Error (WAPE %): sum(|y - y_pred|) / sum(y) * 100."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    total_y = np.sum(np.abs(yt))
    if total_y == 0:
        return 0.0
    return float(np.sum(np.abs(yt - yp)) / total_y * 100.0)


def mape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-5) -> float:
    """Mean Absolute Percentage Error (MAPE %)."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(np.abs(yt) < eps, eps, np.abs(yt))
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def pinball_loss(y_true: Sequence[float], y_quantile: Sequence[float], alpha: float) -> float:
    """Quantile Pinball Loss for interval calibration."""
    yt, yq = np.asarray(y_true), np.asarray(y_quantile)
    diff = yt - yq
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    quantiles: Mapping[float, Sequence[float]] | None = None,
) -> dict[str, float]:
    """Compute all point and probabilistic forecast evaluation metrics."""
    metrics: dict[str, float] = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }

    if quantiles:
        for alpha, q_pred in quantiles.items():
            pct = int(alpha * 100)
            metrics[f"Pinball_q{pct}"] = pinball_loss(y_true, q_pred, alpha)

    return metrics


@dataclass(slots=True, kw_only=True)
class RollingBacktest:
    """Rolling-window cross-validation and evaluation engine for time-series forecasters."""

    horizon: int = 24
    stride: int = 24
    min_train_hours: int = 24 * 14  # 14 days minimum context

    def run(
        self,
        df: pl.DataFrame,
        forecasters: Sequence[Forecaster],
        target_col: str = "demand_mw",
        feature_cols: list[str] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Execute rolling-window backtest across all forecasters.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            (predictions_df, summary_leaderboard_df)
        """
        sorted_df = df.sort("timestamp")
        total_rows = sorted_df.height
        required_rows = self.min_train_hours + self.horizon

        if total_rows < required_rows:
            raise ValueError(
                f"Dataset size ({total_rows}) is smaller than min_train_hours + horizon "
                f"({required_rows})"
            )

        window_starts = list(
            range(self.min_train_hours, total_rows - self.horizon + 1, self.stride)
        )
        logger.info(
            "Executing backtest over {} rolling windows (horizon={}h, stride={}h)",
            len(window_starts),
            self.horizon,
            self.stride,
        )

        all_predictions: list[dict[str, Any]] = []
        model_scores: dict[str, list[dict[str, float]]] = {f.name: [] for f in forecasters}

        for step_idx, split_idx in enumerate(window_starts):
            train_window = sorted_df.slice(0, split_idx)
            test_window = sorted_df.slice(split_idx, self.horizon)
            actuals = test_window[target_col].to_list()
            test_timestamps = test_window["timestamp"].to_list()

            for forecaster in forecasters:
                try:
                    forecaster.fit(train_window, target_col=target_col, feature_cols=feature_cols)
                    out = forecaster.forecast(
                        context_df=train_window,
                        horizon=self.horizon,
                        target_col=target_col,
                        future_covariates_df=test_window,
                    )
                    metrics = compute_metrics(actuals, out.point_forecast, out.quantiles)
                    model_scores[forecaster.name].append(metrics)

                    for i in range(self.horizon):
                        pred_record = {
                            "window_idx": step_idx,
                            "model": forecaster.name,
                            "timestamp": test_timestamps[i],
                            "actual": actuals[i],
                            "prediction": out.point_forecast[i],
                        }
                        for alpha, q_vals in out.quantiles.items():
                            pct = int(alpha * 100)
                            pred_record[f"q{pct}"] = q_vals[i]
                        all_predictions.append(pred_record)

                except Exception as exc:
                    logger.warning(
                        "Error evaluating {} in window {}: {}", forecaster.name, step_idx, exc
                    )

        predictions_df = pl.DataFrame(all_predictions)

        # Build leaderboard
        leaderboard_rows: list[dict[str, Any]] = []
        for model_name, score_list in model_scores.items():
            if not score_list:
                continue
            avg_metrics = {
                "model": model_name,
                "MAE": float(np.mean([s["MAE"] for s in score_list])),
                "RMSE": float(np.mean([s["RMSE"] for s in score_list])),
                "WAPE": float(np.mean([s["WAPE"] for s in score_list])),
                "MAPE": float(np.mean([s["MAPE"] for s in score_list])),
                "evaluated_windows": len(score_list),
            }
            leaderboard_rows.append(avg_metrics)

        leaderboard_df = pl.DataFrame(leaderboard_rows).sort("WAPE")
        return predictions_df, leaderboard_df


def classify_curtailment_risk(
    df: pl.DataFrame,
    demand_col: str = "demand_mw",
    pv_col: str = "pv_mw",
    wind_col: str = "wind_mw",
    threshold: float = 0.75,
) -> pl.DataFrame:
    """Classify grid renewable overgeneration and curtailment risk alerts.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with demand and renewable generation metrics
    demand_col : str
        Name of demand column
    pv_col : str
        Name of solar PV generation column
    wind_col : str
        Name of wind generation column
    threshold : float
        Penetration ratio above which alert is triggered (default: 0.75 = 75% of demand)

    Returns
    -------
    pl.DataFrame
        DataFrame with risk_level ('NORMAL', 'WATCH', 'CRITICAL') and risk_score
    """
    total_re = pl.col(pv_col) + pl.col(wind_col)
    penetration = total_re / pl.when(pl.col(demand_col) > 0).then(pl.col(demand_col)).otherwise(1.0)

    # Risk level classification
    risk_level_expr = (
        pl.when(penetration >= 0.90)
        .then(pl.lit("CRITICAL"))
        .when(penetration >= threshold)
        .then(pl.lit("WATCH"))
        .otherwise(pl.lit("NORMAL"))
        .alias("curtailment_alert_level")
    )

    return df.with_columns(
        [
            total_re.alias("total_renewable_mw"),
            penetration.alias("renewable_penetration_ratio"),
            risk_level_expr,
        ]
    )
