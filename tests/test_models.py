"""Unit tests for forecasting models, TimesFM wrapper, baselines, and evaluation."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.models.base import ForecastOutput
from src.models.baselines import (
    GBDTForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from src.models.evaluation import (
    RollingBacktest,
    classify_curtailment_risk,
    compute_metrics,
    mae,
    pinball_loss,
    rmse,
    wape,
)
from src.models.timesfm_model import TimesFMModel


def _create_synthetic_series(n_hours: int = 120) -> pl.DataFrame:
    """Helper to generate continuous hourly synthetic energy time-series."""
    base_ts = datetime(2024, 6, 1, 0, 0)
    timestamps = [base_ts + timedelta(hours=i) for i in range(n_hours)]
    # Daily 24h cycle
    demand = [15000.0 + 3000.0 * (i % 24) / 24.0 for i in range(n_hours)]
    pv = [
        max(0.0, 4000.0 * ((i % 24) - 6) / 12.0) if 6 <= (i % 24) <= 18 else 0.0
        for i in range(n_hours)
    ]
    wind = [2000.0 + 500.0 * (i % 12) for i in range(n_hours)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": demand,
            "pv_mw": pv,
            "wind_mw": wind,
            "hour": [i % 24 for i in range(n_hours)],
            "demand_mw_lag_24h": [demand[max(0, i - 24)] for i in range(n_hours)],
        }
    )


# --- ForecastOutput Schema Tests ---


def test_forecast_output_to_dataframe() -> None:
    """Test schema conversion from ForecastOutput to Polars DataFrame."""
    now = datetime(2024, 6, 1, 12, 0)
    timestamps = [now + timedelta(hours=i) for i in range(1, 4)]
    out = ForecastOutput(
        model_name="TestModel",
        target_name="demand_mw",
        timestamps=timestamps,
        point_forecast=[16000.0, 16200.0, 16100.0],
        quantiles={0.1: [15000.0, 15100.0, 15050.0], 0.9: [17000.0, 17200.0, 17150.0]},
    )

    df = out.to_dataframe()
    assert df.height == 3
    assert "demand_mw_pred" in df.columns
    assert "demand_mw_q10" in df.columns
    assert "demand_mw_q90" in df.columns
    assert df["demand_mw_pred"][0] == 16000.0


# --- Forecaster Tests ---


def test_seasonal_naive_forecaster() -> None:
    """Test Seasonal Naive 24h forecasting logic."""
    df = _create_synthetic_series(48)
    model = SeasonalNaiveForecaster(season_length=24)
    model.fit(df, target_col="demand_mw")

    out = model.forecast(context_df=df, horizon=24, target_col="demand_mw")
    assert len(out.point_forecast) == 24
    assert 0.1 in out.quantiles
    assert 0.9 in out.quantiles
    # Horizon 24 should mirror last 24h of history
    expected_first = float(df["demand_mw"][-24])
    assert pytest.approx(out.point_forecast[0], rel=1e-3) == expected_first


def test_persistence_forecaster() -> None:
    """Test persistence naive forecaster."""
    df = _create_synthetic_series(30)
    model = PersistenceForecaster()
    model.fit(df, target_col="demand_mw")

    out = model.forecast(context_df=df, horizon=12, target_col="demand_mw")
    assert len(out.point_forecast) == 12
    last_val = float(df["demand_mw"][-1])
    assert all(p == last_val for p in out.point_forecast)


def test_gbdt_forecaster() -> None:
    """Test GBDT autoregressive model training and inference."""
    df = _create_synthetic_series(72)
    model = GBDTForecaster(max_iter=30)
    model.fit(df, target_col="demand_mw", feature_cols=["hour", "demand_mw_lag_24h"])

    out = model.forecast(context_df=df, horizon=12, target_col="demand_mw")
    assert len(out.point_forecast) == 12
    assert 0.1 in out.quantiles
    assert 0.9 in out.quantiles
    # Point prediction between lower and upper quantile
    assert out.quantiles[0.1][0] <= out.point_forecast[0] <= out.quantiles[0.9][0]


def test_timesfm_forecaster() -> None:
    """Test TimesFM zero-shot wrapper execution and intervals."""
    df = _create_synthetic_series(48)
    model = TimesFMModel()
    model.fit(df, target_col="demand_mw")

    out = model.forecast(context_df=df, horizon=24, target_col="demand_mw")
    assert len(out.point_forecast) == 24
    assert len(out.timestamps) == 24
    assert 0.1 in out.quantiles
    assert 0.9 in out.quantiles
    assert out.quantiles[0.1][0] <= out.quantiles[0.9][0]


# --- Metrics & Evaluation Tests ---


def test_metrics_computation() -> None:
    """Test error metrics computation (MAE, RMSE, WAPE, Pinball)."""
    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 190.0, 300.0]  # errors: +10, -10, 0

    assert mae(y_true, y_pred) == pytest.approx(20.0 / 3.0)
    assert wape(y_true, y_pred) == pytest.approx((20.0 / 600.0) * 100.0)
    assert rmse(y_true, y_pred) == pytest.approx((200.0 / 3.0) ** 0.5)

    q_loss = pinball_loss(y_true, [90.0, 180.0, 280.0], alpha=0.9)
    assert q_loss > 0.0

    all_metrics = compute_metrics(y_true, y_pred, quantiles={0.9: [120.0, 210.0, 310.0]})
    assert "MAE" in all_metrics
    assert "WAPE" in all_metrics
    assert "Pinball_q90" in all_metrics


def test_rolling_backtest() -> None:
    """Test rolling backtesting engine generating predictions and leaderboard."""
    df = _create_synthetic_series(72)
    models = [SeasonalNaiveForecaster(season_length=24), PersistenceForecaster()]

    backtest = RollingBacktest(horizon=12, stride=12, min_train_hours=36)
    pred_df, leaderboard_df = backtest.run(df, forecasters=models, target_col="demand_mw")

    assert pred_df.height > 0
    assert "model" in pred_df.columns
    assert "actual" in pred_df.columns
    assert "prediction" in pred_df.columns

    assert leaderboard_df.height == 2
    assert "WAPE" in leaderboard_df.columns
    assert "MAE" in leaderboard_df.columns


def test_curtailment_risk_classifier() -> None:
    """Test renewable curtailment classification thresholds."""
    df = pl.DataFrame(
        {
            "demand_mw": [10000.0, 10000.0, 10000.0],
            "pv_mw": [1000.0, 4000.0, 6000.0],
            "wind_mw": [1000.0, 4000.0, 3500.0],
        }
    )
    # Row 0: 2000 MW / 10000 MW = 20% -> NORMAL
    # Row 1: 8000 MW / 10000 MW = 80% -> WATCH (>= 75%)
    # Row 2: 9500 MW / 10000 MW = 95% -> CRITICAL (>= 90%)

    classified = classify_curtailment_risk(df, threshold=0.75)
    assert classified["curtailment_alert_level"].to_list() == ["NORMAL", "WATCH", "CRITICAL"]
