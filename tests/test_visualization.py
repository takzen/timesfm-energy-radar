"""Unit tests for visualization components and dashboard helper functions."""

from datetime import datetime, timedelta

import plotly.graph_objects as go
import polars as pl

from src.app import get_or_create_sample_dataset
from src.visualization.charts import (
    plot_curtailment_gauge,
    plot_forecast_with_intervals,
    plot_generation_mix,
    plot_wape_comparison,
)


def test_plot_forecast_with_intervals() -> None:
    """Test Plotly forecast figure structure and trace counts."""
    base_ts = datetime(2024, 6, 1, 0, 0)
    timestamps = [base_ts + timedelta(hours=i) for i in range(24)]
    point_forecast = [16000.0 + i * 10 for i in range(24)]
    q10 = [15000.0 + i * 10 for i in range(24)]
    q90 = [17000.0 + i * 10 for i in range(24)]

    fig = plot_forecast_with_intervals(
        timestamps=timestamps,
        point_forecast=point_forecast,
        q10=q10,
        q90=q90,
        actual_timestamps=timestamps[:12],
        actual_values=point_forecast[:12],
    )

    assert isinstance(fig, go.Figure)
    # Expected traces: Actuals, Q90 upper, Q10 lower filled, Point forecast
    assert len(fig.data) == 4
    assert fig.layout.paper_bgcolor == "#131722"


def test_plot_generation_mix() -> None:
    """Test stacked generation mix chart generation."""
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 6, 1, 0, 0) + timedelta(hours=i) for i in range(10)],
            "demand_mw": [16000.0] * 10,
            "pv_mw": [1000.0] * 10,
            "wind_mw": [2000.0] * 10,
            "thermal_hydro_mw": [13000.0] * 10,
        }
    )

    fig = plot_generation_mix(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # Thermal, Wind, PV, Demand line


def test_plot_curtailment_gauge() -> None:
    """Test gauge meter creation for various risk tiers."""
    fig_normal = plot_curtailment_gauge(0.45, alert_level="NORMAL")
    assert isinstance(fig_normal, go.Figure)
    assert fig_normal.data[0].value == 45.0

    fig_crit = plot_curtailment_gauge(0.95, alert_level="CRITICAL")
    assert isinstance(fig_crit, go.Figure)
    assert fig_crit.data[0].value == 95.0


def test_plot_wape_comparison() -> None:
    """Test WAPE bar chart creation from leaderboard DataFrame."""
    lb_df = pl.DataFrame(
        {
            "model": ["Google TimesFM", "LightGBM", "Seasonal Naive"],
            "WAPE": [4.2, 5.1, 7.8],
            "MAE": [650.0, 780.0, 1100.0],
        }
    )

    fig = plot_wape_comparison(lb_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["Google TimesFM", "LightGBM", "Seasonal Naive"]


def test_get_or_create_sample_dataset() -> None:
    """Test realistic sample dataset synthesis for app startup."""
    df = get_or_create_sample_dataset()
    assert isinstance(df, pl.DataFrame)
    assert df.height > 100
    assert "demand_mw" in df.columns
    assert "pv_mw" in df.columns
    assert "wind_mw" in df.columns
    assert "rce_pln_mwh" in df.columns
    assert "hour" in df.columns
