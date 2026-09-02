"""Plotly interactive visualization components using TradingView / FinTech modern styling."""

from collections.abc import Sequence
from datetime import datetime

import plotly.graph_objects as go
import polars as pl

# TradingView / FinTech Pro Dark Theme Palette
THEME = {
    "paper_bg": "#131722",
    "plot_bg": "#1e222d",
    "grid_color": "#2a2e39",
    "text_color": "#f8fafc",
    "text_muted": "#94a3b8",
    "actual_color": "#00f0ff",  # Neon Cyan
    "forecast_color": "#a855f7",  # Electric Violet
    "interval_fill": "rgba(168, 85, 247, 0.16)",
    "pv_color": "#f59e0b",  # Amber Solar
    "wind_color": "#10b981",  # Emerald Wind
    "thermal_color": "#475569",  # Slate Conventional
    "demand_color": "#ff3b69",  # Neon Coral
}


def plot_forecast_with_intervals(
    timestamps: Sequence[datetime],
    point_forecast: Sequence[float],
    q10: Sequence[float] | None = None,
    q90: Sequence[float] | None = None,
    actual_timestamps: Sequence[datetime] | None = None,
    actual_values: Sequence[float] | None = None,
    target_name: str = "demand_mw",
    model_name: str = "Google TimesFM",
    title: str = "Power Grid Demand Forecast with Uncertainty Bands",
    unit: str = "MW",
) -> go.Figure:
    """Generate an interactive Plotly chart with point forecast and 10%-90% prediction intervals."""
    fig = go.Figure()

    # 1. Historical / Actuals line (Neon Cyan)
    if actual_timestamps is not None and actual_values is not None and len(actual_values) > 0:
        fig.add_trace(
            go.Scatter(
                x=list(actual_timestamps),
                y=list(actual_values),
                mode="lines+markers",
                name="Actual",
                line={"color": THEME["actual_color"], "width": 2.5},
                marker={"size": 4, "color": THEME["actual_color"]},
                hovertemplate="<b>Actual</b>: %{y:,.1f} "
                + unit
                + "<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            )
        )

    # 2. Uncertainty intervals (filled area between Q10 and Q90 in Electric Violet)
    if q10 is not None and q90 is not None and len(q10) > 0 and len(q90) > 0:
        ts_list = list(timestamps)
        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=ts_list,
                y=list(q90),
                mode="lines",
                line={"width": 0},
                name="90% Quantile",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Lower bound with fill
        fig.add_trace(
            go.Scatter(
                x=ts_list,
                y=list(q10),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=THEME["interval_fill"],
                name="10%-90% Prediction Band",
                hoverinfo="skip",
            )
        )

    # 3. Point forecast line (Electric Violet)
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(point_forecast),
            mode="lines+markers",
            name=f"{model_name} Forecast",
            line={"color": THEME["forecast_color"], "width": 3, "dash": "solid"},
            marker={"size": 5, "color": THEME["forecast_color"]},
            hovertemplate="<b>"
            + model_name
            + "</b>: %{y:,.1f} "
            + unit
            + "<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 18, "color": THEME["text_color"]}},
        xaxis={
            "title": "Time (Europe/Warsaw)",
            "gridcolor": THEME["grid_color"],
            "showgrid": True,
            "zeroline": False,
        },
        yaxis={
            "title": f"Value ({unit})",
            "gridcolor": THEME["grid_color"],
            "showgrid": True,
            "zeroline": False,
        },
        template="plotly_dark",
        paper_bgcolor=THEME["paper_bg"],
        plot_bgcolor=THEME["plot_bg"],
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": THEME["text_color"]},
        },
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        hovermode="x unified",
    )
    return fig


def plot_generation_mix(
    df: pl.DataFrame,
    title: str = "PSE Generation Mix Breakdown vs Demand",
) -> go.Figure:
    """Stacked area chart comparing PV, Wind, and Thermal/Hydro generation against Grid Demand."""
    fig = go.Figure()

    timestamps = df["timestamp"].to_list()

    # Stacked components with TradingView palette
    if "thermal_hydro_mw" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["thermal_hydro_mw"].to_list(),
                mode="lines",
                name="Thermal & Hydro",
                stackgroup="generation",
                line={"width": 0.5, "color": THEME["thermal_color"]},
                fillcolor="rgba(71, 85, 105, 0.65)",
                hovertemplate="Thermal/Hydro: %{y:,.0f} MW<extra></extra>",
            )
        )

    if "wind_mw" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["wind_mw"].to_list(),
                mode="lines",
                name="Wind",
                stackgroup="generation",
                line={"width": 0.5, "color": THEME["wind_color"]},
                fillcolor="rgba(16, 185, 129, 0.65)",
                hovertemplate="Wind: %{y:,.0f} MW<extra></extra>",
            )
        )

    if "pv_mw" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["pv_mw"].to_list(),
                mode="lines",
                name="Solar (PV)",
                stackgroup="generation",
                line={"width": 0.5, "color": THEME["pv_color"]},
                fillcolor="rgba(245, 158, 11, 0.65)",
                hovertemplate="Solar PV: %{y:,.0f} MW<extra></extra>",
            )
        )

    # Demand reference overlay line (Neon Coral)
    if "demand_mw" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["demand_mw"].to_list(),
                mode="lines",
                name="Total Demand (KSE)",
                line={"color": THEME["demand_color"], "width": 3, "dash": "dot"},
                hovertemplate="Total Demand: %{y:,.0f} MW<extra></extra>",
            )
        )

    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 18, "color": THEME["text_color"]}},
        xaxis={
            "title": "Time (Europe/Warsaw)",
            "gridcolor": THEME["grid_color"],
            "showgrid": True,
            "zeroline": False,
        },
        yaxis={
            "title": "Power (MW)",
            "gridcolor": THEME["grid_color"],
            "showgrid": True,
            "zeroline": False,
        },
        template="plotly_dark",
        paper_bgcolor=THEME["paper_bg"],
        plot_bgcolor=THEME["plot_bg"],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        hovermode="x unified",
    )
    return fig


def plot_curtailment_gauge(
    curtailment_ratio: float,
    alert_level: str = "NORMAL",
) -> go.Figure:
    """Gauge meter displaying current/forecast renewable penetration and curtailment risk."""
    pct = min(100.0, max(0.0, curtailment_ratio * 100.0))

    if alert_level == "CRITICAL":
        bar_color = THEME["demand_color"]
    elif alert_level == "WATCH":
        bar_color = THEME["pv_color"]
    else:
        bar_color = THEME["wind_color"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 42, "color": THEME["text_color"]}},
            title={
                "text": f"<b>Renewable Penetration (Risk: {alert_level})</b>",
                "font": {"size": 16, "color": THEME["text_color"]},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": THEME["text_muted"]},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": THEME["plot_bg"],
                "borderwidth": 1,
                "bordercolor": THEME["grid_color"],
                "steps": [
                    {"range": [0, 75], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [75, 90], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [90, 100], "color": "rgba(255, 59, 105, 0.35)"},
                ],
                "threshold": {
                    "line": {"color": THEME["demand_color"], "width": 4},
                    "thickness": 0.8,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=THEME["paper_bg"],
        font={"color": THEME["text_color"]},
        margin={"l": 30, "r": 30, "t": 40, "b": 30},
        height=260,
    )
    return fig


def plot_wape_comparison(leaderboard_df: pl.DataFrame) -> go.Figure:
    """Bar chart comparing WAPE (%) across benchmarked models in TradingView palette."""
    fig = go.Figure()

    models = leaderboard_df["model"].to_list()
    wapes = leaderboard_df["WAPE"].to_list()

    colors = [THEME["forecast_color"] if "TimesFM" in m else THEME["actual_color"] for m in models]

    fig.add_trace(
        go.Bar(
            x=models,
            y=wapes,
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{w:.2f}%" for w in wapes],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>WAPE: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "<b>Model Accuracy Comparison: WAPE (%)</b>",
            "font": {"size": 16, "color": THEME["text_color"]},
        },
        xaxis={"title": "Model", "gridcolor": THEME["grid_color"]},
        yaxis={"title": "WAPE (%) - Lower is Better", "gridcolor": THEME["grid_color"]},
        template="plotly_dark",
        paper_bgcolor=THEME["paper_bg"],
        plot_bgcolor=THEME["plot_bg"],
        margin={"l": 40, "r": 20, "t": 50, "b": 50},
    )
    return fig
