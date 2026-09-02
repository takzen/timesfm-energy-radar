"""Plotly interactive visualization components for forecasts, generation mix,
and curtailment risk.
"""

from collections.abc import Sequence
from datetime import datetime

import plotly.graph_objects as go
import polars as pl


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

    # 1. Historical / Actuals line
    if actual_timestamps is not None and actual_values is not None and len(actual_values) > 0:
        fig.add_trace(
            go.Scatter(
                x=list(actual_timestamps),
                y=list(actual_values),
                mode="lines+markers",
                name="Actual",
                line={"color": "#38bdf8", "width": 2.5},
                marker={"size": 4},
                hovertemplate="<b>Actual</b>: %{y:.1f} "
                + unit
                + "<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            )
        )

    # 2. Uncertainty intervals (filled area between Q10 and Q90)
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
                fillcolor="rgba(245, 158, 11, 0.20)",
                name="10%-90% Prediction Band",
                hoverinfo="skip",
            )
        )

    # 3. Point forecast line
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(point_forecast),
            mode="lines+markers",
            name=f"{model_name} Forecast",
            line={"color": "#f59e0b", "width": 3, "dash": "solid"},
            marker={"size": 5},
            hovertemplate="<b>"
            + model_name
            + "</b>: %{y:.1f} "
            + unit
            + "<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 18, "color": "#f8fafc"}},
        xaxis={"title": "Time (Europe/Warsaw)", "gridcolor": "#334155", "showgrid": True},
        yaxis={"title": f"Value ({unit})", "gridcolor": "#334155", "showgrid": True},
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#f8fafc"},
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

    # Stacked components
    if "thermal_hydro_mw" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["thermal_hydro_mw"].to_list(),
                mode="lines",
                name="Thermal & Hydro",
                stackgroup="generation",
                line={"width": 0.5, "color": "#94a3b8"},
                fillcolor="rgba(148, 163, 184, 0.6)",
                hovertemplate="Thermal/Hydro: %{y:.0f} MW<extra></extra>",
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
                line={"width": 0.5, "color": "#34d399"},
                fillcolor="rgba(52, 211, 153, 0.7)",
                hovertemplate="Wind: %{y:.0f} MW<extra></extra>",
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
                line={"width": 0.5, "color": "#fbbf24"},
                fillcolor="rgba(251, 191, 36, 0.7)",
                hovertemplate="Solar PV: %{y:.0f} MW<extra></extra>",
            )
        )

    # Demand reference overlay line
    if "demand_mw" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["demand_mw"].to_list(),
                mode="lines",
                name="Total Demand (KSE)",
                line={"color": "#f43f5e", "width": 3, "dash": "dot"},
                hovertemplate="Total Demand: %{y:.0f} MW<extra></extra>",
            )
        )

    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 18, "color": "#f8fafc"}},
        xaxis={"title": "Time (Europe/Warsaw)", "gridcolor": "#334155", "showgrid": True},
        yaxis={"title": "Power (MW)", "gridcolor": "#334155", "showgrid": True},
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
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
        bar_color = "#f43f5e"
    elif alert_level == "WATCH":
        bar_color = "#fbbf24"
    else:
        bar_color = "#34d399"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 42, "color": "#f8fafc"}},
            title={
                "text": f"<b>Renewable Penetration (Risk: {alert_level})</b>",
                "font": {"size": 16, "color": "#f8fafc"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#1e293b",
                "borderwidth": 2,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 75], "color": "rgba(52, 211, 153, 0.15)"},
                    {"range": [75, 90], "color": "rgba(251, 191, 36, 0.25)"},
                    {"range": [90, 100], "color": "rgba(244, 63, 94, 0.35)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.8,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#0f172a",
        font={"color": "#f8fafc"},
        margin={"l": 30, "r": 30, "t": 40, "b": 30},
        height=260,
    )
    return fig


def plot_wape_comparison(leaderboard_df: pl.DataFrame) -> go.Figure:
    """Bar chart comparing WAPE (%) across benchmarked models."""
    fig = go.Figure()

    models = leaderboard_df["model"].to_list()
    wapes = leaderboard_df["WAPE"].to_list()

    colors = ["#f59e0b" if "TimesFM" in m else "#38bdf8" for m in models]

    fig.add_trace(
        go.Bar(
            x=models,
            y=wapes,
            marker={"color": colors},
            text=[f"{w:.2f}%" for w in wapes],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>WAPE: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "<b>Model Accuracy Comparison: WAPE (%)</b>",
            "font": {"size": 16, "color": "#f8fafc"},
        },
        xaxis={"title": "Model", "gridcolor": "#334155"},
        yaxis={"title": "WAPE (%) - Lower is Better", "gridcolor": "#334155"},
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        margin={"l": 40, "r": 20, "t": 50, "b": 50},
    )
    return fig
