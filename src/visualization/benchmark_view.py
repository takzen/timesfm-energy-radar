"""Benchmark leaderboard view rendering comparison tables and error attribution plots."""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.visualization.charts import plot_wape_comparison


def render_benchmark_view(
    leaderboard_df: pl.DataFrame,
    predictions_df: pl.DataFrame | None = None,
) -> None:
    """Render the model benchmark leaderboard and diagnostic error attribution in Streamlit."""
    st.markdown("## 🏆 Forecasting Models Benchmark Leaderboard")
    st.markdown(
        "Quantitative error comparison across rolling-window backtesting evaluations "
        "on Polish Grid data."
    )

    if leaderboard_df.is_empty():
        st.warning("No backtest results available to display in leaderboard.")
        return

    # 1. Summary Leaderboard Table
    st.markdown("### 📋 Overall Metrics Summary")
    formatted_df = leaderboard_df.with_columns(
        [
            pl.col("MAE").round(2),
            pl.col("RMSE").round(2),
            pl.col("WAPE").round(2),
            pl.col("MAPE").round(2),
        ]
    ).to_pandas()

    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True,
    )

    # 2. Visual WAPE Comparison Bar Chart
    st.markdown("### 📊 WAPE (%) Accuracy Comparison")
    fig_wape = plot_wape_comparison(leaderboard_df)
    st.plotly_chart(fig_wape, use_container_width=True)

    # 3. Peak-hour / Diurnal Error Attribution
    if (
        predictions_df is not None
        and not predictions_df.is_empty()
        and "timestamp" in predictions_df.columns
    ):
        st.markdown("### ⏰ Diurnal Error Attribution (Error by Hour of Day)")
        st.markdown(
            "Analyzing error patterns across peak demand hours (08:00–12:00, 18:00–21:00) "
            "and solar peak hours (11:00–14:00)."
        )

        # Compute hourly absolute error
        annotated = predictions_df.with_columns(
            [
                pl.col("timestamp").dt.hour().alias("hour"),
                (pl.col("actual") - pl.col("prediction")).abs().alias("abs_error"),
            ]
        )

        hourly_error = (
            annotated.group_by(["hour", "model"])
            .agg(pl.col("abs_error").mean().alias("mean_abs_error"))
            .sort(["hour", "model"])
        )

        fig_hourly = go.Figure()
        for model_name in leaderboard_df["model"].to_list():
            sub = hourly_error.filter(pl.col("model") == model_name)
            if not sub.is_empty():
                fig_hourly.add_trace(
                    go.Scatter(
                        x=sub["hour"].to_list(),
                        y=sub["mean_abs_error"].to_list(),
                        mode="lines+markers",
                        name=model_name,
                        line={"width": 2.5},
                    )
                )

        fig_hourly.update_layout(
            title={
                "text": "<b>Mean Absolute Error (MAE) by Hour of Day (0–23h)</b>",
                "font": {"color": "#f8fafc"},
            },
            xaxis={"title": "Hour of Day (CET/CEST)", "dtick": 2, "gridcolor": "#334155"},
            yaxis={"title": "MAE (MW / PLN)", "gridcolor": "#334155"},
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            legend={"orientation": "h", "y": 1.1, "x": 1, "xanchor": "right"},
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
