"""Streamlit Mission Control application entrypoint for PSE TimesFM Energy Radar."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import streamlit as st
from loguru import logger

from src.config import settings
from src.models.base import Forecaster
from src.models.baselines import (
    GBDTForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from src.models.evaluation import RollingBacktest
from src.models.timesfm_model import TimesFMModel
from src.processing.features import build_feature_pipeline
from src.visualization.benchmark_view import render_benchmark_view
from src.visualization.charts import (
    plot_curtailment_gauge,
    plot_forecast_with_intervals,
    plot_generation_mix,
)

# Set page configuration
st.set_page_config(
    page_title="⚡ PSE TimesFM Energy Radar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Generating realistic KSE grid dataset...")
def get_or_create_sample_dataset() -> pl.DataFrame:
    """Load processed dataset from disk or synthesize representative Polish grid data."""
    processed_path = settings.processed_data_dir / "kse_hourly_features.parquet"
    if processed_path.exists():
        try:
            return pl.read_parquet(processed_path)
        except Exception as e:
            logger.warning("Failed to load existing parquet ({}): generating sample.", e)

    # Synthesize realistic Polish power grid hourly data for past 60 days
    n_hours = 24 * 60
    base_ts = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=n_hours)
    timestamps = [base_ts + timedelta(hours=i) for i in range(n_hours)]

    rng = np.random.default_rng(42)
    # Seasonal diurnal demand pattern (approx 14,000 - 23,000 MW in Poland)
    hours = np.array([ts.hour for ts in timestamps])
    weekdays = np.array([ts.weekday() for ts in timestamps])
    is_wknd = weekdays >= 5

    diurnal_curve = np.sin((hours - 6) / 24.0 * 2 * np.pi) * 3500.0 + 17500.0
    weekend_discount = np.where(is_wknd, -2500.0, 0.0)
    noise = rng.normal(0, 400.0, n_hours)
    demand = np.clip(diurnal_curve + weekend_discount + noise, 12000.0, 26000.0)

    # PV generation (bell curve between 06:00 and 20:00, up to 9,000 MW peak)
    solar_intensity = np.maximum(0.0, np.sin((hours - 6) / 14.0 * np.pi))
    solar_mask = (hours >= 6) & (hours <= 20)
    pv = np.where(solar_mask, solar_intensity * (6500.0 + rng.uniform(-1000, 2000, n_hours)), 0.0)
    pv = np.clip(pv, 0.0, 11000.0)

    # Wind generation (cyclical weather fronts, 1,000 - 8,500 MW)
    wind_waves = np.sin(np.arange(n_hours) / 36.0) * 2500.0 + 3500.0 + rng.normal(0, 500, n_hours)
    wind = np.clip(wind_waves, 200.0, 8500.0)

    # Thermal & Hydro dispatch to meet remainder
    thermal = np.maximum(4000.0, demand - (pv + wind) + rng.normal(0, 200, n_hours))

    # Balancing Market Prices (RCE in PLN/MWh, typical 200 - 800 PLN, dips during solar peak)
    rce = 450.0 + (demand / 30.0) - (pv / 15.0) + rng.normal(0, 40, n_hours)
    rce = np.clip(rce, -50.0, 1200.0)

    # Weather covariates
    temp = 18.0 + 7.0 * np.sin((hours - 9) / 24.0 * 2 * np.pi) + rng.normal(0, 1.5, n_hours)
    wind_spd = np.clip(wind / 700.0 + rng.normal(0, 1.0, n_hours), 1.0, 25.0)

    raw_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": demand,
            "pv_mw": pv,
            "wind_mw": wind,
            "thermal_hydro_mw": thermal,
            "rce_pln_mwh": rce,
            "temperature_2m": temp,
            "wind_speed_100m": wind_spd,
        }
    )

    featured_df = build_feature_pipeline(raw_df, include_lags=True)
    return featured_df


def main() -> None:
    """Main Streamlit execution logic."""
    st.sidebar.title("⚡ TimesFM Energy Radar")
    st.sidebar.markdown(
        "**Zero-Shot Power Grid & Renewable Curtailment Forecasting** on Polish PSE data."
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "🔮 Live Forecasting & Radar",
            "🏆 Benchmark Leaderboard",
            "⚙️ Data Ingestion & System Info",
        ],
    )

    df = get_or_create_sample_dataset()

    # Sidebar parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Forecast Parameters")
    horizon = st.sidebar.selectbox("Forecast Horizon", [24, 48, 168], index=0)
    target_metric = st.sidebar.selectbox(
        "Target Metric",
        ["demand_mw", "rce_pln_mwh"],
        format_func=lambda x: (
            "🔌 KSE Power Demand (MW)" if x == "demand_mw" else "💰 Balancing Price RCE (PLN/MWh)"
        ),
    )

    model_options = [
        "Google TimesFM (Zero-Shot)",
        "LightGBM / GBDT Autoregressive",
        "Seasonal Naive (24h)",
        "Persistence",
    ]
    selected_model_name = st.sidebar.selectbox("Forecasting Engine", model_options, index=0)

    unit = "MW" if target_metric == "demand_mw" else "PLN/MWh"

    if page == "🔮 Live Forecasting & Radar":
        st.title("⚡ Polish Power Grid (KSE) Energy Radar")
        st.markdown(
            "Multi-horizon zero-shot forecasting powered by **Google TimesFM** "
            "with meteorological and calendar exogenous conditioning."
        )

        # Top KPI Metrics Cards
        latest = df.tail(1)
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        cur_demand = float(latest["demand_mw"][0])
        cur_pv = float(latest["pv_mw"][0])
        cur_wind = float(latest["wind_mw"][0])
        cur_rce = float(latest["rce_pln_mwh"][0])
        cur_re_ratio = (cur_pv + cur_wind) / max(1.0, cur_demand)

        kpi1.metric(
            "Current Demand", f"{cur_demand:,.0f} MW", delta=f"{cur_demand - 18000:,.0f} MW vs Avg"
        )
        kpi2.metric(
            "Solar (PV) Gen",
            f"{cur_pv:,.0f} MW",
            delta=f"{cur_pv / max(1.0, cur_demand) * 100:.1f}% Share",
        )
        kpi3.metric(
            "Wind Gen",
            f"{cur_wind:,.0f} MW",
            delta=f"{cur_wind / max(1.0, cur_demand) * 100:.1f}% Share",
        )
        kpi4.metric("Balancing Price", f"{cur_rce:.1f} PLN", delta=f"{cur_rce - 450.0:.1f} PLN")

        risk_level = (
            "CRITICAL" if cur_re_ratio >= 0.90 else ("WATCH" if cur_re_ratio >= 0.75 else "NORMAL")
        )
        kpi5.metric("Curtailment Risk", risk_level, delta=f"{cur_re_ratio * 100:.1f}% OZE")

        st.markdown("---")

        # 1. Forecasting Execution
        with st.spinner(f"Generating {horizon}h forecast with {selected_model_name}..."):
            split_idx = df.height - horizon
            context_df = df.slice(0, split_idx)
            actual_test_df = df.slice(split_idx, horizon)

            forecaster: Forecaster
            if selected_model_name == "Google TimesFM (Zero-Shot)":
                forecaster = TimesFMModel()
            elif selected_model_name == "LightGBM / GBDT Autoregressive":
                forecaster = GBDTForecaster()
            elif selected_model_name == "Seasonal Naive (24h)":
                forecaster = SeasonalNaiveForecaster(season_length=24)
            else:
                forecaster = PersistenceForecaster()

            forecaster.fit(context_df, target_col=target_metric)
            forecast_out = forecaster.forecast(
                context_df=context_df,
                horizon=horizon,
                target_col=target_metric,
                future_covariates_df=actual_test_df,
            )

        # Plot Forecast Chart
        col_main, col_gauge = st.columns([3, 1])

        with col_main:
            fig_fc = plot_forecast_with_intervals(
                timestamps=forecast_out.timestamps,
                point_forecast=forecast_out.point_forecast,
                q10=forecast_out.quantiles.get(0.1),
                q90=forecast_out.quantiles.get(0.9),
                actual_timestamps=actual_test_df["timestamp"].to_list(),
                actual_values=actual_test_df[target_metric].to_list(),
                target_name=target_metric,
                model_name=selected_model_name,
                title=f"{selected_model_name}: {horizon}h Forecast vs Actuals",
                unit=unit,
            )
            st.plotly_chart(fig_fc, width="stretch")

        with col_gauge:
            st.markdown("#### ⚠️ Curtailment Meter")
            fig_gauge = plot_curtailment_gauge(cur_re_ratio, alert_level=risk_level)
            st.plotly_chart(fig_gauge, width="stretch")

            if risk_level == "CRITICAL":
                st.error(
                    "🚨 **CRITICAL OVERGENERATION RISK**: Renewables exceed 90% of grid demand! "
                    "Potential grid curtailment required."
                )
            elif risk_level == "WATCH":
                st.warning(
                    "⚠️ **WATCH ALERT**: High renewable penetration (>75%). "
                    "Dynamic balancing required."
                )
            else:
                st.success("✅ **STABLE GRID**: Grid operates within nominal operating reserves.")

        # 2. Generation Mix Breakdown
        st.markdown("### 🔋 KSE National Generation Breakdown")
        recent_window = df.tail(168)  # Last 7 days
        fig_mix = plot_generation_mix(recent_window)
        st.plotly_chart(fig_mix, width="stretch")

    elif page == "🏆 Benchmark Leaderboard":
        st.title("🏆 Time-Series Foundation Models Benchmark")
        st.markdown(
            "Systematic rolling-window backtesting comparing **Google TimesFM** "
            "against classical machine learning and baseline heuristics."
        )

        with st.spinner("Running rolling-window backtesting across all candidate forecasters..."):
            forecaster_pool = [
                TimesFMModel(),
                GBDTForecaster(max_iter=50),
                SeasonalNaiveForecaster(season_length=24),
                PersistenceForecaster(),
            ]
            backtest_engine = RollingBacktest(horizon=24, stride=48, min_train_hours=24 * 14)
            preds_df, lb_df = backtest_engine.run(
                df=df,
                forecasters=forecaster_pool,
                target_col=target_metric,
            )

        render_benchmark_view(leaderboard_df=lb_df, predictions_df=preds_df)

    else:
        st.title("⚙️ System Architecture & Data Status")
        st.markdown("### 📡 PSE & Weather Connector Health")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### PSE Open Data API (v2)")
            st.markdown(f"- **Endpoint**: `{settings.pse_api_base_url}`")
            st.markdown(f"- **Timeout**: `{settings.pse_timeout_seconds}s`")
            st.markdown(f"- **Max Retries**: `{settings.pse_max_retries}`")
            st.markdown(
                "- **Reports ingested**: `/his-wlk-cal` (Load & Gen), `/rce-pln` (Balancing Prices)"
            )

        with c2:
            st.markdown("#### Open-Meteo Weather Service")
            st.markdown(f"- **Archive URL**: `{settings.open_meteo_archive_url}`")
            st.markdown(
                f"- **Coordinates**: Lat `{settings.poland_centroid_lat}`, "
                f"Lon `{settings.poland_centroid_lon}`"
            )
            st.markdown(f"- **Timezone**: `{settings.timezone}`")
            st.markdown("- **Features**: GHI, DNI, Wind 100m, Temperature 2m, Cloud Cover")

        st.markdown("---")
        st.markdown("### 🗄️ Dataset Schema Inspection")
        st.dataframe(df.head(10).to_pandas(), width="stretch")


if __name__ == "__main__":
    main()
