# ⚡ PSE TimesFM Radar: Zero-Shot Power Grid & Renewable Forecasting

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TimesFM](https://img.shields.io/badge/Model-Google_TimesFM-orange.svg)](https://github.com/google-research/timesfm)
[![Data](https://img.shields.io/badge/Data-PSE_Open_Data-green.svg)](https://www.pse.pl/dane-systemowe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A multivariate time-series forecasting pipeline benchmarking **Google's TimesFM** zero-shot foundation model against classical baselines (LightGBM, Prophet, ARIMA) on the Polish National Power System (PSE) data.

## 🎯 What it does

- **🔌 Grid Load & Price Forecasting:** Zero-shot multi-horizon forecasting of power demand and balancing market prices using exogenous covariates (temperature, wind speed, calendar features).
- **☀️ Renewable Curtailment Risk:** Predicts sudden surges in PV/Wind production that risk power grid oversupply.
- **📈 Zero-Shot vs. Classical Benchmark:** Comprehensive error metrics (MAE, RMSE, WAPE) comparing Google TimesFM against tuned LightGBM & Chronos models.
- **🖥️ Interactive Dashboard:** Streamlit app visualizing live predictions vs. actuals with historical anomaly detection.

## 🏗️ Architecture

1. **Ingestion:** Automated ETL pulling hourly grid metrics from the official PSE API + Open-Meteo weather APIs.
2. **Forecasting Engine:** Google TimesFM (`pip install timesfm`) configured with dynamic covariate conditioning.
3. **Evaluation Suite:** Rolling-window backtesting and error attribution across seasons/peak hours.
4. **UI:** Streamlit & Plotly interactive dashboard.
