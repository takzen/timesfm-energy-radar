# ⚡ PSE TimesFM Radar: Zero-Shot Power Grid & Renewable Forecasting

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TimesFM](https://img.shields.io/badge/Model-Google_TimesFM-orange.svg)](https://github.com/google-research/timesfm)
[![Data](https://img.shields.io/badge/Data-PSE_Open_Data-green.svg)](https://www.pse.pl/dane-systemowe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A high-performance multivariate time-series forecasting system benchmarking **Google's TimesFM** (Time Series Foundation Model) zero-shot capabilities against classical baselines (LightGBM, Seasonal Naive) on the **Polish National Power System (PSE - Polskie Sieci Elektroenergetyczne)** and **Open-Meteo** weather reanalysis data.

---

## 🎯 Project Overview & Highlights

- **🔌 Grid Load & Balancing Price Forecasting:** Zero-shot multi-horizon forecasting (24h, 48h, 168h) of national electric load (KSE) and Balancing Market Prices (`RCE` / `CRO`).
- **☀️ Renewable Generation & Curtailment Risk:** Jointly forecasts solar (PV) and wind power generation, alerting to imminent overgeneration events and grid curtailment risks.
- **🌤️ Exogenous Weather Conditioning:** Seamlessly incorporates meteorological covariates (GHI, direct solar radiation, 100m wind speed, 2m temperature, cloud cover) via Open-Meteo APIs.
- **📊 Strict Benchmarking Suite:** Quantifies rolling-window accuracy using MAE, RMSE, WAPE, and quantile pinball loss.
- **🖥️ Interactive Mission Control Dashboard:** Production-ready Streamlit UI providing real-time forecasts, uncertainty intervals, generation mix decomposition, and benchmark leaderboards.

---

## 🏗️ Architecture & Data Pipeline

```mermaid
flowchart TD
    subgraph Ingestion["🔌 Ingestion Layer"]
        PSE["PSE API v2<br/><i>Load, PV, Wind, Thermal, RCE Prices</i>"]
        METEO["Open-Meteo API<br/><i>Solar Irradiance, Wind 100m, Temperature</i>"]
        RAW[("data/raw/<br/>Raw Parquet Storage")]
        PSE --> RAW
        METEO --> RAW
    end

    subgraph Processing["🧹 ETL & Feature Engineering"]
        ETL["Temporal Alignment & Cleaning<br/><i>Europe/Warsaw TZ, DST handling, Missing Imputation</i>"]
        FEATS["Exogenous Feature Store<br/><i>Calendar, Lags, Weather Covariates</i>"]
        PROC[("data/processed/<br/>Engineered Datasets")]
        RAW --> ETL --> FEATS --> PROC
    end

    subgraph Models["🧠 Multi-Horizon Forecasting Engine"]
        TFM["Google TimesFM<br/><i>Zero-Shot Foundation Model</i>"]
        BASE["Baselines<br/><i>LightGBM, Seasonal Naive</i>"]
        PROC --> TFM
        PROC --> BASE
    end

    subgraph Evaluation["📊 Evaluation & Risk Suite"]
        METRICS["Benchmark Engine<br/><i>MAE, RMSE, WAPE, Quantile Loss</i>"]
        RISK["Renewable Curtailment Risk<br/><i>PV/Wind Oversupply Anomaly Detection</i>"]
        TFM --> METRICS
        BASE --> METRICS
        TFM --> RISK
    end

    subgraph Presentation["🖥️ Mission Control UI"]
        DASH["Streamlit & Plotly Dashboard<br/><i>Forecast Charts, Intervals, Curtailment Alerts, Leaderboard</i>"]
        METRICS --> DASH
        RISK --> DASH
    end

    style PSE fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style METEO fill:#e0f2f1,stroke:#00897b,stroke-width:2px
    style RAW fill:#f5f5f5,stroke:#757575,stroke-width:1px
    style TFM fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style RISK fill:#ffebee,stroke:#e53935,stroke-width:2px
    style DASH fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
```

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                 # Raw ingested Parquet files (gitignored)
│   ├── processed/           # Cleaned, time-aligned datasets (gitignored)
│   └── models/              # Cached weights / checkpoints (gitignored)
├── src/
│   ├── ingestion/           # PSE & Open-Meteo API connectors, CLI fetcher
│   │   ├── pse.py           # PSE API client with exponential backoff
│   │   ├── weather.py       # Open-Meteo client for archive & forecast
│   │   └── fetch.py         # Orchestrator & CLI ingestion runner
│   ├── processing/          # Cleaning, resampling, feature engineering
│   ├── models/              # TimesFM wrapper, baselines, evaluation
│   ├── visualization/       # Streamlit UI & Plotly visualization modules
│   ├── config.py            # Pydantic v2 application configuration
│   └── app.py               # Streamlit application entrypoint
├── tests/                   # Pytest suite with mocked network tests
├── scripts/                 # CLI tools, benchmark scripts, runners
├── pyproject.toml           # PEP 621 dependencies & tooling configs
├── Makefile                 # Development tasks (lint, format, test, run)
├── LICENSE                  # MIT License (Krzysztof Pika)
└── README.md                # Project documentation
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.11 or 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 1. Clone the repository
```bash
git clone https://github.com/takzen/timesfm-energy-radar.git
cd timesfm-energy-radar
```

### 2. Set up virtual environment and install dependencies
```bash
# Using uv (fastest)
uv venv --python 3.11
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 3. Configure environment
```bash
cp .env.example .env
```
Default configuration values in `.env.example` will work out of the box without requiring paid API keys.

---

## 💻 CLI Usage

### Ingest National Grid & Weather Data
Fetch historical and forecast PSE energy metrics along with Open-Meteo weather features:
```bash
# Ingest past 30 days of PSE and weather data
python -m src.ingestion.fetch --days 30

# Ingest specific date range
python -m src.ingestion.fetch --start-date 2026-01-01 --end-date 2026-02-01
```

### Run Quality Verification
```bash
make lint        # Run ruff checks
make format      # Format code with ruff
make typecheck   # Static typecheck with mypy
make test        # Run pytest test suite
```

### Launch Interactive Dashboard
```bash
make run
# or
streamlit run src/app.py
```

---

## 📜 License

Distributed under the **MIT License**. Copyright (c) 2026 Krzysztof Pika. See [LICENSE](LICENSE) for full details.
