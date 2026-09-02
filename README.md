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
        PSE["<b>PSE API v2</b><br/>Load, PV, Wind, Thermal, RCE Prices"]
        METEO["<b>Open-Meteo API</b><br/>Solar Irradiance, Wind 100m, Temperature"]
        RAW[("<b>data/raw/</b><br/>Raw Parquet Storage")]
        PSE --> RAW
        METEO --> RAW
    end

    subgraph Processing["🧹 ETL & Feature Engineering"]
        ETL["<b>Temporal Alignment & Cleaning</b><br/>Europe/Warsaw TZ, DST handling, Imputation"]
        FEATS["<b>Exogenous Feature Store</b><br/>Calendar, Lags, Weather Covariates"]
        PROC[("<b>data/processed/</b><br/>Engineered Datasets")]
        RAW --> ETL --> FEATS --> PROC
    end

    subgraph Models["🧠 Multi-Horizon Forecasting Engine"]
        TFM["<b>Google TimesFM</b><br/>Zero-Shot Foundation Model"]
        BASE["<b>Baselines</b><br/>LightGBM, Seasonal Naive"]
        PROC --> TFM
        PROC --> BASE
    end

    subgraph Evaluation["📊 Evaluation & Risk Suite"]
        METRICS["<b>Benchmark Engine</b><br/>MAE, RMSE, WAPE, Quantile Loss"]
        RISK["<b>Renewable Curtailment Risk</b><br/>PV/Wind Oversupply Anomaly Detection"]
        TFM --> METRICS
        BASE --> METRICS
        TFM --> RISK
    end

    subgraph Presentation["🖥️ Mission Control UI"]
        DASH["<b>Streamlit & Plotly Dashboard</b><br/>Forecast Charts, Intervals, Curtailment Alerts, Leaderboard"]
        METRICS --> DASH
        RISK --> DASH
    end

    classDef blueNode fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef greenNode fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef amberNode fill:#451a03,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;
    classDef redNode fill:#4c0519,stroke:#f43f5e,stroke-width:2px,color:#f8fafc;
    classDef purpleNode fill:#3b0764,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef darkNode fill:#18181b,stroke:#a1a1aa,stroke-width:2px,color:#f8fafc;

    class PSE,METEO blueNode;
    class RAW,PROC darkNode;
    class ETL,FEATS greenNode;
    class TFM amberNode;
    class BASE darkNode;
    class METRICS,RISK redNode;
    class DASH purpleNode;
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

### Run Systematic Model Benchmark
Evaluate Google TimesFM zero-shot performance against GBDT and statistical baselines across rolling-window backtesting:
```bash
# Run 24h rolling-window benchmark
make benchmark
# or with custom parameters:
uv run python scripts/run_benchmark.py --target-col demand_mw --horizon 24 --stride 24
```

#### Benchmark Results (24h Forecast Horizon on Polish Power Grid)

| Rank | Model | WAPE (%) | MAE (MW) | RMSE (MW) | MAPE (%) |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 | **Google TimesFM 3.0 (Zero-Shot)** | **2.07%** | **348.9** | **435.7** | **2.16%** |
| 🥈 | LightGBM / GBDT Autoregressive | 2.44% | 413.3 | 518.8 | 2.51% |
| 🥉 | Seasonal Naive (24h Diurnal) | 4.72% | 795.1 | 984.8 | 4.85% |
| 4 | Persistence (Last Value) | 14.48% | 2,453.1 | 2,855.0 | 14.03% |

### Prepare / Download Processed Dataset
```bash
# Download and process live PSE + Open-Meteo data
uv run python scripts/download_sample_data.py --days 30

# Or generate representative synthetic Polish grid dataset instantly
make sample-data
```

### Launch Interactive Mission Control Dashboard
```bash
make run
# or directly via uv:
uv run streamlit run src/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** (or configured port) in your browser:
- **🔮 Live Forecasting**: Zero-shot multi-horizon inference (24h, 48h, 168h), TradingView Dark theme, $q_{10}-q_{90}$ uncertainty bands, and interactive generation mix breakdown.
- **⚠️ Curtailment Radar**: Real-time renewable penetration gauge and grid overgeneration alerts (`NORMAL`, `WATCH`, `CRITICAL`).
- **🏆 Benchmark Leaderboard**: In-depth comparison metrics and diurnal error attribution across peak hours.

### Run Quality Verification
```bash
make lint        # Run ruff checks
make format      # Format code with ruff
make typecheck   # Static typecheck with mypy
make test        # Run full pytest test suite (35 tests)
```

---

## 📜 License

Distributed under the **MIT License**. Copyright (c) 2026 Krzysztof Pika. See [LICENSE](LICENSE) for full details.
