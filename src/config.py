"""Application configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration class for TimesFM Energy Radar."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General Application Settings
    app_name: str = "TimesFM Energy Radar"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production", "testing"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Base Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    raw_data_dir: Path = Field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = Field(default_factory=lambda: Path("data/processed"))
    models_cache_dir: Path = Field(default_factory=lambda: Path("data/models"))

    # PSE (Polskie Sieci Elektroenergetyczne) API
    pse_api_base_url: str = "https://api.raporty.pse.pl/api"
    pse_timeout_seconds: int = 30
    pse_max_retries: int = 3

    # Open-Meteo Weather API
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    poland_centroid_lat: float = 52.0693
    poland_centroid_lon: float = 19.4803

    # TimesFM Engine Configuration
    timesfm_model_id: str = "google/timesfm-3.0-pytorch"
    timesfm_device: str = "cpu"
    default_forecast_horizon: int = 24
    default_context_length: int = 512

    # Timezone
    timezone: str = "Europe/Warsaw"

    # Streamlit UI Configuration
    streamlit_server_port: int = 8501
    streamlit_server_address: str = "0.0.0.0"

    def ensure_directories(self) -> None:
        """Ensure that all necessary data directories exist on the filesystem."""
        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_cache_dir,
        ):
            full_path = self.project_root / path if not path.is_absolute() else path
            full_path.mkdir(parents=True, exist_ok=True)


# Global settings instance singleton
settings = Settings()
