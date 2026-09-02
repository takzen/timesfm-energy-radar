"""Tests for configuration settings."""

from pathlib import Path

from src.config import Settings


def test_default_settings() -> None:
    """Test default values of Settings."""
    cfg = Settings()
    assert cfg.app_name == "TimesFM Energy Radar"
    assert cfg.environment in ["development", "staging", "production", "testing"]
    assert cfg.default_forecast_horizon == 24
    assert cfg.timezone == "Europe/Warsaw"
    assert cfg.pse_api_base_url.startswith("https://")
    assert cfg.poland_centroid_lat == 52.0693


def test_ensure_directories(tmp_path: Path) -> None:
    """Test directory creation logic."""
    cfg = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "custom_data",
        raw_data_dir=tmp_path / "custom_data" / "raw",
        processed_data_dir=tmp_path / "custom_data" / "processed",
        models_cache_dir=tmp_path / "custom_data" / "models",
    )
    cfg.ensure_directories()
    assert (tmp_path / "custom_data" / "raw").is_dir()
    assert (tmp_path / "custom_data" / "processed").is_dir()
    assert (tmp_path / "custom_data" / "models").is_dir()
