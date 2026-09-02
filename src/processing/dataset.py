"""Dataset persistence, DuckDB analytical querying, and chronological temporal splitting."""

from pathlib import Path

import duckdb
import polars as pl
from loguru import logger

from src.config import settings


def save_processed_dataset(
    df: pl.DataFrame,
    destination_path: Path | None = None,
) -> Path:
    """Save processed feature dataset into Parquet format.

    Parameters
    ----------
    df : pl.DataFrame
        Processed dataset DataFrame
    destination_path : Path | None
        Target file path (defaults to data/processed/kse_hourly_features.parquet)

    Returns
    -------
    Path
        Absolute or relative path to saved Parquet file
    """
    out_path = destination_path or (settings.processed_data_dir / "kse_hourly_features.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.write_parquet(out_path)
    logger.info(
        "Saved processed dataset ({} rows, {} cols) to {}", df.height, len(df.columns), out_path
    )
    return out_path


def load_processed_dataset(source_path: Path | None = None) -> pl.DataFrame:
    """Load processed feature dataset from Parquet.

    Parameters
    ----------
    source_path : Path | None
        Source Parquet file path (defaults to data/processed/kse_hourly_features.parquet)

    Returns
    -------
    pl.DataFrame
        Loaded Polars DataFrame
    """
    path = source_path or (settings.processed_data_dir / "kse_hourly_features.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {path}")

    df = pl.read_parquet(path)
    logger.info("Loaded processed dataset ({} rows) from {}", df.height, path)
    return df


def query_duckdb(
    sql_query: str,
    parquet_path: Path | None = None,
) -> pl.DataFrame:
    """Execute high-speed analytical SQL queries via DuckDB over the dataset.

    Parameters
    ----------
    sql_query : str
        SQL query string. Use 'dataset' as the table name if parquet_path is provided.
    parquet_path : Path | None
        Optional path to Parquet file to register as 'dataset' view

    Returns
    -------
    pl.DataFrame
        Query result formatted as a Polars DataFrame
    """
    con = duckdb.connect(database=":memory:")
    try:
        if parquet_path:
            norm_path = str(parquet_path).replace("\\", "/")
            con.execute(f"CREATE VIEW dataset AS SELECT * FROM read_parquet('{norm_path}')")

        arrow_table = con.execute(sql_query).arrow()
        return pl.from_arrow(arrow_table)  # type: ignore[return-value]
    finally:
        con.close()


def time_series_split(
    df: pl.DataFrame,
    test_hours: int = 168,  # 7 days
    val_hours: int = 168,  # 7 days
    timestamp_col: str = "timestamp",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Perform strict chronological Train / Validation / Test split without future leakage.

    Parameters
    ----------
    df : pl.DataFrame
        Time-series DataFrame sorted chronologically
    test_hours : int
        Number of hours reserved for final test set (default: 168h = 7 days)
    val_hours : int
        Number of hours reserved for validation set (default: 168h = 7 days)
    timestamp_col : str
        Timestamp column name

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (train_df, val_df, test_df) strictly partitioned without temporal overlap
    """
    if df.is_empty():
        raise ValueError("Cannot split empty DataFrame.")

    sorted_df = df.sort(timestamp_col)
    total_rows = sorted_df.height
    required_min = test_hours + val_hours + 24

    if total_rows < required_min:
        raise ValueError(
            f"Dataset has only {total_rows} rows, but requires at least {required_min} "
            f"for train + val ({val_hours}h) + test ({test_hours}h)."
        )

    val_start_idx = total_rows - (val_hours + test_hours)
    test_start_idx = total_rows - test_hours

    train_df = sorted_df.slice(0, val_start_idx)
    val_df = sorted_df.slice(val_start_idx, val_hours)
    test_df = sorted_df.slice(test_start_idx, test_hours)

    logger.info(
        "Time-series split: Train={} rows, Val={} rows, Test={} rows",
        train_df.height,
        val_df.height,
        test_df.height,
    )

    return train_df, val_df, test_df
