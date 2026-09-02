"""Feature engineering pipeline: calendar attributes, Polish statutory holidays,
lags, and renewable indicators.
"""

import math
from collections.abc import Sequence
from datetime import date, timedelta

import polars as pl
from loguru import logger


def get_easter_sunday(year: int) -> date:
    """Compute Easter Sunday for a given Gregorian year using the Butcher's Computus algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l_val = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_val) // 451
    month = (h + l_val - 7 * m + 114) // 31
    day = ((h + l_val - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def get_polish_statutory_holidays(year: int) -> set[date]:
    """Return all official Polish public holidays (dni ustawowo wolne od pracy) for a given year."""
    easter = get_easter_sunday(year)

    holidays: set[date] = {
        date(year, 1, 1),  # Nowy Rok
        date(year, 1, 6),  # Święto Trzech Króli
        easter,  # Wielkanoc (Niedziela Wielkanocna)
        easter + timedelta(days=1),  # Poniedziałek Wielkanocny
        date(year, 5, 1),  # Święto Pracy
        date(year, 5, 3),  # Święto Konstytucji 3 Maja
        easter + timedelta(days=49),  # Zesłanie Ducha Świętego (Zielone Świątki)
        easter + timedelta(days=60),  # Boże Ciało
        date(year, 8, 15),  # Wniebowzięcie NMP / Święto Wojska Polskiego
        date(year, 11, 1),  # Wszystkich Świętych
        date(year, 11, 11),  # Święto Niepodległości
        date(year, 12, 25),  # Boże Narodzenie (dzień 1)
        date(year, 12, 26),  # Boże Narodzenie (dzień 2)
    }
    return holidays


def is_polish_holiday(dt: date) -> bool:
    """Check whether a given date is an official Polish public holiday."""
    holidays = get_polish_statutory_holidays(dt.year)
    return dt in holidays


def add_calendar_features(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Add calendar, temporal cyclical encodings, and Polish statutory holiday indicators.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing timestamp column
    timestamp_col : str
        Name of timestamp column

    Returns
    -------
    pl.DataFrame
        DataFrame augmented with temporal and holiday features
    """
    if df.is_empty():
        return df

    # Extract distinct years from timestamps to pre-calculate all relevant holidays
    timestamps: list[date] = df[timestamp_col].dt.date().unique().to_list()
    years = {d.year for d in timestamps}
    all_holidays: set[date] = set()
    for y in years:
        all_holidays.update(get_polish_statutory_holidays(y))

    # Convert holiday set to list for Polars is_in
    holiday_list = list(all_holidays)

    augmented = df.with_columns(
        [
            pl.col(timestamp_col).dt.hour().alias("hour"),
            pl.col(timestamp_col).dt.weekday().alias("day_of_week"),  # 1=Mon, 7=Sun in polars
            pl.col(timestamp_col).dt.month().alias("month"),
            pl.col(timestamp_col).dt.ordinal_day().alias("day_of_year"),
            # Weekend indicator (Saturday=6, Sunday=7 in Polars)
            (pl.col(timestamp_col).dt.weekday() >= 6).cast(pl.Int32).alias("is_weekend"),
            # Holiday indicator
            pl.col(timestamp_col).dt.date().is_in(holiday_list).cast(pl.Int32).alias("is_holiday"),
        ]
    )

    # Cyclical sin/cos encodings
    augmented = augmented.with_columns(
        [
            (2 * math.pi * pl.col("hour") / 24).sin().alias("sin_hour"),
            (2 * math.pi * pl.col("hour") / 24).cos().alias("cos_hour"),
            (2 * math.pi * (pl.col("day_of_week") - 1) / 7).sin().alias("sin_day_of_week"),
            (2 * math.pi * (pl.col("day_of_week") - 1) / 7).cos().alias("cos_day_of_week"),
            (2 * math.pi * (pl.col("month") - 1) / 12).sin().alias("sin_month"),
            (2 * math.pi * (pl.col("month") - 1) / 12).cos().alias("cos_month"),
        ]
    )

    return augmented


def add_renewable_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute grid renewable indicators, net load, and curtailment risk indicators.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with demand_mw, pv_mw, and wind_mw

    Returns
    -------
    pl.DataFrame
        DataFrame with added renewable interaction metrics
    """
    if df.is_empty():
        return df

    expressions: list[pl.Expr] = []

    has_demand = "demand_mw" in df.columns
    has_pv = "pv_mw" in df.columns
    has_wind = "wind_mw" in df.columns

    if has_pv and has_wind:
        expressions.append((pl.col("pv_mw") + pl.col("wind_mw")).alias("total_renewable_mw"))

    if has_demand and has_pv and has_wind:
        expressions.extend(
            [
                # Net load: demand minus volatile renewables
                (pl.col("demand_mw") - (pl.col("pv_mw") + pl.col("wind_mw"))).alias("net_load_mw"),
                # Renewable penetration ratio
                (
                    (pl.col("pv_mw") + pl.col("wind_mw"))
                    / pl.when(pl.col("demand_mw") > 0).then(pl.col("demand_mw")).otherwise(1.0)
                ).alias("renewable_penetration"),
                # Curtailment Risk Indicator (renewables exceed 75% of demand)
                ((pl.col("pv_mw") + pl.col("wind_mw")) >= (pl.col("demand_mw") * 0.75))
                .cast(pl.Int32)
                .alias("curtailment_risk_flag"),
            ]
        )

    if expressions:
        return df.with_columns(expressions)
    return df


def add_lag_and_rolling_features(
    df: pl.DataFrame,
    target_cols: Sequence[str] = ("demand_mw", "rce_pln_mwh"),
    lags: Sequence[int] = (24, 48, 168),
    rolling_windows: Sequence[int] = (24, 168),
) -> pl.DataFrame:
    """Generate autoregressive lag features and rolling statistics for baseline forecasting.

    Parameters
    ----------
    df : pl.DataFrame
        Hourly time-aligned DataFrame
    target_cols : Sequence[str]
        Columns to generate lags and rolling metrics for
    lags : Sequence[int]
        Lag steps in hours (e.g. 24 = same hour yesterday, 168 = same hour last week)
    rolling_windows : Sequence[int]
        Rolling window sizes in hours

    Returns
    -------
    pl.DataFrame
        DataFrame augmented with lag and rolling features
    """
    if df.is_empty():
        return df

    expressions: list[pl.Expr] = []

    for col in target_cols:
        if col not in df.columns:
            continue

        # Lags
        for lag in lags:
            expressions.append(pl.col(col).shift(lag).alias(f"{col}_lag_{lag}h"))

        # Rolling statistics (computed over shifted series to avoid target leakage)
        for w in rolling_windows:
            expressions.extend(
                [
                    pl.col(col).shift(1).rolling_mean(window_size=w).alias(f"{col}_roll_mean_{w}h"),
                    pl.col(col).shift(1).rolling_std(window_size=w).alias(f"{col}_roll_std_{w}h"),
                ]
            )

    if expressions:
        return df.with_columns(expressions)
    return df


def build_feature_pipeline(
    df: pl.DataFrame,
    include_lags: bool = True,
) -> pl.DataFrame:
    """End-to-end feature engineering pipeline applying calendar, renewable, and lag transforms.

    Parameters
    ----------
    df : pl.DataFrame
        Cleaned hourly dataset
    include_lags : bool
        Whether to generate autoregressive lag and rolling window features

    Returns
    -------
    pl.DataFrame
        Processed dataset with all features engineered
    """
    logger.info("Executing feature engineering pipeline on {} rows.", df.height)
    featured = add_calendar_features(df)
    featured = add_renewable_features(featured)

    if include_lags:
        featured = add_lag_and_rolling_features(featured)

    logger.info("Feature pipeline complete: generated {} columns.", len(featured.columns))
    return featured
