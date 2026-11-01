"""
data/loader.py
--------------
Unified data loader: returns a standardised DataFrame regardless of whether
the source is a real CSV (FAO, World Bank, USDA) or the synthetic generator.

Usage:
    from data.loader import load_market_data
    df = load_market_data()                      # synthetic
    df = load_market_data("data/raw/fao.csv")    # real FAO CSV
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.synthetic import SyntheticMarket, MarketConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required columns the environment expects (after loading)
# ---------------------------------------------------------------------------
REQUIRED_COLS = {
    "wheat_price",
    "supply_index",
    "shock_active",
    "shock_severity",
    "weather_index",
}


def load_market_data(
    source: Optional[str | Path] = None,
    n_synthetic_days: int = 2000,
    seed: int = 42,
    commodity: str = "wheat",
    split: str = "all",
) -> pd.DataFrame:
    """
    Load market data from a file or generate it synthetically.

    Parameters
    ----------
    source : path to a CSV file, or None to use synthetic data
    n_synthetic_days : days to generate when using synthetic data
    seed : random seed for synthetic generation
    commodity : primary commodity name (used when parsing real CSVs)
    split : "all", "train", "val" or "test". Real data is split chronologically
        (first 60% / next 15% / last 25% of trading days). Ignored for
        synthetic data, whose train/eval sets differ by seed instead.

    Returns
    -------
    DataFrame with DatetimeIndex and at least the REQUIRED_COLS columns.

    `source` may be a single CSV file, or a folder (e.g. "data/raw") holding
    wheat_zw.csv, fao_ffpi.csv and oni.txt -- the real-data bundle.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")

    if source is None:
        logger.info("No source provided -- using synthetic market data.")
        return _generate_synthetic(n_synthetic_days, seed)

    path = Path(source)
    if not path.exists():
        logger.warning("File %s not found -- falling back to synthetic data.", path)
        return _generate_synthetic(n_synthetic_days, seed)

    logger.info("Loading real data from %s", path)
    if path.is_dir():
        df = _load_real_bundle(path)
    else:
        df = _load_csv(path, commodity)
    return _apply_split(df, split)


SPLITS = {"all", "train", "val", "test"}
SPLIT_BOUNDS = (0.60, 0.75)   # train | val | test  (60% / 15% / 25%)

# Real-data bundle: file names expected inside the data folder
WHEAT_FILE = "wheat_zw.csv"     # Date,Close  (yfinance ZW=F, trading days)
FAO_FILE = "fao_ffpi.csv"       # FAO Food Price Index (monthly, has Cereals)
ONI_FILE = "oni.txt"            # NOAA ENSO index, whitespace table

# A published value only becomes usable some days after its period ends,
# otherwise the agent would see the future.
FAO_LAG_DAYS = 7    # FAO publishes in the first days of the next month
ONI_LAG_DAYS = 7    # ONI for a 3-month season is final ~1 month after it ends

# FAO cereals year-over-year jump that counts as a shock
SHOCK_YOY_THRESHOLD = 0.15    # +15% vs a year ago -> shock_active
SHOCK_YOY_SATURATION = 0.45   # +45% or more -> shock_severity = 1.0

ONI_SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
               "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


# ---------------------------------------------------------------------------
# Synthetic path
# ---------------------------------------------------------------------------

def _generate_synthetic(n_days: int, seed: int) -> pd.DataFrame:
    market = SyntheticMarket(seed=seed)
    df = market.generate(n_days=n_days)
    _validate(df)
    return df


# ---------------------------------------------------------------------------
# Real-data CSV parsers
# ---------------------------------------------------------------------------

def _load_csv(path: Path, commodity: str) -> pd.DataFrame:
    """
    Attempt to detect and parse the CSV format.
    Currently supports:
      - FAO Food Price Index (monthly)
      - World Bank Pink Sheet (monthly, wide format)
      - Generic: any CSV with a date column and a price column
    """
    raw = pd.read_csv(path, header=0, encoding="utf-8-sig")
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    # --- FAO format detection ---
    if "food_price_index" in raw.columns or "cereals_price_index" in raw.columns:
        return _parse_fao(raw, commodity)

    # --- World Bank Pink Sheet ---
    if "commodity" in raw.columns or "monthly_prices" in str(path).lower():
        return _parse_world_bank(raw, commodity)

    # --- Generic fallback ---
    return _parse_generic(raw, commodity)


def _parse_fao(raw: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """
    FAO Food Price Index CSV.
    Expected columns: Date, Cereal Price Index, (others)
    """
    # FAO date column is typically 'date' or 'month'
    date_col = _find_col(raw, ["date", "month", "period"])
    price_col = _find_col(raw, ["cereal", "cereals", "food", "wheat"])

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[date_col])
    df["wheat_price"] = pd.to_numeric(raw[price_col], errors="coerce")
    df = df.dropna().set_index("date").sort_index()

    # Resample to daily via forward-fill (FAO is monthly)
    df = df.resample("D").ffill()

    # Add synthetic macro columns (no shock/supply data in FAO)
    df = _augment_with_synthetic_macro(df, seed=0)
    _validate(df)
    return df


def _parse_world_bank(raw: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """World Bank Pink Sheet wide format."""
    date_col = _find_col(raw, ["date", "month", "period", "year"])
    # Find commodity price column: fuzzy-match on commodity name
    price_col = _find_col(raw, [commodity, "wheat", "maize", "price"])

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[date_col])
    df["wheat_price"] = pd.to_numeric(raw[price_col], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    df = df.resample("D").ffill()
    df = _augment_with_synthetic_macro(df, seed=1)
    _validate(df)
    return df


def _parse_generic(raw: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """Last-resort generic parser: find a date column and a numeric price column."""
    date_col = _find_col(raw, ["date", "time", "datetime", "day", "month", "year"])
    price_col = _find_col(raw, [commodity, "price", "close", "value", "index"])

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[date_col])
    df["wheat_price"] = pd.to_numeric(raw[price_col], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    df = df.resample("D").ffill()
    df = _augment_with_synthetic_macro(df, seed=2)
    _validate(df)
    return df


def _augment_with_synthetic_macro(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    When real price data is used but macro columns are absent, generate
    synthetic supply/shock/weather signals aligned to the same date range.
    """
    n = len(df)
    market = SyntheticMarket(seed=seed)
    syn = market.generate(n_days=n, start_date=str(df.index[0].date()))

    for col in ["supply_index", "shock_active", "shock_severity", "weather_index", "conflict_index"]:
        if col not in df.columns:
            df[col] = syn[col].values

    # Normalise price to same scale as synthetic baseline
    if "wheat_price" in df.columns:
        scale = 200.0 / df["wheat_price"].iloc[0]
        df["wheat_price"] = df["wheat_price"] * scale

    return df


# ---------------------------------------------------------------------------
# Real-data bundle: wheat futures + FAO cereals (shocks) + NOAA ONI (weather)
# ---------------------------------------------------------------------------

def _load_real_bundle(folder: Path) -> pd.DataFrame:
    """
    Combine three real sources into the table the environment expects.

    wheat_price     <- wheat futures close (trading days only, no weekends)
    shock_active    <- FAO cereals index up more than 15% vs a year ago
    shock_severity  <- same change scaled to 0..1 (15% -> 0, 45%+ -> 1)
    weather_index   <- NOAA ONI anomaly (El Nino > 0, La Nina < 0)
    supply_index    <- PLACEHOLDER constant 0.5 (no real data yet)
    conflict_index  <- PLACEHOLDER constant 0.0 (no real data yet)
    """
    paths = {n: folder / n for n in (WHEAT_FILE, FAO_FILE, ONI_FILE)}
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {folder}. Download them into that folder "
            f"(see README, 'Real data')."
        )

    prices = _read_wheat_futures(paths[WHEAT_FILE])
    index = prices.index
    df = pd.DataFrame({"wheat_price": prices.values}, index=index)

    yoy = _asof(_read_fao_cereals_yoy(paths[FAO_FILE]), index).fillna(0.0)
    span = SHOCK_YOY_SATURATION - SHOCK_YOY_THRESHOLD
    df["shock_active"] = (yoy > SHOCK_YOY_THRESHOLD).astype(int)
    df["shock_severity"] = ((yoy - SHOCK_YOY_THRESHOLD) / span).clip(0.0, 1.0)

    df["weather_index"] = _asof(_read_oni(paths[ONI_FILE]), index).fillna(0.0)

    df["supply_index"] = 0.5
    df["conflict_index"] = 0.0
    logger.warning(
        "supply_index and conflict_index are constant placeholders "
        "(no real data wired in yet)."
    )

    # Same price scale as the synthetic data (first price -> 200).
    df["wheat_price"] = df["wheat_price"] * (200.0 / df["wheat_price"].iloc[0])
    _validate(df)
    return df


def _read_wheat_futures(path: Path) -> pd.Series:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    date_col = _find_col(raw, ["date"])
    price_col = _find_col(raw, ["close", "price"])
    dates = pd.to_datetime(raw[date_col], errors="coerce")
    vals = pd.to_numeric(raw[price_col], errors="coerce")
    mask = dates.notna() & vals.notna() & (vals > 0)
    s = pd.Series(vals[mask].values, index=pd.DatetimeIndex(dates[mask]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index = s.index.astype("datetime64[ns]")
    if len(s) < 300:
        raise ValueError(f"{path} has only {len(s)} usable price rows.")
    return s


def _read_fao_cereals_yoy(path: Path) -> pd.Series:
    """Year-over-year change of the FAO Cereals index, indexed by the date
    each value became public (month end + FAO_LAG_DAYS)."""
    with open(path, encoding="utf-8-sig") as fh:
        lines = fh.readlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().lower().startswith("date,")),
        None,
    )
    if header_idx is None:
        raise ValueError(f"No 'Date,...' header row found in {path}.")

    raw = pd.read_csv(path, skiprows=header_idx, encoding="utf-8-sig")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    date_col = _find_col(raw, ["date"])
    price_col = _find_col(raw, ["cereal"])

    dates = pd.to_datetime(raw[date_col], format="%Y-%m", errors="coerce")
    if dates.notna().sum() == 0:
        dates = pd.to_datetime(raw[date_col], errors="coerce")
    vals = pd.to_numeric(raw[price_col], errors="coerce")
    mask = dates.notna() & vals.notna()

    s = pd.Series(vals[mask].values, index=pd.PeriodIndex(dates[mask], freq="M"))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    yoy = s / s.shift(12) - 1.0

    avail = s.index.to_timestamp(how="end").normalize() + pd.Timedelta(days=FAO_LAG_DAYS)
    out = pd.Series(yoy.values, index=avail.astype("datetime64[ns]")).dropna()
    return out.sort_index()


def _read_oni(path: Path) -> pd.Series:
    """NOAA ONI anomaly, indexed by the date each season's value became public.

    Rows look like 'DJF 1950 25.01 -1.32'. A season is labelled by its centre
    month (DJF -> January) and is final ~1 month after it ends.
    """
    raw = pd.read_csv(path, sep=r"\s+", engine="python")
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    for need in ("SEAS", "YR", "ANOM"):
        if need not in raw.columns:
            raise ValueError(f"{path}: expected columns SEAS YR TOTAL ANOM, got {list(raw.columns)}")

    season_no = raw["SEAS"].astype(str).str.strip().map(
        {name: i for i, name in enumerate(ONI_SEASONS)}
    )
    year = pd.to_numeric(raw["YR"], errors="coerce")
    anom = pd.to_numeric(raw["ANOM"], errors="coerce")
    mask = season_no.notna() & year.notna() & anom.notna()

    centre = pd.PeriodIndex.from_fields(
        year=year[mask].astype(int).values,
        month=season_no[mask].astype(int).values + 1,
        freq="M",
    )
    avail = (centre + 1).to_timestamp(how="end").normalize() + pd.Timedelta(days=ONI_LAG_DAYS)
    out = pd.Series(anom[mask].values, index=avail.astype("datetime64[ns]"))
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _asof(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Value of `series` as known on each date in `index` (last published
    value at or before that date; NaN before the first publication)."""
    return series.reindex(index.astype("datetime64[ns]"), method="ffill")


def _apply_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Chronological split: train | val | test = 60% | 15% | 25% of rows."""
    if split == "all":
        return df
    n = len(df)
    i1, i2 = int(n * SPLIT_BOUNDS[0]), int(n * SPLIT_BOUNDS[1])
    part = {"train": df.iloc[:i1], "val": df.iloc[i1:i2], "test": df.iloc[i2:]}[split]
    logger.info("Using %s split: %s to %s (%d rows)", split,
                part.index[0].date(), part.index[-1].date(), len(part))
    return part.copy()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column matching a candidate.

    Candidates are tried in priority order (earlier = preferred), so
    ["cereals", "food"] picks a 'cereals' column even if a 'food_price_index'
    column appears earlier in the file.
    """
    for cand in candidates:
        for col in df.columns:
            if cand in col:
                return col
    raise ValueError(
        f"Could not find a column matching any of {candidates} in {list(df.columns)}"
    )


def _validate(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Loaded data is missing required columns: {missing}")
    if df.isnull().any().any():
        n_nan = df.isnull().sum().sum()
        logger.warning("Data contains %d NaN values — forward-filling.", n_nan)
        df.ffill(inplace=True)
        df.bfill(inplace=True)


if __name__ == "__main__":
    df = load_market_data()
    print(df.head())
    print(df.dtypes)
    print(f"Rows: {len(df)}")
