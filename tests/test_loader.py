"""Tests for the real-data CSV paths in data/loader.py (no internet needed)."""
import numpy as np
import pandas as pd

from data.loader import REQUIRED_COLS, load_market_data


def _write_fao(path):
    d = pd.date_range("2015-01-01", periods=96, freq="MS")
    pd.DataFrame({
        "Date": d,
        "Food Price Index": np.linspace(100, 140, 96),
        "Cereals": np.linspace(90, 200, 96),
    }).to_csv(path, index=False)


def test_fao_csv_loads_with_required_columns(tmp_path):
    f = tmp_path / "fao.csv"
    _write_fao(f)
    df = load_market_data(f)
    assert REQUIRED_COLS.issubset(df.columns)
    assert df.index.is_monotonic_increasing
    assert not df.isnull().any().any()


def test_fao_uses_cereals_not_overall_index(tmp_path):
    f = tmp_path / "fao.csv"
    _write_fao(f)
    df = load_market_data(f)
    ratio = df["wheat_price"].iloc[-1] / df["wheat_price"].iloc[0]
    assert abs(ratio - 200 / 90) < 0.05   # cereals ratio, not 140/100


def test_generic_daily_csv_loads(tmp_path):
    f = tmp_path / "zw.csv"
    dates = pd.bdate_range("2015-01-01", periods=600)
    pd.DataFrame({"Date": dates, "Close": 500 + np.cumsum(np.random.default_rng(0).normal(size=600))}).to_csv(f, index=False)
    df = load_market_data(f)
    assert REQUIRED_COLS.issubset(df.columns)
    assert len(df) >= 600   # weekends forward-filled to daily
