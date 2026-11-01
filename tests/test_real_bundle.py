"""Tests for the real-data bundle (wheat futures + FAO cereals + NOAA ONI).
Uses small fake files in the real formats, so no internet is needed."""
import numpy as np
import pandas as pd
import pytest

from data.loader import REQUIRED_COLS, load_market_data
from envs.famine_market import FamineMarketEnv


def _write_bundle(folder, spike_month="2015-06"):
    # wheat: trading days 2005-2016
    dates = pd.bdate_range("2005-01-03", "2016-12-30")
    rng = np.random.default_rng(0)
    close = 300 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": close}).to_csv(
        folder / "wheat_zw.csv", index=False)

    # FAO: real layout (title rows, blank row, trailing empty columns)
    months = pd.period_range("2003-01", "2016-12", freq="M")
    cereals = [100.0 if m < pd.Period(spike_month) else 150.0 for m in months]
    lines = ["FAO Food Price Index" + "," * 20, "2014-2016=100" + "," * 20,
             "Date,Food Price Index,Meat,Dairy,Cereals,Oils,Sugar" + "," * 14,
             "," * 20]
    for m, c in zip(months, cereals):
        lines.append(f"{m},{c + 5},90,90,{c},90,90" + "," * 14)
    (folder / "fao_ffpi.csv").write_text("\n".join(lines) + "\n")

    # ONI: real layout; DJF 2015 = +1.5, everything else 0
    rows = [" SEAS  YR   TOTAL   ANOM"]
    for yr in range(2003, 2017):
        for s in ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
                  "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]:
            anom = 1.5 if (s, yr) == ("DJF", 2015) else 0.0
            rows.append(f"  {s} {yr}  26.00  {anom:.2f}")
    (folder / "oni.txt").write_text("\n".join(rows) + "\n")


@pytest.fixture
def bundle(tmp_path):
    _write_bundle(tmp_path)
    return tmp_path


def test_bundle_has_required_columns_and_no_nans(bundle):
    df = load_market_data(bundle)
    assert REQUIRED_COLS.issubset(df.columns)
    assert not df.isnull().any().any()
    assert df.index.is_monotonic_increasing
    assert df["wheat_price"].iloc[0] == pytest.approx(200.0)


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="wheat_zw.csv"):
        load_market_data(tmp_path)


def test_fao_shock_appears_only_after_publication_lag(bundle):
    df = load_market_data(bundle)
    # 2015-06 jumps +50% yoy; month end 06-30 + 7 days = published 2015-07-07
    assert df.loc["2015-07-06", "shock_active"] == 0
    assert df.loc["2015-07-07", "shock_active"] == 1
    assert df.loc["2015-07-07", "shock_severity"] == pytest.approx(1.0)
    assert df.loc["2014-03-03", "shock_active"] == 0


def test_oni_mapped_to_centre_month_with_lag(bundle):
    df = load_market_data(bundle)
    # DJF 2015 (centre Jan) ends Feb; published end Feb + 7 days = 2015-03-07
    assert df.loc["2015-03-06", "weather_index"] == 0.0
    assert df.loc["2015-03-09", "weather_index"] == pytest.approx(1.5)
    # JFM 2015 (0.0) replaces it once published (2015-04-07)
    assert df.loc["2015-04-08", "weather_index"] == 0.0


def test_splits_are_chronological_and_disjoint(bundle):
    parts = {s: load_market_data(bundle, split=s) for s in ("train", "val", "test")}
    assert parts["train"].index.max() < parts["val"].index.min()
    assert parts["val"].index.max() < parts["test"].index.min()
    total = sum(len(p) for p in parts.values())
    assert total == len(load_market_data(bundle))


def test_bad_split_name_rejected(bundle):
    with pytest.raises(ValueError):
        load_market_data(bundle, split="holdout")


def test_split_ignored_for_synthetic():
    a = load_market_data(None, n_synthetic_days=300, seed=1, split="train")
    b = load_market_data(None, n_synthetic_days=300, seed=1, split="all")
    assert len(a) == len(b) == 300


def test_env_runs_on_real_bundle_test_split(bundle):
    env = FamineMarketEnv(data_source=bundle, data_split="test",
                          episode_length=100, seed=0)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(100):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r) and env.observation_space.contains(obs)
        if term or trunc:
            break
