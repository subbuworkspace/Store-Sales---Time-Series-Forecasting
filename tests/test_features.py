import numpy as np
import pandas as pd

from src.features import build_features, zero_flags


def _panel(t):
    return build_features(t["train"], t["test"], t["stores"], t["oil"], t["hol"], t["trans"])


def test_grid_is_complete_and_fills_missing_days(tiny):
    p = _panel(tiny)
    assert len(p.df) == p.n_dates * p.n_series
    assert p.horizon == 16
    xmas = p.df[p.df["date"] == "2015-12-25"]
    assert len(xmas) == p.n_series and xmas["sales"].isna().all()


def test_no_leakage_lags_use_data_at_least_H_days_old(tiny):
    p = _panel(tiny)
    df, H = p.df, p.horizon
    row = df[(df["date"] == "2017-01-05") & (df["store_nbr"] == 2) & (df["family"] == "DAIRY")].iloc[0]
    src_day = row["date"] - pd.Timedelta(days=H)
    expected = df[(df["date"] == src_day) & (df["store_nbr"] == 2) & (df["family"] == "DAIRY")]["y"].iloc[0]
    assert np.isclose(row[f"lag_{H}"], expected, atol=1e-5)
    # the 7-day rolling mean must end on date - H as well
    s = df[(df["store_nbr"] == 2) & (df["family"] == "DAIRY")].set_index("date")["y"]
    window = s.loc[src_day - pd.Timedelta(days=6): src_day]
    assert np.isclose(row["rmean_7"], window.mean(), atol=1e-5)


def test_transferred_holiday_not_flagged_but_transfer_day_is(tiny):
    df = _panel(tiny).df
    flag = lambda d: df.loc[df["date"] == d, "hol_national"].iloc[0]
    assert flag("2015-10-09") == 0   # transferred away
    assert flag("2015-10-12") == 1   # day it was actually celebrated
    assert flag("2015-12-24") == 1   # Additional holiday


def test_local_holiday_only_for_matching_city(tiny):
    df = _panel(tiny).df
    d = df[df["date"] == "2016-07-25"]
    quito = d[d["city"] == "Quito"]["hol_local"].unique()
    cuenca = d[d["city"] == "Cuenca"]["hol_local"].unique()
    assert list(quito) == [1] and list(cuenca) == [0]


def test_zero_flags_detects_dead_series(tiny):
    t = tiny
    t["train"].loc[(t["train"]["store_nbr"] == 1) & (t["train"]["family"] == "SEAFOOD")
                   & (t["train"]["date"] > "2016-11-30"), "sales"] = 0.0
    p = _panel(t)
    zf = zero_flags(p.df, p.n_dates, p.n_series, p.last_train)
    flagged = p.df.iloc[: p.n_series][zf][["store_nbr", "family"]]
    assert list(flagged["store_nbr"].astype(int)) == [1]
    assert list(flagged["family"].astype(str)) == ["SEAFOOD"]
