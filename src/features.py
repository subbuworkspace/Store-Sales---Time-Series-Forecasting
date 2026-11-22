"""Feature engineering for the direct multi-horizon forecast.

Every lag / rolling feature of the target is built from data that is at least
H days old (H = number of test days), so all H test days can be predicted in
one shot, without recursive forecasting.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd

from .config import HOL_TYPES, QUAKE_DATE, ZERO_WINDOW

CATS = ["family", "store_nbr", "city", "state", "type", "cluster"]
ROLL_WINDOWS = (7, 14, 28, 56, 112)


class Panel(NamedTuple):
    df: pd.DataFrame      # long table, ordered date -> store -> family
    feats: list           # model feature names
    cats: list            # categorical feature names
    horizon: int          # H, number of forecast days
    last_train: pd.Timestamp
    n_dates: int
    n_series: int         # store x family combinations


def build_features(train, test, stores, oil, hol, trans) -> Panel:
    last_train = train["date"].max()
    H = test["date"].nunique()

    dates = pd.date_range(train["date"].min(), test["date"].max(), freq="D")
    store_ids = np.sort(train["store_nbr"].unique())
    families = np.sort(train["family"].unique())
    nd, ns = len(dates), len(store_ids) * len(families)

    # Complete date x store x family grid -> row-shift lags are true day-lags
    # (the raw data skips days such as 25 December).
    grid = pd.MultiIndex.from_product(
        [dates, store_ids, families], names=["date", "store_nbr", "family"]
    ).to_frame(index=False)
    cols = ["id", "date", "store_nbr", "family", "onpromotion", "sales"]
    test = test.assign(sales=np.nan)
    obs = pd.concat([train[cols], test[cols]], ignore_index=True)
    df = grid.merge(obs, on=["date", "store_nbr", "family"], how="left")
    df["onpromotion"] = df["onpromotion"].fillna(0)
    df = df.merge(stores, on="store_nbr", how="left")

    # ---- holidays (transfer-aware) ----
    h = hol.copy()
    h["transferred"] = h["transferred"].astype(str).str.lower().eq("true")
    h = h[~h["transferred"]]  # a transferred day is NOT a holiday on its original date
    nat = h[h["locale"] == "National"]
    df["hol_national"] = df["date"].isin(nat.loc[nat["type"].isin(HOL_TYPES), "date"]).astype("int8")
    df["event_national"] = df["date"].isin(nat.loc[nat["type"] == "Event", "date"]).astype("int8")
    df["work_day"] = df["date"].isin(h.loc[h["type"] == "Work Day", "date"]).astype("int8")
    reg = (h[h["locale"] == "Regional"][["date", "locale_name"]].drop_duplicates()
           .rename(columns={"locale_name": "state"}).assign(hol_regional=1))
    loc = (h[h["locale"] == "Local"][["date", "locale_name"]].drop_duplicates()
           .rename(columns={"locale_name": "city"}).assign(hol_local=1))
    df = df.merge(reg, on=["date", "state"], how="left").merge(loc, on=["date", "city"], how="left")
    df[["hol_regional", "hol_local"]] = df[["hol_regional", "hol_local"]].fillna(0).astype("int8")

    # ---- fixed ordering: date-major, then store, then family ----
    df = df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    assert len(df) == nd * ns, "date x store x family grid is not complete"
    df["sid"] = np.tile(np.arange(ns), nd)  # series id per row
    df["y"] = np.log1p(df["sales"])

    def wide(col):  # (n_dates, n_series) matrix
        return pd.DataFrame(df[col].to_numpy().reshape(nd, ns))

    def add(name, w):
        df[name] = w.to_numpy().ravel().astype("float32")

    # ---- target lags / rolling stats (all >= H days old) ----
    Y = wide("y")
    k0 = int(np.ceil(H / 7))
    lags = sorted({H, 364} | {7 * k for k in range(k0, k0 + 4)})  # same-weekday lags
    for L in lags:
        add(f"lag_{L}", Y.shift(L))
    Yh = Y.shift(H)
    for w in ROLL_WINDOWS:
        add(f"rmean_{w}", Yh.rolling(w, min_periods=1).mean())
    add("rstd_28", Yh.rolling(28, min_periods=2).std())
    add("zero_frac_28", (Yh == 0).astype(float).where(Yh.notna()).rolling(28, min_periods=1).mean())

    # ---- promotions (known in the future -> no shift needed) ----
    P = wide("onpromotion")
    add("promo_roll7", P.rolling(7, min_periods=1).mean())
    add("promo_roll28", P.rolling(28, min_periods=1).mean())
    df["promo_log"] = np.log1p(df["onpromotion"]).astype("float32")

    # ---- oil ----
    o = oil.set_index("date")["dcoilwtico"].astype(float).reindex(dates).ffill().bfill()
    df["oil"] = df["date"].map(o).astype("float32")
    df["oil_roll7"] = df["date"].map(o.rolling(7, min_periods=1).mean()).astype("float32")

    # ---- transactions (store level, lagged by H) ----
    T = trans.pivot_table(index="date", columns="store_nbr", values="transactions").reindex(dates)
    Tr = T.shift(H).rolling(28, min_periods=1).mean()
    Tl = (Tr.rename_axis(index="date", columns="store_nbr").reset_index()
          .melt(id_vars="date", value_name="trans_roll28"))
    df = df.merge(Tl, on=["date", "store_nbr"], how="left")
    df = df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)

    # ---- calendar ----
    d = df["date"]
    df["dow"] = d.dt.dayofweek.astype("int8")
    df["dom"] = d.dt.day.astype("int8")
    df["month"] = d.dt.month.astype("int8")
    df["doy"] = d.dt.dayofyear.astype("int16")
    df["woy"] = d.dt.isocalendar().week.astype("int16")
    df["is_month_end"] = d.dt.is_month_end.astype("int8")
    df["is_payday"] = ((d.dt.day == 15) | d.dt.is_month_end).astype("int8")
    dq = (d - QUAKE_DATE).dt.days
    df["quake"] = np.where((dq >= 0) & (dq <= 60), np.exp(-dq.clip(lower=0) / 14.0), 0.0).astype("float32")

    for c in CATS:
        df[c] = df[c].astype("category")

    feats = (
        CATS
        + [f"lag_{L}" for L in lags]
        + [f"rmean_{w}" for w in ROLL_WINDOWS]
        + ["rstd_28", "zero_frac_28", "promo_roll7", "promo_roll28", "promo_log", "onpromotion",
           "oil", "oil_roll7", "trans_roll28",
           "hol_national", "event_national", "work_day", "hol_regional", "hol_local",
           "dow", "dom", "month", "doy", "woy", "is_month_end", "is_payday", "quake"]
    )
    return Panel(df, feats, list(CATS), H, last_train, nd, ns)


def zero_flags(df, n_dates, n_series, end_date, days=ZERO_WINDOW):
    """True for each series with no sales in the `days` days up to `end_date`."""
    S = df["sales"].to_numpy().reshape(n_dates, n_series)
    idx = pd.DatetimeIndex(df["date"].iloc[::n_series])
    m = (idx > end_date - pd.Timedelta(days=days)) & (idx <= end_date)
    return np.nansum(S[m], axis=0) == 0
