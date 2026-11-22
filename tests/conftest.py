import numpy as np
import pandas as pd
import pytest

FAMILIES = ["BEVERAGES", "DAIRY", "SEAFOOD"]
STORES = [1, 2, 3]


@pytest.fixture
def tiny():
    """Small synthetic dataset in the Kaggle schema (3 stores x 3 families)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-06-01", "2016-12-31")
    tdates = pd.date_range("2017-01-01", "2017-01-16")  # H = 16

    def make(ds, with_sales, start):
        rows, i = [], start
        for d in ds:
            for s in STORES:
                for f in FAMILIES:
                    sales = float(rng.poisson(20 + 5 * s + (8 if d.dayofweek >= 5 else 0)))
                    rows.append((i, d, s, f, sales if with_sales else np.nan, int(rng.integers(0, 4))))
                    i += 1
        return pd.DataFrame(rows, columns=["id", "date", "store_nbr", "family", "sales", "onpromotion"])

    train = make(dates, True, 0)
    train = train[~((train["date"].dt.month == 12) & (train["date"].dt.day == 25))]  # raw data skips 25 Dec
    test = make(tdates, False, 100000).drop(columns="sales")
    stores = pd.DataFrame({
        "store_nbr": STORES, "city": ["Quito", "Quito", "Cuenca"],
        "state": ["Pichincha", "Pichincha", "Azuay"], "type": ["A", "B", "A"], "cluster": [1, 2, 1],
    })
    all_dates = pd.date_range(dates.min(), tdates.max())
    oil = pd.DataFrame({"date": all_dates, "dcoilwtico": rng.uniform(40, 60, len(all_dates))})
    hol = pd.DataFrame({
        "date": pd.to_datetime(["2015-12-24", "2015-10-09", "2015-10-12", "2016-07-25"]),
        "type": ["Additional", "Holiday", "Transfer", "Holiday"],
        "locale": ["National", "National", "National", "Local"],
        "locale_name": ["Ecuador", "Ecuador", "Ecuador", "Quito"],
        "description": ["x", "orig (transferred)", "moved", "local"],
        "transferred": ["False", "True", "False", "False"],
    })
    tt = train[["date", "store_nbr"]].drop_duplicates()
    trans = tt.assign(transactions=rng.integers(500, 3000, len(tt)))
    return dict(train=train, test=test, stores=stores, oil=oil, hol=hol, trans=trans)


@pytest.fixture
def tiny_csv_dir(tiny, tmp_path):
    """Write the tiny dataset as CSVs using Excel-style M/D/YYYY dates."""
    def fmt(df):
        df = df.copy()
        df["date"] = df["date"].dt.strftime("%m/%d/%Y")
        return df

    d = tmp_path / "raw"
    d.mkdir()
    fmt(tiny["train"]).to_csv(d / "train.csv", index=False)
    fmt(tiny["test"]).to_csv(d / "test.csv", index=False)
    tiny["test"][["id"]].assign(sales=0).to_csv(d / "sample_submission.csv", index=False)
    tiny["stores"].to_csv(d / "stores.csv", index=False)
    fmt(tiny["oil"]).to_csv(d / "oil.csv", index=False)
    fmt(tiny["hol"]).to_csv(d / "holidays_events.csv", index=False)
    fmt(tiny["trans"]).to_csv(d / "transactions.csv", index=False)
    return d
