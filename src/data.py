"""Loading and light cleaning of the Kaggle CSV files."""
from pathlib import Path

import pandas as pd

from .config import REQUIRED_FILES


def parse_dates(s: pd.Series) -> pd.Series:
    """Strict date parse that tolerates Excel-style files.

    Tries ISO (YYYY-MM-DD), then M/D/YYYY, then D/M/YYYY. A wrong guess raises on
    any day > 12, so the first format that parses the whole column is the right one.
    """
    s = s.astype(str).str.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(s)  # last resort


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = parse_dates(df["date"])
    return df


def load_datasets(data_dir) -> dict:
    """Load all competition files into a dict of DataFrames."""
    data_dir = Path(data_dir)
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in '{data_dir}': {', '.join(missing)}.\n"
            "Download them from https://www.kaggle.com/competitions/"
            "store-sales-time-series-forecasting/data and put them in data/raw/ "
            "(or pass --data-dir)."
        )
    return {
        "train": read_csv(data_dir / "train.csv"),
        "test": read_csv(data_dir / "test.csv"),
        "sample": pd.read_csv(data_dir / "sample_submission.csv"),
        "stores": pd.read_csv(data_dir / "stores.csv"),
        "oil": read_csv(data_dir / "oil.csv"),
        "holidays": read_csv(data_dir / "holidays_events.csv"),
        "transactions": read_csv(data_dir / "transactions.csv"),
    }
