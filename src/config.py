"""Central configuration: paths, constants and LightGBM parameters."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Put the Kaggle CSVs in data/raw, or point STORE_SALES_DATA / --data-dir elsewhere.
DATA_DIR = Path(os.environ.get("STORE_SALES_DATA", ROOT / "data" / "raw"))
OUTPUT_DIR = ROOT / "outputs"

REQUIRED_FILES = [
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
]

# Rows before this date are dropped: many store/family series had not opened yet
# (all-zero history) and only add noise.
TRAIN_START = "2015-01-01"

# A series with no sales at all in this many days up to the forecast origin is predicted as 0.
ZERO_WINDOW = 21

# Magnitude 7.8 earthquake in Ecuador - affected supermarket sales for weeks.
QUAKE_DATE = pd.Timestamp("2016-04-16")
HOL_TYPES = ["Holiday", "Transfer", "Additional", "Bridge"]

LGB_PARAMS = dict(
    objective="regression",  # L2 on log1p(sales) == optimises RMSLE
    metric="rmse",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=42,
)
