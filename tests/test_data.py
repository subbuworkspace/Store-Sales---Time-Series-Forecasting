import pandas as pd
import pytest

from src.data import load_datasets, parse_dates


def test_parse_iso():
    out = parse_dates(pd.Series(["2017-08-15", "2017-08-16"]))
    assert out.iloc[0] == pd.Timestamp("2017-08-15")


def test_parse_us_format():
    out = parse_dates(pd.Series(["1/1/2013", "8/15/2017"]))
    assert out.iloc[1] == pd.Timestamp("2017-08-15")


def test_parse_day_first_format():
    out = parse_dates(pd.Series(["1/1/2013", "15/8/2017"]))
    assert out.iloc[1] == pd.Timestamp("2017-08-15")


def test_missing_files_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing files"):
        load_datasets(tmp_path)
