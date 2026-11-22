import json

import pandas as pd

from src.train import run


def test_end_to_end_writes_valid_submission(tiny_csv_dir, tmp_path):
    out = tmp_path / "out"
    metrics = run(tiny_csv_dir, out, num_boost_round=30, early_stopping=10, verbose=False)

    sub = pd.read_csv(out / "submission.csv")
    sample = pd.read_csv(tiny_csv_dir / "sample_submission.csv")
    assert list(sub.columns) == ["id", "sales"]
    assert len(sub) == len(sample) and set(sub["id"]) == set(sample["id"])
    assert sub["sales"].notna().all() and (sub["sales"] >= 0).all()

    saved = json.loads((out / "metrics.json").read_text())
    assert saved["horizon_days"] == 16
    assert metrics["validation_rmsle_model"] < metrics["validation_rmsle_baseline"] * 1.5
