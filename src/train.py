"""Train, validate and predict.

    python -m src.train                       # data in data/raw
    python -m src.train --data-dir "D:\\path\\to\\Store Sales data"
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import DATA_DIR, LGB_PARAMS, OUTPUT_DIR, TRAIN_START
from .data import load_datasets
from .features import build_features, zero_flags


def rmsle_from_log(y_log, p_log) -> float:
    """Targets/predictions are already log1p-transformed -> RMSLE == RMSE."""
    return float(np.sqrt(np.mean((np.asarray(y_log) - np.asarray(p_log)) ** 2)))


def run(data_dir=DATA_DIR, out_dir=OUTPUT_DIR, train_start=TRAIN_START,
        num_boost_round=5000, early_stopping=100, verbose=True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    ds = load_datasets(data_dir)
    panel = build_features(ds["train"], ds["test"], ds["stores"], ds["oil"],
                           ds["holidays"], ds["transactions"])
    df, feats, cats, H = panel.df, panel.feats, panel.cats, panel.horizon
    last_train, nd, ns = panel.last_train, panel.n_dates, panel.n_series
    say(f"Train up to {last_train.date()} | forecast horizon: {H} days | series: {ns}")

    has_y = (df["y"].notna() & (df["date"] >= pd.Timestamp(train_start))
             & (df["date"] <= last_train))
    valid_start = last_train - pd.Timedelta(days=H - 1)
    fit = df[has_y & (df["date"] < valid_start)]
    val = df[has_y & (df["date"] >= valid_start)]
    say(f"Fit rows: {len(fit):,} | validation rows: {len(val):,} "
        f"({valid_start.date()} -> {last_train.date()})")

    # ---- 1) validation on the last H days of train ----
    dfit = lgb.Dataset(fit[feats], fit["y"], categorical_feature=cats, free_raw_data=False)
    dval = lgb.Dataset(val[feats], val["y"], categorical_feature=cats, reference=dfit)
    callbacks = [lgb.early_stopping(early_stopping, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(100))
    booster = lgb.train(LGB_PARAMS, dfit, num_boost_round=num_boost_round,
                        valid_sets=[dval], valid_names=["valid"], callbacks=callbacks)
    best = booster.best_iteration or num_boost_round

    pv = booster.predict(val[feats], num_iteration=best)
    zf = zero_flags(df, nd, ns, valid_start - pd.Timedelta(days=1))
    pv = np.where(zf[val["sid"].to_numpy()], 0.0, np.clip(pv, 0, None))
    model_score = rmsle_from_log(val["y"], pv)
    baseline_score = rmsle_from_log(val["y"], val["rmean_28"].fillna(0))
    say(f"\nValidation RMSLE  model    : {model_score:.5f}")
    say(f"Validation RMSLE  baseline : {baseline_score:.5f}  (28-day mean)")

    # ---- 2) refit on everything and predict the test period ----
    full = df[has_y]
    dfull = lgb.Dataset(full[feats], full["y"], categorical_feature=cats)
    final = lgb.train(LGB_PARAMS, dfull, num_boost_round=max(50, int(best * 1.1)))

    te = df[df["date"] > last_train]
    pt = final.predict(te[feats])
    zf = zero_flags(df, nd, ns, last_train)
    pt = np.where(zf[te["sid"].to_numpy()], 0.0, np.clip(pt, 0, None))
    sub = pd.DataFrame({"id": te["id"].astype(int).to_numpy(), "sales": np.expm1(pt)})

    sample = ds["sample"]
    if set(sample["id"]) != set(sub["id"]):
        raise ValueError("Prediction ids do not match sample_submission.csv")
    sub = sample[["id"]].merge(sub, on="id", how="left")
    sub.to_csv(out_dir / "submission.csv", index=False)

    imp = (pd.DataFrame({"feature": feats, "gain": final.feature_importance("gain")})
           .sort_values("gain", ascending=False))
    imp.to_csv(out_dir / "feature_importance.csv", index=False)

    metrics = {
        "validation_rmsle_model": model_score,
        "validation_rmsle_baseline": baseline_score,
        "best_iteration": int(best),
        "horizon_days": int(H),
        "n_series": int(ns),
        "last_train_date": str(last_train.date()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    say(f"\nSaved {out_dir / 'submission.csv'} ({len(sub):,} rows)")
    say(imp.head(10).to_string(index=False))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Store Sales forecasting pipeline")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="folder with the Kaggle CSVs")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--train-start", default=TRAIN_START)
    args = ap.parse_args()
    run(args.data_dir, args.out_dir, args.train_start)


if __name__ == "__main__":
    main()
