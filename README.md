# Store Sales - Time Series Forecasting

[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

Forecasting daily unit sales for **33 product families x 54 Favorita stores** (Ecuador) for the next
16 days, 

## Approach

| Step | What |
|------|------|
| Direct forecasting | All lag / rolling features are built from data **at least 16 days old**, so every test day is predicted in one shot - no recursive error build-up. |
| Complete panel | A full date x store x family grid is built so missing days (e.g. 25 Dec) don't corrupt lags. |
| Target | `log1p(sales)` with an L2 objective, which optimises RMSLE directly. |
| Features | Same-weekday lags (21/28/35/42/364), rolling mean/std, promotions, oil price, lagged store transactions, store metadata, national / regional / local holidays (transfer-aware), paydays (15th & month-end), earthquake effect (16 Apr 2016). |
| Model | LightGBM, early stopping on the last 16 days of train, then refit on all data. |
| Post-processing | Predictions clipped at 0; series with no sales in the last 21 days are forced to 0. |

## Repository structure

```
store-sales-forecasting/
├── src/
│   ├── config.py        # paths, constants, LightGBM params
│   ├── data.py          # CSV loading (handles ISO and Excel-style dates)
│   ├── features.py      # feature engineering
│   └── train.py         # validate -> refit -> submission.csv
├── notebooks/01_eda.ipynb
├── tests/               # pytest suite (leakage, holidays, end-to-end)
├── docs/INTERVIEW_QA.md # concepts + interview questions for this project
├── data/raw/            # put the Kaggle CSVs here (git-ignored)
├── outputs/             # submission.csv, metrics.json, feature_importance.csv (git-ignored)
├── .vscode/             # interpreter, pytest and run configs
├── .github/workflows/   # CI: runs pytest on every push
├── requirements.txt
└── requirements-dev.txt
```

## Quick start (Windows + VS Code)

```powershell
git clone https://github.com/<your-username>/store-sales-forecasting.git
cd store-sales-forecasting

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

1. Download the data from the [competition data page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
   and put the 7 CSV files in `data/raw/` (see [data/README.md](data/README.md)).
2. Run the pipeline:

```powershell
python -m src.train
# or, with the data somewhere else:
python -m src.train --data-dir "D:\path\to\Store Sales data"
```

3. Results are written to `outputs/`:
   - `submission.csv` - upload this on the competition's *Submit Predictions* page
   - `metrics.json` - validation RMSLE for the model and a 28-day-mean baseline
   - `feature_importance.csv`

In VS Code: open the folder, select the `.venv` interpreter (`Ctrl+Shift+P` -> *Python: Select Interpreter*),
then press **F5** and pick *Train + predict*.

Run the tests with `pytest -q`.

## Results

| Model | Validation RMSLE (last 16 days of train) | Public leaderboard |
|-------|------------------------------------------|--------------------|
| 28-day mean baseline | _fill in from `outputs/metrics.json`_ | - |
| LightGBM (this repo) | _fill in_ | _fill in_ |

## Ideas to improve

- Tune LightGBM (`num_leaves`, `min_data_in_leaf`, `learning_rate`) with time-based CV.
- Train one model per product family, or add family x store target encodings.
- Add sales-since-last-promotion and holiday-distance features.
- Blend with a second model (e.g. XGBoost / CatBoost, or a linear model on trend + seasonality).
- Weight recent months more heavily.

## Data source

Competition data (c) Corporacion Favorita, provided via Kaggle under the competition rules. The data is not
redistributed in this repository.

## License

MIT - see [LICENSE](LICENSE).
