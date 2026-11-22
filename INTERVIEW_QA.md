# Interview Q&A - Store Sales Forecasting

**1. What is the problem and the metric?**
Predict daily unit sales for every store x product-family pair for the next 16 days. The metric is RMSLE, which
penalises relative rather than absolute errors and treats under- and over-prediction of small values fairly.

**2. Why train on `log1p(sales)`?**
RMSLE = RMSE computed on `log(1 + y)`. Training with an L2 loss on the log target optimises the competition metric
directly, and it also tames the heavy right skew of sales. Predictions are mapped back with `expm1`.

**3. What is direct vs recursive multi-step forecasting?**
Recursive: predict day 1, feed it back as a lag to predict day 2, and so on - errors compound. Direct: build features
that only use information available at the forecast origin and predict every horizon in one model. Here all lags and
rolling stats are >= 16 days old, so no future values are ever needed.

**4. How did you avoid data leakage?**
Lags and rolling windows are shifted by the horizon (16 days); validation is the last 16 days of train, mirroring the
test period; promotions are allowed unshifted because they are known in advance. A unit test checks that each lag
really equals the value 16 days earlier.

**5. Why build a complete date x store x family grid?**
The raw file skips days (e.g. 25 December). Row-based shifts would then misalign lags. Reindexing to a full grid makes
`shift(k)` a true k-day lag.

**6. How did you handle holidays?**
National, regional (matched on state) and local (matched on city) flags. Rows with `transferred=True` are *not*
holidays on their original date; the matching `Transfer` row marks the day it was actually celebrated. `Bridge`,
`Additional` and `Work Day` rows are handled too.

**7. Which domain effects did you model?**
Public-sector paydays (15th and month-end), the 16 Apr 2016 earthquake (decaying effect for ~60 days), oil price
(Ecuador is oil-dependent), promotions, and day-of-week / yearly seasonality.

**8. Why LightGBM instead of ARIMA/Prophet for 1,782 series?**
One global model shares information across series, handles many covariates (promotions, holidays, store metadata),
trains fast and scales; per-series classical models would be 1,782 separate fits with no cross-learning.

**9. How do you validate a time-series model?**
Never shuffle. Hold out the most recent window that matches the forecast horizon, train on earlier data, then refit
on everything with the best number of rounds (x1.1) for the final prediction. For more robustness use rolling-origin
(expanding-window) cross-validation.

**10. What would you improve next?**
Rolling-origin CV for tuning, per-family models or target encodings, a blend with a second model family, and
time-decay sample weights.
