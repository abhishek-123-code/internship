import warnings, json, itertools
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
BLUE, ORANGE, GREEN, GREY = "#3b6ea5", "#a5573b", "#3f8f5b", "#777777"
out = {}

# ------------------------------------------------------------------ 1. Data
df = pd.read_csv("hotels_clean.csv", parse_dates=["arrival_date"])
y = df[df.is_canceled == 0].groupby("arrival_date").size().asfreq("D")
out["n_days"] = len(y)
out["missing_days"] = int(y.isna().sum())
out["start"], out["end"] = str(y.index.min().date()), str(y.index.max().date())

# anomaly check: extreme-outlier fence (3 x IQR) on robust STL residuals
stl = STL(y, period=7, robust=True).fit()
resid = stl.resid
q1, q3 = resid.quantile([0.25, 0.75])
iqr = q3 - q1
anom = resid[(resid < q1 - 3 * iqr) | (resid > q3 + 3 * iqr)]
out["anomalies_flagged"] = [(str(i.date()), int(y[i])) for i in anom.index]
out["desc"] = y.describe().round(2).to_dict()

# ------------------------------------------------------------------ 2. Exploration charts
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(y, color=BLUE, lw=0.8, label="Daily arrivals")
ax.plot(y.rolling(28, center=True).mean(), color=ORANGE, lw=2, label="28-day rolling mean")
ax.set_title("Daily Arrivals of Non-Cancelled Bookings (Jul 2015 - Aug 2017)")
ax.set_ylabel("Bookings arriving"); ax.legend()
plt.tight_layout(); plt.savefig("charts/ts_series.png"); plt.close()

stl_full = STL(y, period=7, robust=True).fit()
stl_year = STL(y.rolling(7, center=True, min_periods=1).mean(), period=365, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
axes[0].plot(y, color=BLUE, lw=0.7); axes[0].set_ylabel("Observed")
axes[1].plot(stl_year.trend, color=ORANGE); axes[1].set_ylabel("Trend")
axes[2].plot(stl_full.seasonal, color=GREEN, lw=0.7); axes[2].set_ylabel("Weekly")
axes[3].plot(stl_full.resid, color=GREY, lw=0.6); axes[3].set_ylabel("Residual")
fig.suptitle("STL Decomposition (weekly seasonal component; smoothed annual trend)")
plt.tight_layout(); plt.savefig("charts/ts_stl.png"); plt.close()

dow = y.groupby(y.index.dayofweek).mean()
mon = y.groupby(y.index.month).mean()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].bar(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], dow.values, color=BLUE); axes[0].set_title("Mean Arrivals by Weekday")
axes[1].bar(["J","F","M","A","M","J","J","A","S","O","N","D"], mon.values, color=ORANGE); axes[1].set_title("Mean Arrivals by Month")
plt.tight_layout(); plt.savefig("charts/ts_seasonality.png"); plt.close()

# stationarity
adf_p = adfuller(y)[1]
adf_p_d = adfuller(y.diff().dropna())[1]
adf_p_s = adfuller(y.diff(7).dropna())[1]
out["adf_p_level"], out["adf_p_diff1"], out["adf_p_seasdiff7"] = round(adf_p, 4), round(adf_p_d, 6), round(adf_p_s, 6)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
plot_acf(y.diff(7).dropna(), lags=42, ax=axes[0]); axes[0].set_title("ACF (after seasonal diff, lag 7)")
plot_pacf(y.diff(7).dropna(), lags=42, ax=axes[1], method="ywm"); axes[1].set_title("PACF (after seasonal diff, lag 7)")
plt.tight_layout(); plt.savefig("charts/ts_acf_pacf.png"); plt.close()

# ------------------------------------------------------------------ 3. Split
H = 56
train, test = y.iloc[:-H], y.iloc[-H:]
out["train_range"] = [str(train.index.min().date()), str(train.index.max().date()), len(train)]
out["test_range"] = [str(test.index.min().date()), str(test.index.max().date()), len(test)]

def fourier(index, K, period=365.25, t0=None):
    t = (index - pd.Timestamp("2015-07-01")).days.values.astype(float)
    cols = {}
    for k in range(1, K + 1):
        cols[f"s{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"c{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)

def metrics(actual, pred, train_series):
    e = actual.values - np.asarray(pred)
    mae = np.mean(np.abs(e)); rmse = np.sqrt(np.mean(e ** 2))
    mape = np.mean(np.abs(e) / actual.values) * 100
    smape = np.mean(2 * np.abs(e) / (np.abs(actual.values) + np.abs(pred))) * 100
    scale = np.mean(np.abs(train_series.values[7:] - train_series.values[:-7]))
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, sMAPE=smape, MASE=mae / scale)

# ------------------------------------------------------------------ 4. Models
def f_snaive7(tr, h):
    last = tr.iloc[-7:].values
    return np.tile(last, int(np.ceil(h / 7)))[:h]

def f_snaive_avg4(tr, h):
    last = np.mean([tr.iloc[-7 * (i + 1):len(tr) - 7 * i].values for i in range(4)], axis=0)
    return np.tile(last, int(np.ceil(h / 7)))[:h]

def f_hw(tr, h, trend=None, damped=False):
    m = ExponentialSmoothing(tr, trend=trend, damped_trend=damped, seasonal="add", seasonal_periods=7,
                             initialization_method="estimated").fit()
    return m.forecast(h).values

def f_sarimax(tr, h, order=(1, 0, 1), sorder=(1, 1, 1, 7), K=3):
    ex_tr, ex_te = fourier(tr.index, K), fourier(pd.date_range(tr.index[-1] + pd.Timedelta(days=1), periods=h), K)
    m = SARIMAX(tr, exog=ex_tr, order=order, seasonal_order=sorder,
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
    return m.forecast(h, exog=ex_te).values

# --- Parameter tuning (SARIMAX order + Fourier K) by AIC on training data
grid = []
for p, q, P, Q, K in itertools.product([0, 1, 2], [0, 1], [0, 1], [1], [2, 3, 4]):
    try:
        ex = fourier(train.index, K)
        m = SARIMAX(train, exog=ex, order=(p, 0, q), seasonal_order=(P, 1, Q, 7),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=150)
        grid.append(dict(p=p, q=q, P=P, Q=Q, K=K, AIC=m.aic))
    except Exception as e:
        pass
grid = pd.DataFrame(grid).sort_values("AIC").reset_index(drop=True)
grid.to_csv("tuning_grid.csv", index=False)
best = grid.iloc[0]
out["best_params"] = {k: (int(best[k])) for k in ["p", "q", "P", "Q", "K"]}
out["best_aic"] = round(float(best["AIC"]), 1)
out["grid_top5"] = grid.head(5).round(1).to_dict("records")
out["grid_size"] = len(grid)
bp = out["best_params"]

models = {
    "Seasonal Naive (7d)": lambda tr, h: f_snaive7(tr, h),
    "Seasonal Avg (last 4 wks)": lambda tr, h: f_snaive_avg4(tr, h),
    "Holt-Winters (add. season)": lambda tr, h: f_hw(tr, h),
    "Holt-Winters (damped trend)": lambda tr, h: f_hw(tr, h, trend="add", damped=True),
    "SARIMA(1,1,1)(1,1,1,7) no annual": lambda tr, h: SARIMAX(tr, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(h).values,
    "SARIMAX + Fourier (tuned)": lambda tr, h: f_sarimax(tr, h, (bp["p"], 0, bp["q"]), (bp["P"], 1, bp["Q"], 7), bp["K"]),
}

preds, res = {}, {}
for name, fn in models.items():
    p_ = fn(train, H)
    preds[name] = p_
    res[name] = metrics(test, p_, train)
res_df = pd.DataFrame(res).T.round(2)
res_df.to_csv("holdout_metrics.csv")
out["holdout"] = res_df.to_dict("index")

# --- Rolling-origin cross-validation (3 folds x 28d) on the training portion
cv_res = {n: [] for n in models}
for cut in [len(train) - 84, len(train) - 56, len(train) - 28]:
    tr_, te_ = train.iloc[:cut], train.iloc[cut:cut + 28]
    for name, fn in models.items():
        try:
            cv_res[name].append(metrics(te_, fn(tr_, 28), tr_)["MAE"])
        except Exception:
            cv_res[name].append(np.nan)
cv_df = pd.DataFrame({k: [np.nanmean(v)] for k, v in cv_res.items()}, index=["CV MAE (3 folds)"]).T.round(2)
out["cv"] = cv_df["CV MAE (3 folds)"].to_dict()

best_name = res_df["MAE"].idxmin()
out["best_holdout_model"] = best_name

# ------------------------------------------------------------------ 5. Plots
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(train.iloc[-90:], color=GREY, lw=1, label="Training (last 90 days)")
ax.plot(test, color="black", lw=1.6, label="Actual")
ax.plot(test.index, preds["SARIMAX + Fourier (tuned)"], color=ORANGE, lw=1.6, label="SARIMAX + Fourier")
ax.plot(test.index, preds["Holt-Winters (add. season)"], color=BLUE, lw=1.4, ls="--", label="Holt-Winters")
ax.plot(test.index, preds["Seasonal Naive (7d)"], color=GREEN, lw=1.2, ls=":", label="Seasonal naive")
ax.set_title("56-Day Holdout: Forecast vs. Actual"); ax.set_ylabel("Bookings arriving"); ax.legend(ncol=2)
plt.tight_layout(); plt.savefig("charts/fc_holdout.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
order = res_df.sort_values("MAE").index
ax[0].barh(order, res_df.loc[order, "MAE"], color=BLUE); ax[0].invert_yaxis(); ax[0].set_title("Holdout MAE (lower is better)")
ax[1].barh(order, res_df.loc[order, "MASE"], color=ORANGE); ax[1].invert_yaxis(); ax[1].set_title("Holdout MASE"); ax[1].axvline(1, color="k", ls="--", lw=0.8)
ax[1].set_yticklabels([])
plt.tight_layout(); plt.savefig("charts/fc_compare.png"); plt.close()

# residual diagnostics for the tuned model (in-sample)
ex_tr = fourier(train.index, bp["K"])
fit = SARIMAX(train, exog=ex_tr, order=(bp["p"], 0, bp["q"]), seasonal_order=(bp["P"], 1, bp["Q"], 7),
              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
r = fit.resid.iloc[14:]
lb = acorr_ljungbox(r, lags=[7, 14, 28], return_df=True)
out["ljungbox_p"] = lb["lb_pvalue"].round(4).to_dict()
out["resid_mean"], out["resid_std"] = round(float(r.mean()), 2), round(float(r.std()), 2)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
axes[0].plot(r, color=BLUE, lw=0.6); axes[0].set_title("Residuals over time")
sns.histplot(r, bins=30, kde=True, ax=axes[1], color=ORANGE); axes[1].set_title("Residual distribution")
plot_acf(r, lags=28, ax=axes[2]); axes[2].set_title("Residual ACF")
plt.tight_layout(); plt.savefig("charts/fc_residuals.png"); plt.close()

# ------------------------------------------------------------------ 6. Final forecast (refit on all data, 8 weeks ahead)
H2 = 56
ex_all = fourier(y.index, bp["K"])
fut_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=H2)
ex_fut = fourier(fut_idx, bp["K"])
fit_all = SARIMAX(y, exog=ex_all, order=(bp["p"], 0, bp["q"]), seasonal_order=(bp["P"], 1, bp["Q"], 7),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
fc = fit_all.get_forecast(H2, exog=ex_fut)
mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.2)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(y.iloc[-120:], color="black", lw=1, label="Observed")
ax.plot(mean, color=ORANGE, lw=1.6, label="Forecast")
ax.fill_between(mean.index, ci.iloc[:, 0], ci.iloc[:, 1], color=ORANGE, alpha=0.2, label="80% interval")
ax.set_title("8-Week-Ahead Forecast (Sep - Oct 2017), model refit on all data"); ax.set_ylabel("Bookings arriving"); ax.legend()
plt.tight_layout(); plt.savefig("charts/fc_future.png"); plt.close()
out["future_mean_first_week"] = round(float(mean.iloc[:7].mean()), 1)
out["future_mean_last_week"] = round(float(mean.iloc[-7:].mean()), 1)

# interval coverage check on holdout (80%)
fit_tr = fit
fc_h = fit_tr.get_forecast(H, exog=fourier(test.index, bp["K"]))
ci_h = fc_h.conf_int(alpha=0.2)
cover = ((test.values >= ci_h.iloc[:, 0].values) & (test.values <= ci_h.iloc[:, 1].values)).mean()
out["coverage_80"] = round(float(cover) * 100, 1)

# weekday-wise error for best
e = pd.Series(test.values - preds["SARIMAX + Fourier (tuned)"], index=test.index)
out["bias_tuned"] = round(float(e.mean()), 2)
out["worst_days"] = [(str(i.date()), round(float(v), 1)) for i, v in e.abs().sort_values(ascending=False).head(3).items()]

json.dump(out, open("forecast_stats.json", "w"), indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
print(res_df)
print(cv_df)
