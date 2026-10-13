import warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
BLUE, ORANGE, GREEN, GREY = "#3b6ea5", "#a5573b", "#3f8f5b", "#777777"
out = {}
RS = 42

df = pd.read_csv("hotels_clean.csv", parse_dates=["arrival_date"])
ok = df[df.is_canceled == 0].copy()
out["n_bookings"] = len(df)
out["n_stayed"] = len(ok)

# ---------------------------------------------------------------- 1. Daily hotel-level panel
CAP = {"City Hotel": 90, "Resort Hotel": 65}   # assumption: ~99th-percentile daily arrivals, rounded
out["capacity_assumption"] = CAP
daily = ok.groupby(["hotel", "arrival_date"]).agg(bookings=("adr", "size"), adr=("adr", "mean"),
                                                    nights=("total_nights", "mean"), revenue=("adr", lambda s: (s * ok.loc[s.index, "total_nights"]).sum())).reset_index()
daily["capacity"] = daily.hotel.map(CAP)
daily["occupancy"] = (daily.bookings / daily.capacity).clip(upper=1.0)
daily["revpar"] = daily.adr * daily.occupancy
daily["dow"] = daily.arrival_date.dt.day_name()
daily["month"] = daily.arrival_date.dt.month_name()
daily["is_weekend"] = daily.arrival_date.dt.dayofweek >= 5
daily["month_num"] = daily.arrival_date.dt.month
daily.to_csv("daily_panel.csv", index=False)
out["daily_shape"] = daily.shape
out["mean_occupancy"] = daily.groupby("hotel").occupancy.mean().round(3).to_dict()
out["mean_adr"] = daily.groupby("hotel").adr.mean().round(2).to_dict()
out["mean_revpar"] = daily.groupby("hotel").revpar.mean().round(2).to_dict()

# ---------------------------------------------------------------- 2. Seasonality and weekday patterns
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
seas = daily.groupby(["hotel", "month"]).agg(adr=("adr", "mean"), occ=("occupancy", "mean"), revpar=("revpar", "mean")).reset_index()
seas["month"] = pd.Categorical(seas.month, MONTHS, ordered=True)
seas = seas.sort_values(["hotel", "month"])
out["seasonality"] = {h: g.set_index("month")[["adr", "occ", "revpar"]].round(2).to_dict("index") for h, g in seas.groupby("hotel")}

DOWS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow = daily.groupby(["hotel", "dow"]).agg(adr=("adr", "mean"), occ=("occupancy", "mean")).reset_index()
dow["dow"] = pd.Categorical(dow.dow, DOWS, ordered=True)
out["dow"] = {h: g.set_index("dow")[["adr", "occ"]].round(2).to_dict("index") for h, g in dow.groupby("hotel")}
out["weekend_premium"] = (daily.groupby(["hotel", "is_weekend"]).adr.mean().unstack().assign(
    premium_pct=lambda d: (d[True] / d[False] - 1) * 100).round(2)).to_dict("index")

fig, ax = plt.subplots(figsize=(9, 4))
for h, c in zip(["City Hotel", "Resort Hotel"], [BLUE, ORANGE]):
    s = daily[daily.hotel == h].set_index("arrival_date").adr.rolling(14).mean()
    ax.plot(s, color=c, lw=1.2, label=h)
ax.set_title("ADR Trend, 14-Day Rolling Average"); ax.set_ylabel("ADR"); ax.legend()
plt.tight_layout(); plt.savefig("charts/rv_trend.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for h, c in zip(["City Hotel", "Resort Hotel"], [BLUE, ORANGE]):
    g = seas[seas.hotel == h]
    ax[0].plot(g.month, g.adr, "o-", color=c, label=h)
    ax[1].plot(g.month, g.occ * 100, "o-", color=c, label=h)
ax[0].set_title("Average ADR by Month"); ax[0].tick_params(axis="x", rotation=45); ax[0].legend()
ax[1].set_title("Average Occupancy % by Month"); ax[1].tick_params(axis="x", rotation=45); ax[1].legend()
plt.tight_layout(); plt.savefig("charts/rv_seasonality.png"); plt.close()

piv = daily.pivot_table(index="dow", columns="month", values="adr", aggfunc="mean").reindex(DOWS)[[m for m in MONTHS if m in daily.month.unique()]]
plt.figure(figsize=(10, 4))
sns.heatmap(piv, cmap="YlOrRd", annot=True, fmt=".0f", cbar_kws={"label": "Mean ADR"})
plt.title("ADR Heatmap: Day of Week x Month (both hotels combined)"); plt.xlabel(""); plt.ylabel("")
plt.tight_layout(); plt.savefig("charts/rv_heatmap.png"); plt.close()

# ---------------------------------------------------------------- 3. Hedonic price regression: what predicts ADR at booking level
reg = ok.copy()
reg["log_adr"] = np.log1p(reg.adr)
reg["log_lead"] = np.log1p(reg.lead_time)
reg["weekend_share"] = np.where(reg.total_nights > 0, reg.stays_in_weekend_nights / reg.total_nights.replace(0, np.nan), 0)
reg["weekend_share"] = reg.weekend_share.fillna(0)
reg["month"] = pd.Categorical(pd.to_datetime(reg.arrival_date).dt.month_name(), MONTHS, ordered=True)
reg["hotel_c"] = pd.Categorical(reg.hotel)
reg["segment_c"] = pd.Categorical(reg.market_segment)
sub = reg.sample(min(40000, len(reg)), random_state=RS)  # subsample for speed with many dummies

formula = "log_adr ~ log_lead + total_nights + total_guests + has_children + C(hotel_c) + C(month) + C(segment_c) + weekend_share + total_of_special_requests"
sub["has_children"] = ((sub.children + sub.babies) > 0).astype(int)
model = smf.ols(formula, data=sub).fit(cov_type="HC1")
out["hedonic_r2"] = round(model.rsquared, 3)
out["hedonic_r2_adj"] = round(model.rsquared_adj, 3)
out["hedonic_n"] = int(model.nobs)
coefs = model.params.drop([c for c in model.params.index if c == "Intercept"])
pvals = model.pvalues.drop([c for c in model.pvalues.index if c == "Intercept"])
key_terms = ["log_lead", "total_nights", "total_guests", "has_children", "weekend_share", "total_of_special_requests", "C(hotel_c)[T.Resort Hotel]"]
out["hedonic_key"] = {t: {"coef": round(float(coefs[t]), 4), "p": round(float(pvals[t]), 4)} for t in key_terms if t in coefs.index}
month_coefs = {k.split("T.")[1].rstrip("]"): round(float(v), 4) for k, v in coefs.items() if k.startswith("C(month)")}
seg_coefs = {k.split("T.")[1].rstrip("]"): round(float(v), 4) for k, v in coefs.items() if k.startswith("C(segment_c)")}
out["hedonic_month"] = month_coefs
out["hedonic_segment"] = seg_coefs
with open("hedonic_summary.txt", "w") as f:
    f.write(model.summary().as_text())

fig, ax = plt.subplots(figsize=(8, 4.2))
items = sorted(month_coefs.items(), key=lambda x: MONTHS.index(x[0]) if x[0] in MONTHS else 0)
ax.bar([i[0][:3] for i in items], [i[1] * 100 for i in items], color=BLUE)
ax.set_title("Month Effect on ADR vs. January (%), Controlling for Booking Characteristics")
ax.set_ylabel("% difference in ADR"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("charts/rv_month_effect.png"); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.2))
items = sorted(seg_coefs.items(), key=lambda x: x[1])
ax.barh([i[0] for i in items], [i[1] * 100 for i in items], color=ORANGE)
ax.set_title("Market Segment Effect on ADR vs. Aviation (%)"); ax.axvline(0, color="k", lw=0.6)
plt.tight_layout(); plt.savefig("charts/rv_segment_effect.png"); plt.close()

# ---------------------------------------------------------------- 4. Price-demand relationship / elasticity
d = daily.copy()
d["log_bookings"] = np.log(d.bookings)
d["log_adr"] = np.log(d.adr.clip(lower=1))
naive = smf.ols("log_bookings ~ log_adr", data=d).fit(cov_type="HC1")
out["elasticity_naive"] = {"coef": round(float(naive.params["log_adr"]), 3), "p": round(float(naive.pvalues["log_adr"]), 4), "r2": round(naive.rsquared, 3)}

ctrl = smf.ols("log_bookings ~ log_adr + C(month) + C(dow) + C(hotel)", data=d).fit(cov_type="HC1")
out["elasticity_controlled"] = {"coef": round(float(ctrl.params["log_adr"]), 3), "p": round(float(ctrl.pvalues["log_adr"]), 4), "r2": round(ctrl.rsquared, 3)}

# within-hotel, within-month elasticity (detrend both series by month x hotel mean, then regress residuals)
d["adr_resid"] = d.groupby(["hotel", "month"]).adr.transform(lambda s: s - s.mean())
d["book_resid"] = d.groupby(["hotel", "month"]).bookings.transform(lambda s: s - s.mean())
sp_r, sp_p = stats.pearsonr(d.adr_resid, d.book_resid)
out["within_month_corr"] = {"r": round(float(sp_r), 3), "p": round(float(sp_p), 4)}
within = smf.ols("book_resid ~ adr_resid", data=d).fit(cov_type="HC1")
out["within_month_slope"] = round(float(within.params["adr_resid"]), 4)

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
axes[0].scatter(d.adr, d.bookings, s=6, alpha=0.25, color=BLUE)
axes[0].set_title("Raw: Daily ADR vs. Bookings"); axes[0].set_xlabel("ADR"); axes[0].set_ylabel("Bookings")
axes[1].scatter(np.exp(d.log_adr), np.exp(ctrl.fittedvalues), s=6, alpha=0.2, color=GREY)
axes[1].set_title("Controlled Model: Fitted vs. ADR"); axes[1].set_xlabel("ADR"); axes[1].set_ylabel("Fitted bookings")
axes[2].scatter(d.adr_resid, d.book_resid, s=6, alpha=0.25, color=ORANGE)
xs = np.linspace(d.adr_resid.min(), d.adr_resid.max(), 50)
axes[2].plot(xs, within.params["Intercept"] + within.params["adr_resid"] * xs, color="black", lw=1.5)
axes[2].set_title("Within Month x Hotel: ADR vs. Bookings\n(deviations from group mean)", fontsize=9)
axes[2].set_xlabel("ADR deviation"); axes[2].set_ylabel("Bookings deviation")
plt.tight_layout(); plt.savefig("charts/rv_elasticity.png"); plt.close()

# ---------------------------------------------------------------- 5. Cancellation-adjusted revenue risk by price band
b = df.copy()
b["price_band"] = pd.qcut(b.adr, 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
pb = b.groupby("price_band").agg(bookings=("adr", "size"), cancel_rate=("is_canceled", "mean"), mean_adr=("adr", "mean")).reset_index()
pb["expected_revpar_index"] = (pb.mean_adr * (1 - pb.cancel_rate))
pb["expected_revpar_index"] = (pb.expected_revpar_index / pb.expected_revpar_index.iloc[0] * 100).round(1)
out["price_band"] = pb.round(3).to_dict("records")

fig, ax = plt.subplots(figsize=(7.5, 4))
ax2 = ax.twinx()
ax.bar(pb.price_band, pb.cancel_rate * 100, color=ORANGE, alpha=0.7, label="Cancellation rate %")
ax2.plot(pb.price_band, pb.mean_adr, "o-", color=BLUE, lw=2, label="Mean ADR")
ax.set_ylabel("Cancellation rate (%)"); ax2.set_ylabel("Mean ADR"); ax.set_title("Cancellation Risk by Price Band")
lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.tight_layout(); plt.savefig("charts/rv_price_band.png"); plt.close()

# ---------------------------------------------------------------- 6. Competitive benchmark: City vs Resort
comp = ok.groupby("hotel").agg(adr=("adr", "mean"), nights=("total_nights", "mean"), guests=("total_guests", "mean"),
                                lead=("lead_time", "mean")).round(2)
comp["cancel_rate"] = df.groupby("hotel").is_canceled.mean().round(3) * 100
out["competitive"] = comp.to_dict("index")
chan = ok.groupby(["hotel", "market_segment"]).adr.mean().unstack().round(1)
out["channel_adr"] = chan.to_dict("index")
chan_share = (ok.groupby(["hotel", "market_segment"]).size().unstack().fillna(0))
chan_share = (chan_share.div(chan_share.sum(axis=1), axis=0) * 100).round(1)
out["channel_share"] = chan_share.to_dict("index")

# ---------------------------------------------------------------- 7. Why the estimated elasticity cannot drive pricing directly
# The naive and controlled log-log coefficients above are POSITIVE (demand rises with price).
# This is the classic simultaneity problem in observational pricing data: hotels raise ADR through
# their own revenue-management systems precisely WHEN they anticipate high demand, so price and demand
# are jointly determined by the same underlying seasonal/demand shock. The historical data therefore
# traces out a "reaction curve", not a demand curve, and cannot be used to size a price change.
out["endogeneity_note"] = True

# ---------------------------------------------------------------- 8. Illustrative revenue-optimisation scenarios
# Because a causal demand elasticity cannot be identified from this dataset, optimisation is run under
# ASSUMED elasticities spanning the range commonly cited in hotel revenue-management literature
# (inelastic loyal/corporate demand to more elastic price-sensitive leisure demand). These are what-if
# scenarios to demonstrate the RevPAR-optimisation method, not a prediction from the data itself.
ASSUMED_E = {"Inelastic (-0.5)": -0.5, "Moderate (-1.0)": -1.0, "Elastic (-1.8)": -1.8}

def price_scenario(hotel_name, e):
    g = daily[daily.hotel == hotel_name]
    base_adr = g.adr.mean(); base_bookings = g.bookings.mean(); cap = CAP[hotel_name]
    rows = []
    for chg in np.arange(-0.40, 0.61, 0.02):
        new_adr = base_adr * (1 + chg)
        new_bookings = base_bookings * (1 + chg) ** e
        new_occ = min(max(new_bookings, 0) / cap, 1.0)
        new_revpar = new_adr * new_occ
        rows.append((round(chg * 100, 1), round(new_adr, 1), round(new_occ * 100, 1), round(new_revpar, 2)))
    return pd.DataFrame(rows, columns=["price_change_pct", "adr", "occupancy_pct", "revpar"])

scenarios = {}
opt_table = []
for hotel_name in ["City Hotel", "Resort Hotel"]:
    for label, e in ASSUMED_E.items():
        sc = price_scenario(hotel_name, e)
        scenarios[(hotel_name, label)] = sc
        opt = sc.loc[sc.revpar.idxmax()]
        base = sc[sc.price_change_pct == 0].iloc[0]
        opt_table.append(dict(hotel=hotel_name, elasticity=label, opt_price_chg=opt.price_change_pct,
                              opt_adr=opt.adr, opt_occ=opt.occupancy_pct, opt_revpar=opt.revpar,
                              base_revpar=base.revpar, revpar_gain_pct=round((opt.revpar / base.revpar - 1) * 100, 1)))
    sc.to_csv(f"scenario_{hotel_name.split()[0].lower()}.csv", index=False)
out["opt_table"] = opt_table

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
cols = {"Inelastic (-0.5)": GREEN, "Moderate (-1.0)": BLUE, "Elastic (-1.8)": ORANGE}
for a, hotel_name in zip(ax, ["City Hotel", "Resort Hotel"]):
    for label, e in ASSUMED_E.items():
        sc = scenarios[(hotel_name, label)]
        a.plot(sc.price_change_pct, sc.revpar, "-", color=cols[label], lw=2, label=label)
        opt = sc.loc[sc.revpar.idxmax()]
        a.scatter([opt.price_change_pct], [opt.revpar], color=cols[label], zorder=5, edgecolor="black")
    a.axvline(0, color="grey", ls="--", lw=0.8)
    a.set_title(f"{hotel_name}: Illustrative RevPAR vs. Price Change"); a.set_xlabel("Price change (%)"); a.set_ylabel("RevPAR (proxy)")
    a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("charts/rv_scenario.png"); plt.close()

json.dump(out, open("revenue_stats.json", "w"), indent=2, default=str)
print(json.dumps({k: out[k] for k in ["mean_occupancy", "mean_adr", "mean_revpar", "hedonic_r2", "hedonic_key",
                  "elasticity_naive", "elasticity_controlled", "within_month_corr", "within_month_slope",
                  "opt_table", "weekend_premium", "capacity_assumption", "competitive"]}, indent=1, default=str))
