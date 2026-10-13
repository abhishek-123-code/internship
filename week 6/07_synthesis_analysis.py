import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
BLUE, ORANGE, GREEN, GREY = "#3b6ea5", "#a5573b", "#3f8f5b", "#777777"
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]

df = pd.read_csv("hotels_segmented.csv", parse_dates=["arrival_date"])
df["month"] = pd.Categorical(df.arrival_date.dt.month_name(), MONTHS, ordered=True)

fam = df.groupby("month").apply(lambda g: (g.cluster == 0).mean() * 100).reindex(MONTHS)
loy = df.groupby("month").apply(lambda g: (g.cluster == 2).mean() * 100).reindex(MONTHS)
adr = df.groupby("month").adr.mean().reindex(MONTHS)

out = {}
out["fam_share_by_month"] = fam.round(1).to_dict()
out["loy_share_by_month"] = loy.round(1).to_dict()
out["fam_overall_share"] = round((df.cluster == 0).mean() * 100, 1)
out["loy_overall_share"] = round((df.cluster == 2).mean() * 100, 1)
out["fam_peak_share"] = round(fam.max(), 1)
out["fam_peak_month"] = fam.idxmax()
out["loy_trough_share"] = round(loy.min(), 1)
out["loy_trough_month"] = loy.idxmin()
q = df.adr.quantile([0.2, 0.4, 0.6, 0.8]).values
fam_adr_med = df[df.cluster == 0].adr.median()
out["fam_adr_quintile"] = int(np.searchsorted(q, fam_adr_med) + 1)
out["fam_cancel_rate"] = round(df[df.cluster == 0].is_canceled.mean() * 100, 1)
json.dump(out, open("synth_stats.json", "w"), indent=2)

fig, ax1 = plt.subplots(figsize=(9.5, 4.3))
ax2 = ax1.twinx()
ax1.plot(MONTHS, fam.values, "o-", color=ORANGE, lw=2, label="Families segment: % of bookings")
ax1.plot(MONTHS, loy.values, "o-", color=GREEN, lw=2, label="Loyal Regulars segment: % of bookings")
ax2.bar(MONTHS, adr.values, color=BLUE, alpha=0.18, label="Mean ADR (right axis)")
ax1.set_ylabel("Segment share of bookings (%)")
ax2.set_ylabel("Mean ADR")
ax1.set_title("Segment Mix Tracks Seasonal Demand and Price")
ax1.tick_params(axis="x", rotation=45)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("charts/synth_segment_season.png")
plt.close()
print(json.dumps(out, indent=1))
