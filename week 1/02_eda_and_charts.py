import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

df = pd.read_csv("hotels_clean.csv", parse_dates=['arrival_date', 'reservation_status_date'])

stats = {}

# Summary stats by hotel type
stats['by_hotel_counts'] = df['hotel'].value_counts().to_dict()
stats['cancel_rate_overall'] = round(df['is_canceled'].mean() * 100, 2)
stats['cancel_rate_by_hotel'] = (df.groupby('hotel')['is_canceled'].mean() * 100).round(2).to_dict()
stats['adr_by_hotel'] = df.groupby('hotel')['adr'].mean().round(2).to_dict()
stats['lead_time_mean'] = round(df['lead_time'].mean(), 1)
stats['lead_time_canceled_vs_not'] = df.groupby('is_canceled')['lead_time'].mean().round(1).to_dict()
stats['avg_total_nights'] = round(df['total_nights'].mean(), 2)
stats['top_countries'] = df['country'].value_counts().head(5).to_dict()
stats['cancel_by_segment'] = (df.groupby('market_segment')['is_canceled'].mean() * 100).round(2).sort_values(ascending=False).to_dict()
stats['cancel_by_deposit'] = (df.groupby('deposit_type')['is_canceled'].mean() * 100).round(2).to_dict()

corr_cols = ['lead_time', 'adr', 'total_nights', 'total_guests', 'previous_cancellations',
             'booking_changes', 'total_of_special_requests', 'days_in_waiting_list', 'is_canceled']
corr = df[corr_cols].corr()
stats['corr_with_cancel'] = corr['is_canceled'].drop('is_canceled').round(3).sort_values(ascending=False).to_dict()

with open("eda_stats.json", "w") as f:
    json.dump(stats, f, indent=2, default=str)
print(json.dumps(stats, indent=2, default=str))

# ---------- Charts ----------

# 1. Histogram: lead_time
plt.figure(figsize=(7,4.5))
sns.histplot(df['lead_time'], bins=40, color="#3b6ea5")
plt.title("Distribution of Booking Lead Time")
plt.xlabel("Lead Time (days)")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.savefig("charts/hist_lead_time.png")
plt.close()

# 2. Histogram: adr
plt.figure(figsize=(7,4.5))
sns.histplot(df['adr'], bins=40, color="#3b6ea5")
plt.title("Distribution of Average Daily Rate (ADR)")
plt.xlabel("ADR (currency units)")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.savefig("charts/hist_adr.png")
plt.close()

# 3. Box plot: adr by hotel
plt.figure(figsize=(7,4.5))
sns.boxplot(data=df, x='hotel', y='adr', palette=["#3b6ea5", "#a5573b"])
plt.title("ADR Distribution by Hotel Type")
plt.xlabel("")
plt.ylabel("ADR")
plt.tight_layout()
plt.savefig("charts/box_adr_hotel.png")
plt.close()

# 4. Bar: cancellation rate by month
month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
monthly = df.groupby('arrival_date_month')['is_canceled'].mean().reindex(month_order) * 100
plt.figure(figsize=(8,4.5))
monthly.plot(kind='bar', color="#3b6ea5")
plt.title("Cancellation Rate by Arrival Month")
plt.xlabel("Month")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/bar_cancel_month.png")
plt.close()

# 5. Bar: cancellation rate by market segment
plt.figure(figsize=(8,4.5))
seg = df.groupby('market_segment')['is_canceled'].mean().sort_values(ascending=False) * 100
seg.plot(kind='bar', color="#a5573b")
plt.title("Cancellation Rate by Market Segment")
plt.xlabel("Market Segment")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("charts/bar_cancel_segment.png")
plt.close()

# 6. Correlation heatmap
plt.figure(figsize=(7.5,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink":0.8})
plt.title("Correlation Matrix of Key Numeric Features")
plt.tight_layout()
plt.savefig("charts/heatmap_corr.png")
plt.close()

# 7. Scatter: lead_time vs adr, colored by cancellation
plt.figure(figsize=(7.5,5))
sample = df.sample(4000, random_state=42)
sns.scatterplot(data=sample, x='lead_time', y='adr', hue='is_canceled', alpha=0.5, palette=["#3b6ea5", "#a5573b"], s=18)
plt.title("Lead Time vs. ADR (sampled), by Cancellation Status")
plt.xlabel("Lead Time (days)")
plt.ylabel("ADR")
plt.legend(title="Canceled", labels=["No","Yes"])
plt.tight_layout()
plt.savefig("charts/scatter_leadtime_adr.png")
plt.close()

print("Charts saved.")
