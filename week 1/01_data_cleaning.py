import pandas as pd
import numpy as np

df = pd.read_csv("hotels.csv")

report = {}
report['raw_shape'] = df.shape
report['raw_duplicates'] = int(df.duplicated().sum())
report['missing_before'] = df.isnull().sum()
report['missing_before'] = report['missing_before'][report['missing_before'] > 0].to_dict()

# 1. Missing values
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df = df.drop(columns=['company'])  # >90% missing
df['agent'] = df['agent'].fillna(0)

# 2. Known data bug: bookings with 0 adults, 0 children, 0 babies (no actual guests)
bad_guest_rows = int(((df.adults == 0) & (df.children == 0) & (df.babies == 0)).sum())
df = df[~((df.adults == 0) & (df.children == 0) & (df.babies == 0))].copy()

# 3. Duplicates
dupes_removed = int(df.duplicated().sum())
df = df.drop_duplicates()

# 4. Outliers in adr (average daily rate): negative values and an extreme outlier
adr_negative = int((df.adr < 0).sum())
adr_extreme = int((df.adr >= 1000).sum())
df = df[(df.adr >= 0) & (df.adr < 1000)].copy()

# 5. Outlier handling for lead_time (cap at 99th percentile rather than drop, since long lead times are plausible)
lead_time_99 = df['lead_time'].quantile(0.99)
report['lead_time_99th_pct'] = float(lead_time_99)

# 6. Dates
df['arrival_date'] = pd.to_datetime(
    df.arrival_date_year.astype(str) + df.arrival_date_month + df.arrival_date_day_of_month.astype(str),
    format='%Y%B%d'
)
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

# 7. Total nights / total guests derived features (useful for EDA)
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['total_guests'] = df['adults'] + df['children'] + df['babies']

report['clean_shape'] = df.shape
report['bad_guest_rows_removed'] = bad_guest_rows
report['duplicate_rows_removed'] = dupes_removed
report['adr_negative_removed'] = adr_negative
report['adr_extreme_removed'] = adr_extreme

df.to_csv("hotels_clean.csv", index=False)

import json
with open("clean_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print(json.dumps(report, indent=2, default=str))
