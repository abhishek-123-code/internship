import warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
PAL = ["#3b6ea5", "#a5573b", "#3f8f5b", "#8a5fa8", "#c9a227", "#777777"]
out = {}
RS = 42

df = pd.read_csv("hotels_clean.csv")
out["n_bookings"] = len(df)

# ------------------------------------------------ 1. Feature engineering
df["weekend_share"] = np.where(df.total_nights > 0, df.stays_in_weekend_nights / df.total_nights.replace(0, np.nan), 0)
df["weekend_share"] = df["weekend_share"].fillna(0)
df["has_children"] = ((df.children + df.babies) > 0).astype(int)
df["has_parking"] = (df.required_car_parking_spaces > 0).astype(int)
df["is_repeat"] = df.is_repeated_guest
df["prior_cancel"] = (df.previous_cancellations > 0).astype(int)
df["booking_changed"] = (df.booking_changes > 0).astype(int)

FEATS = ["lead_time", "total_nights", "total_guests", "adr", "total_of_special_requests",
         "booking_changed", "has_children", "is_repeat", "prior_cancel", "weekend_share"]
X = df[FEATS].copy()
skew_before = X[["lead_time", "total_nights", "adr"]].skew().round(2).to_dict()
for c in ["lead_time", "total_nights"]:
    X[c] = np.log1p(X[c])
X["adr"] = X["adr"].clip(upper=X["adr"].quantile(0.995))  # log-transform made ADR skew worse (zero-rate stays); winsorise instead
skew_after = X[["lead_time", "total_nights", "adr"]].skew().round(2).to_dict()
out["skew_before"], out["skew_after"] = skew_before, skew_after
out["features"] = FEATS

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# ------------------------------------------------ 2. Choose k
rng = np.random.RandomState(RS)
sub = rng.choice(len(Xs), 15000, replace=False)
rows = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(Xs)
    lab = km.labels_
    rows.append(dict(k=k, inertia=km.inertia_,
                     silhouette=silhouette_score(Xs[sub], lab[sub]),
                     CH=calinski_harabasz_score(Xs, lab),
                     DB=davies_bouldin_score(Xs, lab)))
kdf = pd.DataFrame(rows)
kdf.to_csv("kmeans_selection.csv", index=False)
out["k_table"] = kdf.round(3).to_dict("records")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
ax[0].plot(kdf.k, kdf.inertia, "o-", color=PAL[0]); ax[0].set_title("Elbow (inertia)")
ax[1].plot(kdf.k, kdf.silhouette, "o-", color=PAL[1]); ax[1].set_title("Silhouette (higher better)")
ax[2].plot(kdf.k, kdf.DB, "o-", color=PAL[2]); ax[2].set_title("Davies-Bouldin (lower better)")
for a in ax: a.set_xlabel("k")
plt.tight_layout(); plt.savefig("charts/seg_kselect.png"); plt.close()

# stability: ARI between fits on two bootstrap halves, evaluated on full data
def stability(k, reps=6):
    aris = []
    for r in range(reps):
        r1 = np.random.RandomState(r).choice(len(Xs), len(Xs) // 2, replace=False)
        r2 = np.random.RandomState(100 + r).choice(len(Xs), len(Xs) // 2, replace=False)
        a = KMeans(k, n_init=5, random_state=r).fit(Xs[r1]).predict(Xs)
        b = KMeans(k, n_init=5, random_state=r + 1).fit(Xs[r2]).predict(Xs)
        aris.append(adjusted_rand_score(a, b))
    return float(np.mean(aris))
stab = {k: round(stability(k), 3) for k in range(2, 9)}
out["stability_ari"] = stab

# ------------------------------------------------ 3. Final model (decided after inspecting metrics)
K = int(json.load(open("k_choice.json"))["k"]) if __import__("os").path.exists("k_choice.json") else 4
out["K"] = K
km = KMeans(n_clusters=K, n_init=25, random_state=RS).fit(Xs)
df["cluster"] = km.labels_
out["final_metrics"] = dict(
    silhouette=round(float(silhouette_score(Xs[sub], km.labels_[sub])), 3),
    CH=round(float(calinski_harabasz_score(Xs, km.labels_)), 1),
    DB=round(float(davies_bouldin_score(Xs, km.labels_)), 3),
)

# hierarchical (Ward) on a sample for comparison
hs = np.random.RandomState(1).choice(len(Xs), 6000, replace=False)
hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Xs[hs])
out["ward_silhouette"] = round(float(silhouette_score(Xs[hs], hc.labels_)), 3)
out["kmeans_silhouette_same_sample"] = round(float(silhouette_score(Xs[hs], km.labels_[hs])), 3)
out["ari_ward_vs_kmeans"] = round(float(adjusted_rand_score(hc.labels_, km.labels_[hs])), 3)
Z = linkage(Xs[hs[:800]], method="ward")
fig, ax = plt.subplots(figsize=(9, 3.8))
dendrogram(Z, truncate_mode="lastp", p=24, no_labels=True, ax=ax, color_threshold=0.7 * Z[-1, 2])
ax.set_title("Ward Dendrogram (800-booking sample, last 24 merges)"); ax.set_ylabel("Distance")
plt.tight_layout(); plt.savefig("charts/seg_dendrogram.png"); plt.close()

# ------------------------------------------------ 4. Profiling
sizes = df.cluster.value_counts().sort_index()
out["sizes"] = sizes.to_dict()
prof_cols = ["lead_time", "total_nights", "total_guests", "adr", "total_of_special_requests",
             "booking_changed", "has_children", "has_parking", "is_repeat", "prior_cancel", "weekend_share", "is_canceled"]
prof = df.groupby("cluster")[prof_cols].mean()
prof_med = df.groupby("cluster")[["lead_time", "adr", "total_nights"]].median()
out["profile"] = prof.round(3).to_dict("index")
out["profile_median"] = prof_med.round(1).to_dict("index")
out["overall"] = df[prof_cols].mean().round(3).to_dict()
out["hotel_mix"] = (df.groupby("cluster")["hotel"].apply(lambda s: (s == "City Hotel").mean()) * 100).round(1).to_dict()
out["segment_mix"] = {int(c): (g.market_segment.value_counts(normalize=True).head(3) * 100).round(1).to_dict() for c, g in df.groupby("cluster")}
out["deposit_mix"] = {int(c): (g.deposit_type.value_counts(normalize=True) * 100).round(1).to_dict() for c, g in df.groupby("cluster")}
out["custtype_mix"] = {int(c): (g.customer_type.value_counts(normalize=True).head(2) * 100).round(1).to_dict() for c, g in df.groupby("cluster")}
out["meal_mix"] = {int(c): (g.meal.value_counts(normalize=True).head(2) * 100).round(1).to_dict() for c, g in df.groupby("cluster")}
out["top_country"] = {int(c): (g.country.value_counts(normalize=True).head(3) * 100).round(1).to_dict() for c, g in df.groupby("cluster")}
# revenue proxy on non-cancelled stays
ok = df[df.is_canceled == 0].copy()
ok["revenue"] = ok.adr * ok.total_nights
rev = ok.groupby("cluster")["revenue"].sum()
out["revenue_share"] = (rev / rev.sum() * 100).round(1).to_dict()
out["revenue_per_booking"] = ok.groupby("cluster")["revenue"].mean().round(1).to_dict()
out["cancel_rate"] = (df.groupby("cluster")["is_canceled"].mean() * 100).round(1).to_dict()
df[["cluster"]].to_csv("cluster_labels.csv")
df.to_csv("hotels_segmented.csv", index=False)

# ------------------------------------------------ 5. Charts
pca = PCA(2, random_state=RS).fit(Xs)
P = pca.transform(Xs)
out["pca_var"] = (pca.explained_variance_ratio_ * 100).round(1).tolist()
ps = np.random.RandomState(3).choice(len(P), 9000, replace=False)
fig, ax = plt.subplots(figsize=(7, 5.2))
for c in range(K):
    m = df.cluster.values[ps] == c
    ax.scatter(P[ps][m, 0], P[ps][m, 1], s=6, alpha=0.45, color=PAL[c], label=f"Segment {c}")
ax.set_xlabel(f"PC1 ({out['pca_var'][0]}%)"); ax.set_ylabel(f"PC2 ({out['pca_var'][1]}%)")
ax.set_title("Segments in PCA Space (9,000-booking sample)"); ax.legend(markerscale=3)
plt.tight_layout(); plt.savefig("charts/seg_pca.png"); plt.close()

z = pd.DataFrame(Xs, columns=FEATS); z["cluster"] = df.cluster.values
zc = z.groupby("cluster").mean()
fig, ax = plt.subplots(figsize=(9, 3.8 + 0.2 * K))
sns.heatmap(zc, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Segment Profiles (standardised feature means; log scale for lead time and nights)")
plt.tight_layout(); plt.savefig("charts/seg_heatmap.png"); plt.close()

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.bar([f"Seg {c}" for c in sizes.index], sizes.values / sizes.sum() * 100, color=PAL[:K])
for i, v in enumerate(sizes.values / sizes.sum() * 100): ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
ax.set_ylabel("% of bookings"); ax.set_title("Segment Sizes")
plt.tight_layout(); plt.savefig("charts/seg_sizes.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
sns.boxplot(data=df, x="cluster", y="adr", ax=ax[0], palette=PAL[:K], showfliers=False); ax[0].set_title("ADR")
sns.boxplot(data=df, x="cluster", y="lead_time", ax=ax[1], palette=PAL[:K], showfliers=False); ax[1].set_title("Lead time (days)")
sns.boxplot(data=df, x="cluster", y="total_nights", ax=ax[2], palette=PAL[:K], showfliers=False); ax[2].set_title("Total nights")
plt.tight_layout(); plt.savefig("charts/seg_boxes.png"); plt.close()

vb = pd.DataFrame({"Cancellation rate %": out["cancel_rate"], "Share of revenue % (non-cancelled)": out["revenue_share"],
                   "Share of bookings %": (sizes / sizes.sum() * 100).round(1).to_dict()})
ax = vb.plot(kind="bar", figsize=(8, 3.8), color=[PAL[1], PAL[2], PAL[0]])
ax.set_xlabel("Segment"); ax.set_title("Business Value by Segment"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("charts/seg_value.png"); plt.close()

json.dump(out, open("segment_stats.json", "w"), indent=2, default=str)
print(json.dumps({k: out[k] for k in ["k_table", "stability_ari", "K", "final_metrics", "ward_silhouette", "kmeans_silhouette_same_sample", "ari_ward_vs_kmeans", "sizes", "cancel_rate", "revenue_share", "pca_var", "skew_before", "skew_after"]}, indent=1, default=str))
print(pd.concat([prof.round(2), prof_med.add_prefix("med_")], axis=1).T)

# ------------------------------------------------ 6. Drill-down of the dominant segment
big = int(sizes.idxmax())
m = df[df.cluster == big].copy()
DF = ["lead_time", "total_nights", "total_guests", "adr", "total_of_special_requests", "booking_changed", "weekend_share"]
Xd = m[DF].copy()
Xd["lead_time"] = np.log1p(Xd.lead_time); Xd["total_nights"] = np.log1p(Xd.total_nights)
Xd["adr"] = Xd.adr.clip(upper=Xd.adr.quantile(0.995))
Xds = StandardScaler().fit_transform(Xd)
ds = np.random.RandomState(0).choice(len(Xds), 12000, replace=False)
out["drill_sil"] = {k: round(float(silhouette_score(Xds[ds], KMeans(k, n_init=10, random_state=RS).fit(Xds).labels_[ds])), 3) for k in [2, 3, 4]}
m["sub"] = KMeans(3, n_init=20, random_state=RS).fit(Xds).labels_
m["rev"] = np.where(m.is_canceled == 0, m.adr * m.total_nights, 0)
tot_rev = float(np.where(df.is_canceled == 0, df.adr * df.total_nights, 0).sum())
g = m.groupby("sub").agg(n=("adr", "size"), lead=("lead_time", "median"), nights=("total_nights", "median"),
                         guests=("total_guests", "mean"), adr=("adr", "median"), req=("total_of_special_requests", "mean"),
                         cancel=("is_canceled", "mean"), rev=("rev", "sum"))
g["share_of_seg"] = g.n / g.n.sum() * 100
g["rev_share_total"] = g.rev / tot_rev * 100
g["city"] = m.groupby("sub").hotel.apply(lambda s: (s == "City Hotel").mean() * 100)
g["online_ta"] = m.groupby("sub").market_segment.apply(lambda s: (s == "Online TA").mean() * 100)
g["share_all_bookings"] = g.n / len(df) * 100
g["cancel"] = g.cancel * 100
out["drill"] = g.drop(columns="rev").round(1).to_dict("index")
out["big_cluster"] = big
json.dump(out, open("segment_stats.json", "w"), indent=2, default=str)
print(g.round(1).T)
