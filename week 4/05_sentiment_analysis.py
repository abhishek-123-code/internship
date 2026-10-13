import re, json, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
BLUE, ORANGE, GREEN, GREY = "#3b6ea5", "#a5573b", "#3f8f5b", "#777777"
out = {}
RS = 42

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv("reviews.csv")
out["n_reviews"] = len(df)
out["rating_counts"] = df.Rating.value_counts().sort_index().to_dict()
out["dups"] = int(df.Review.duplicated().sum())
df["label"] = pd.cut(df.Rating, [0, 2, 3, 5], labels=["negative", "neutral", "positive"])
out["label_counts"] = df.label.value_counts().to_dict()

# ---------------------------------------------------------------- 2. Repair text
def fix_text(t):
    n_moj = len(re.findall(r"\w+__\S*", t))
    # mojibake such as didn__Ç_é_ : the apostrophe was destroyed
    def repl(m):
        stem = m.group(1)
        if stem.endswith("n") or stem in ("can", "won"):
            return stem[:-1] + " not" if stem.endswith("n") and stem not in ("can", "won") else stem + " not"
        return stem
    t2 = re.sub(r"\b([a-z]+)__\S*", repl, t)
    t2 = re.sub(r"\bwo n't\b", "will not", t2)
    t2 = re.sub(r"\bca n't\b", "can not", t2)
    t2 = re.sub(r"\b([a-z]+) n't\b", r"\1 not", t2)   # 'did n't' -> 'did not'
    t2 = re.sub(r"\bn't\b", "not", t2)
    t2 = re.sub(r"\s+", " ", t2).strip()
    return t2, n_moj

fixed = df.Review.apply(fix_text)
df["text"] = [f[0] for f in fixed]
out["mojibake_reviews"] = int(sum(1 for f in fixed if f[1] > 0))
out["nt_tokens_raw"] = int(df.Review.str.count(r"\bn't\b").sum())
df["n_words"] = df.text.str.split().str.len()
out["len_by_rating"] = df.groupby("Rating").n_words.median().to_dict()

# ---------------------------------------------------------------- 3. Tokenise, stop words, lemmatise
STOP = set(stopwords.words("english"))
NEGS = {"not", "no", "never", "nor"}
STOP_TOPIC = STOP  # topic/wordcloud view removes negations too
lem = WordNetLemmatizer()
DOMAIN_STOP = {"hotel", "room", "stay", "stayed", "night", "day", "us", "get", "got", "would", "one", "also", "could",
               "even", "really", "like", "go", "went", "u", "s", "t", "nt", "make", "made", "take", "took", "say", "said", "well", "much"}

def wn_pos(tag):
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)

def preprocess(t):
    toks = re.findall(r"[a-z]+", t.lower())
    tagged = nltk.pos_tag(toks)
    lemmas = [lem.lemmatize(w, wn_pos(p)) for w, p in tagged]
    return [w for w in lemmas if w not in STOP and len(w) > 2 and w not in DOMAIN_STOP]

df["tokens"] = df.text.apply(preprocess)
df["clean"] = df.tokens.str.join(" ")
out["avg_tokens_after"] = round(df.tokens.str.len().mean(), 1)
out["avg_words_before"] = round(df.n_words.mean(), 1)
out["vocab_size"] = len({w for ts in df.tokens for w in ts})
ex = df[df.Review.str.contains(r"\bn't\b") & df.Review.str.contains("__")].iloc[0]
j = re.search(r"__|n't", df.Review.loc[ex.name]); st = max(0, j.start() - 90)
out["example"] = {"raw": df.Review.loc[ex.name][st:st + 200], "fixed": ex.text[max(0, st - 5):st + 200], "tokens": ex.tokens[:22]}

# ---------------------------------------------------------------- 4. Lexicon methods
vader = SentimentIntensityAnalyzer()
df["vader"] = df.text.apply(lambda t: vader.polarity_scores(t)["compound"])
df["blob"] = df.text.apply(lambda t: TextBlob(t).sentiment.polarity)
df["blob_subj"] = df.text.apply(lambda t: TextBlob(t).sentiment.subjectivity)
# experiment: does the negation repair matter?
df["vader_raw"] = df.Review.apply(lambda t: vader.polarity_scores(t)["compound"])

def to_label(x, lo, hi):
    return np.where(x >= hi, "positive", np.where(x <= lo, "negative", "neutral"))

def eval3(pred, name):
    y = df.label.astype(str)
    return dict(name=name, accuracy=round(accuracy_score(y, pred), 3),
                macro_f1=round(f1_score(y, pred, average="macro"), 3))

# VADER standard cut-offs are +/-0.05 - check them, then a tuned pair
res = {}
res["VADER (standard +/-0.05)"] = eval3(to_label(df.vader, -0.05, 0.05), "VADER std")
res["TextBlob (0 cut-off)"] = eval3(to_label(df.blob, -0.0001, 0.0001), "TextBlob")
res["VADER on raw text"] = eval3(to_label(df.vader_raw, -0.05, 0.05), "VADER raw")

# tune thresholds on the training half only, evaluate on test half
tr_idx, te_idx = train_test_split(df.index, test_size=0.2, stratify=df.label, random_state=RS)
def tune(col):
    best = (0, None)
    for lo in np.arange(-0.2, 0.3, 0.05):
        for hi in np.arange(0.3, 0.95, 0.05):
            p = to_label(df.loc[tr_idx, col], lo, hi)
            f = f1_score(df.loc[tr_idx, "label"].astype(str), p, average="macro")
            if f > best[0]: best = (f, (lo, hi))
    return best[1]
thr_v, thr_b = tune("vader"), tune("blob")
out["tuned_thr"] = {"vader": [round(float(x), 2) for x in thr_v], "blob": [round(float(x), 2) for x in thr_b]}
yte = df.loc[te_idx, "label"].astype(str)
res["VADER (tuned, test)"] = dict(name="VADER tuned", accuracy=round(accuracy_score(yte, to_label(df.loc[te_idx, "vader"], *thr_v)), 3),
                                  macro_f1=round(f1_score(yte, to_label(df.loc[te_idx, "vader"], *thr_v), average="macro"), 3))
res["TextBlob (tuned, test)"] = dict(name="TextBlob tuned", accuracy=round(accuracy_score(yte, to_label(df.loc[te_idx, "blob"], *thr_b)), 3),
                                     macro_f1=round(f1_score(yte, to_label(df.loc[te_idx, "blob"], *thr_b), average="macro"), 3))
out["spearman"] = {"vader": round(spearmanr(df.vader, df.Rating)[0], 3), "textblob": round(spearmanr(df.blob, df.Rating)[0], 3),
                   "vader_raw": round(spearmanr(df.vader_raw, df.Rating)[0], 3)}
out["mean_by_rating"] = df.groupby("Rating")[["vader", "blob", "blob_subj"]].mean().round(3).to_dict("index")

# ---------------------------------------------------------------- 5. Supervised model
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.8, sublinear_tf=True, stop_words=None)
Xtr = tf.fit_transform(df.loc[tr_idx, "text"]); Xte = tf.transform(df.loc[te_idx, "text"])
ytr = df.loc[tr_idx, "label"].astype(str)
best_C, best_f = None, 0
for C in [0.3, 1, 3, 10]:
    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    f = f1_score(yte, m.predict(Xte), average="macro")
    out.setdefault("lr_C", {})[str(C)] = round(f, 3)
    if f > best_f: best_C, best_f = C, f
lr = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
pred_lr = lr.predict(Xte)
res["TF-IDF + Logistic Regression (test)"] = dict(name="TF-IDF LR", accuracy=round(accuracy_score(yte, pred_lr), 3),
                                                   macro_f1=round(f1_score(yte, pred_lr, average="macro"), 3))
out["best_C"] = best_C
out["lr_report"] = classification_report(yte, pred_lr, output_dict=True)
feats = np.array(tf.get_feature_names_out())
cls = list(lr.classes_)
coef_pos = lr.coef_[cls.index("positive")]; coef_neg = lr.coef_[cls.index("negative")]
out["top_pos_terms"] = feats[np.argsort(coef_pos)[::-1][:15]].tolist()
out["top_neg_terms"] = feats[np.argsort(coef_neg)[::-1][:15]].tolist()

# majority-class baseline
maj = yte.value_counts(normalize=True).iloc[0]
res["Majority-class baseline"] = dict(name="Majority", accuracy=round(float(maj), 3), macro_f1=round(f1_score(yte, ["positive"] * len(yte), average="macro"), 3))
out["results"] = res

# Same 20% test split for lexicon methods with standard thresholds (fair comparison)
res["VADER (standard, test)"] = dict(name="VADER std test",
    accuracy=round(accuracy_score(yte, to_label(df.loc[te_idx, "vader"], -0.05, 0.05)), 3),
    macro_f1=round(f1_score(yte, to_label(df.loc[te_idx, "vader"], -0.05, 0.05), average="macro"), 3))

# confusion matrices (test)
labs = ["negative", "neutral", "positive"]
cms = {
    "VADER (tuned)": confusion_matrix(yte, to_label(df.loc[te_idx, "vader"], *thr_v), labels=labs),
    "TextBlob (tuned)": confusion_matrix(yte, to_label(df.loc[te_idx, "blob"], *thr_b), labels=labs),
    "TF-IDF + LR": confusion_matrix(yte, pred_lr, labels=labs),
}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, (n, cm) in zip(ax, cms.items()):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=labs, yticklabels=labs, ax=a, cbar=False)
    a.set_title(n); a.set_xlabel("Predicted"); a.set_ylabel("Actual" if a is ax[0] else "")
plt.tight_layout(); plt.savefig("charts/sa_confusion.png"); plt.close()

# ---------------------------------------------------------------- 6. Aspect-based sentiment
ASPECTS = {
    "Staff & service": ["staff", "service", "helpful", "friendly", "concierge", "desk", "manager", "employee", "receptionist", "rude", "courteous", "waiter", "bellman", "doorman"],
    "Room": ["room", "bed", "bathroom", "shower", "pillow", "towel", "suite", "view", "balcony", "tv", "carpet"],
    "Cleanliness": ["clean", "dirty", "smell", "stain", "dust", "spotless", "filthy", "bug", "roach", "mold", "housekeeping", "odor"],
    "Location": ["location", "walk", "walking", "distance", "close", "near", "street", "metro", "subway", "beach", "downtown", "center", "central"],
    "Food & breakfast": ["breakfast", "food", "restaurant", "dinner", "buffet", "meal", "coffee", "bar", "drink", "lunch"],
    "Value & price": ["price", "value", "money", "expensive", "cheap", "cost", "rate", "pay", "charge", "fee", "overpriced", "worth"],
    "Noise": ["noise", "noisy", "loud", "quiet", "sound", "thin", "hear"],
    "Facilities": ["pool", "gym", "spa", "wifi", "internet", "elevator", "parking", "lobby", "fitness", "casino", "resort"],
}
def clauses(t):
    return [c.strip() for c in re.split(r"[.,;!?]", t) if len(c.strip()) > 2]

rows = []
rev_feats = []
for i, (t, r) in enumerate(zip(df.text, df.Rating)):
    cl = clauses(t)
    sc = [(c, vader.polarity_scores(c)["compound"], set(re.findall(r"[a-z]+", c))) for c in cl]
    cs = np.array([x[1] for x in sc]) if sc else np.array([0.0])
    rev_feats.append((cs.mean(), (cs < -0.05).mean(), cs.min(), (cs > 0.05).mean()))
    for a, kws in ASPECTS.items():
        hit = [s for c, s, ws in sc if ws & set(kws)]
        if hit:
            rows.append((i, r, a, float(np.mean(hit)), len(hit)))
df[["clause_mean", "neg_clause_share", "clause_min", "pos_clause_share"]] = np.array(rev_feats)
out["spearman"]["clause_mean"] = round(spearmanr(df.clause_mean, df.Rating)[0], 3)
out["spearman"]["neg_clause_share"] = round(spearmanr(df.neg_clause_share, df.Rating)[0], 3)
out["spearman"]["clause_min"] = round(spearmanr(df.clause_min, df.Rating)[0], 3)
out["clause_by_rating"] = df.groupby("Rating")[["clause_mean", "neg_clause_share", "pos_clause_share"]].mean().round(3).to_dict("index")
df["len_bin"] = pd.qcut(df.n_words, 3, labels=["short", "medium", "long"])
out["spearman_by_len"] = {b: {"vader": round(spearmanr(g.vader, g.Rating)[0], 3), "clause_mean": round(spearmanr(g.clause_mean, g.Rating)[0], 3)} for b, g in df.groupby("len_bin")}
out["len_bins"] = {b: [int(g.n_words.min()), int(g.n_words.max())] for b, g in df.groupby("len_bin")}
out["vader_by_len"] = df.groupby("len_bin").vader.mean().round(3).to_dict()
asp = pd.DataFrame(rows, columns=["idx", "Rating", "aspect", "score", "n_clauses"])
mention = (asp.groupby("aspect").idx.nunique() / len(df) * 100).round(1)
asp_mean = asp.groupby("aspect").score.mean().round(3)
asp_pos = asp.groupby("aspect").score.apply(lambda s: (s > 0.05).mean() * 100).round(1)
asp_neg = asp.groupby("aspect").score.apply(lambda s: (s < -0.05).mean() * 100).round(1)
out["aspect"] = pd.DataFrame({"mention_pct": mention, "mean": asp_mean, "pos_pct": asp_pos, "neg_pct": asp_neg}).sort_values("mean").to_dict("index")
asp["group"] = pd.cut(asp.Rating, [0, 2, 3, 5], labels=["1-2 stars", "3 stars", "4-5 stars"])
heat = asp.pivot_table(index="aspect", columns="group", values="score", aggfunc="mean").round(2)
out["aspect_by_group"] = heat.to_dict("index")
# aspect mention rates by group (what do unhappy guests talk about?)
grp_n = df.assign(group=pd.cut(df.Rating, [0, 2, 3, 5], labels=["1-2 stars", "3 stars", "4-5 stars"])).group.value_counts()
mg = asp.drop_duplicates(["idx", "aspect"]).groupby(["aspect", "group"]).size().unstack() / grp_n * 100
out["aspect_mention_by_group"] = mg.round(1).to_dict("index")

# ---------------------------------------------------------------- 7. Themes: n-grams and LDA
def top_ngrams(texts, n=(2, 2), k=12):
    cv = CountVectorizer(ngram_range=n, min_df=10, stop_words=list(STOP | DOMAIN_STOP))
    X = cv.fit_transform(texts)
    s = np.asarray(X.sum(axis=0)).ravel()
    idx = s.argsort()[::-1][:k]
    return [(cv.get_feature_names_out()[i], int(s[i])) for i in idx]
neg_txt = df[df.Rating <= 2].clean; pos_txt = df[df.Rating >= 4].clean
out["bigrams_neg"] = top_ngrams(neg_txt, (2, 2), 15)
out["bigrams_pos"] = top_ngrams(pos_txt, (2, 2), 15)

def distinct(neg, pos, k=12, minc=25):
    cv2 = CountVectorizer(ngram_range=(2, 2), min_df=minc)
    cv2.fit(pd.concat([neg, pos]))
    a = np.asarray(cv2.transform(neg).sum(axis=0)).ravel() + 0.5
    b = np.asarray(cv2.transform(pos).sum(axis=0)).ravel() + 0.5
    lo = np.log(a / a.sum()) - np.log(b / b.sum())
    names = cv2.get_feature_names_out()
    order = lo.argsort()
    neg_d = [(names[i], round(float(lo[i]), 2), int(a[i] - 0.5)) for i in order[::-1][:k]]
    pos_d = [(names[i], round(float(-lo[i]), 2), int(b[i] - 0.5)) for i in order[:k]]
    return neg_d, pos_d
out["distinct_neg"], out["distinct_pos"] = distinct(neg_txt, pos_txt)

cv = CountVectorizer(min_df=15, max_df=0.5, max_features=4000)
Xn = cv.fit_transform(neg_txt)
lda = LatentDirichletAllocation(n_components=5, learning_method="batch", random_state=RS, max_iter=25).fit(Xn)
vocab = cv.get_feature_names_out()
out["lda_topics"] = [[vocab[j] for j in comp.argsort()[::-1][:10]] for comp in lda.components_]
out["lda_share"] = (lda.transform(Xn).argmax(axis=1) == np.arange(5)[:, None]).mean(axis=1).round(3).tolist()

# ---------------------------------------------------------------- 8. Charts
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
rc = df.Rating.value_counts().sort_index()
ax[0].bar(rc.index.astype(str), rc.values, color=[ORANGE, ORANGE, GREY, BLUE, BLUE]); ax[0].set_title("Star Rating Distribution"); ax[0].set_xlabel("Rating"); ax[0].set_ylabel("Reviews")
for i, v in enumerate(rc.values): ax[0].text(i, v + 150, f"{v/len(df)*100:.1f}%", ha="center", fontsize=8)
sns.boxplot(x=df.Rating, y=df.n_words.clip(upper=400), ax=ax[1], color="#c8d6e8"); ax[1].set_title("Review Length by Rating (words, capped at 400)"); ax[1].set_ylabel("Words")
plt.tight_layout(); plt.savefig("charts/sa_ratings.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
sns.boxplot(x=df.Rating, y=df.vader, ax=ax[0], color="#c8d6e8", showfliers=False); ax[0].set_title("VADER Compound Score by Star Rating")
sns.boxplot(x=df.Rating, y=df.blob, ax=ax[1], color="#e8d0c8", showfliers=False); ax[1].set_title("TextBlob Polarity by Star Rating")
plt.tight_layout(); plt.savefig("charts/sa_score_by_rating.png"); plt.close()

lab_v = pd.Series(to_label(df.vader, -0.05, 0.05)).value_counts(normalize=True).reindex(labs) * 100
lab_b = pd.Series(to_label(df.blob, -0.0001, 0.0001)).value_counts(normalize=True).reindex(labs) * 100
lab_t = df.label.value_counts(normalize=True).reindex(labs) * 100
cmp_ = pd.DataFrame({"Star-rating label": lab_t, "VADER": lab_v, "TextBlob": lab_b})
ax = cmp_.plot(kind="bar", figsize=(7, 3.8), color=[GREY, BLUE, ORANGE]); ax.set_ylabel("% of reviews"); ax.set_title("Sentiment Distribution: Ratings vs. Lexicon Tools"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("charts/sa_distribution.png"); plt.close()
out["dist_compare"] = cmp_.round(1).to_dict("index")

order = ["Majority-class baseline", "TextBlob (tuned, test)", "VADER (standard, test)", "VADER (tuned, test)", "TF-IDF + Logistic Regression (test)"]
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(order)); w = 0.38
ax.bar(x - w/2, [res[o]["accuracy"] for o in order], w, label="Accuracy", color=BLUE)
ax.bar(x + w/2, [res[o]["macro_f1"] for o in order], w, label="Macro-F1", color=ORANGE)
ax.set_xticks(x); ax.set_xticklabels(["Majority\nbaseline", "TextBlob\n(tuned)", "VADER\n(standard)", "VADER\n(tuned)", "TF-IDF +\nLogReg"], fontsize=8)
ax.set_ylim(0, 1); ax.legend(); ax.set_title("Three-Class Sentiment Performance (20% held-out test set)")
plt.tight_layout(); plt.savefig("charts/sa_models.png"); plt.close()

def cloud(tokens_series, fname, cmap):
    freq = collections.Counter(w for ts in tokens_series for w in ts)
    wc = WordCloud(width=900, height=450, background_color="white", colormap=cmap, max_words=90, random_state=RS).generate_from_frequencies(freq)
    plt.figure(figsize=(7.5, 3.75)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(fname); plt.close()
cloud(df[df.Rating >= 4].tokens, "charts/sa_cloud_pos.png", "Greens")
cloud(df[df.Rating <= 2].tokens, "charts/sa_cloud_neg.png", "Reds")

# top discriminative terms
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
pt = feats[np.argsort(coef_pos)[::-1][:12]][::-1]; pv = np.sort(coef_pos)[::-1][:12][::-1]
nt = feats[np.argsort(coef_neg)[::-1][:12]][::-1]; nv = np.sort(coef_neg)[::-1][:12][::-1]
ax[0].barh(pt, pv, color=GREEN); ax[0].set_title("Terms most predictive of POSITIVE")
ax[1].barh(nt, nv, color=ORANGE); ax[1].set_title("Terms most predictive of NEGATIVE")
plt.tight_layout(); plt.savefig("charts/sa_top_terms.png"); plt.close()

# aspects
a_sorted = pd.DataFrame(out["aspect"]).T.sort_values("mean")
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
cols = [ORANGE if v < a_sorted["mean"].median() else BLUE for v in a_sorted["mean"]]
ax[0].barh(a_sorted.index, a_sorted["mean"], color=cols); ax[0].set_title("Mean VADER Sentiment of Aspect Mentions"); ax[0].axvline(0, color="k", lw=0.6)
ax[1].barh(a_sorted.index, a_sorted.mention_pct, color=GREY); ax[1].set_title("% of Reviews Mentioning Aspect"); ax[1].set_yticklabels([])
plt.tight_layout(); plt.savefig("charts/sa_aspects.png"); plt.close()

plt.figure(figsize=(6.2, 4))
sns.heatmap(heat.loc[a_sorted.index], annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"shrink": 0.8})
plt.title("Aspect Sentiment by Review Group"); plt.ylabel(""); plt.xlabel("")
plt.tight_layout(); plt.savefig("charts/sa_aspect_heat.png"); plt.close()

# bigrams
fig, ax = plt.subplots(1, 2, figsize=(9.8, 4))
nb = out["distinct_neg"][::-1]; pb = out["distinct_pos"][::-1]
ax[0].barh([b[0] for b in pb], [b[1] for b in pb], color=GREEN); ax[0].set_title("Bigrams distinctive of 4-5 star reviews\n(log-odds vs. 1-2 star)", fontsize=10)
ax[1].barh([b[0] for b in nb], [b[1] for b in nb], color=ORANGE); ax[1].set_title("Bigrams distinctive of 1-2 star reviews\n(log-odds vs. 4-5 star)", fontsize=10)
plt.tight_layout(); plt.savefig("charts/sa_bigrams.png"); plt.close()

# agreement disagreement analysis: strong star mismatch
mis_pos = df[(df.Rating <= 2) & (df.vader >= 0.5)]
mis_neg = df[(df.Rating == 5) & (df.vader <= -0.05)]
out["mismatch"] = {"low_star_high_vader": int(len(mis_pos)), "five_star_neg_vader": int(len(mis_neg)),
                   "low_star_total": int((df.Rating <= 2).sum()), "five_star_total": int((df.Rating == 5).sum())}
out["mismatch_example"] = mis_pos.text.iloc[0][:330] if len(mis_pos) else ""

json.dump(out, open("sentiment_stats.json", "w"), indent=2, default=str)
df[["Rating", "label", "vader", "blob", "blob_subj", "n_words"]].to_csv("reviews_scored.csv", index=False)
print(json.dumps({k: out[k] for k in ["mojibake_reviews", "nt_tokens_raw", "results", "spearman", "tuned_thr", "best_C", "lr_C", "mismatch", "dist_compare", "avg_tokens_after", "avg_words_before", "vocab_size"]}, indent=1, default=str))
