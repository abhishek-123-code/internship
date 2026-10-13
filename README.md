# internship
# Hospitality Data Science Analysis with Python

A six-week data science project on hotel booking and review data. It covers the full analytics lifecycle: data cleaning, time series forecasting, customer segmentation, sentiment analysis, and revenue management. A final synthesis checks whether the independent analyses agree with each other.

> Built as part of an industrial training programme at [Organisation Name] | B.Tech CSE (Data Science), ABES Engineering College

## Project Overview

| Module | Question | Methods |
|---|---|---|
| 1. Cleaning & EDA | What does the raw data look like, and can it be trusted? | pandas audit, outlier handling, correlation analysis |
| 2. Forecasting | How many guests will arrive each day? | Seasonal naive, Holt-Winters, SARIMA, SARIMAX with Fourier terms, rolling-origin validation |
| 3. Segmentation | Which types of guests do we serve? | K-means, Ward hierarchical clustering, silhouette / Davies-Bouldin / stability (ARI) |
| 4. Sentiment Analysis | How do guests feel, and about what? | VADER, TextBlob, clause-level scoring, aspect-based sentiment, TF-IDF + logistic regression, LDA |
| 5. Revenue Management | What drives price, and what should we charge? | Hedonic OLS regression, price-elasticity estimation, RevPAR scenarios |
| 6. Synthesis | Do the modules agree with each other? | Cross-module checks (segment mix vs. seasonality vs. price) |

## Key Findings

- **Data quality:** 119,390 raw bookings were reduced to 87,221 after removing duplicates, invalid records and outliers. Overall cancellation rate is 27.5%.
- **Forecasting:** All models beat the seasonal-naive benchmark. The best model differs between the single holdout window and rolling validation, so single-window evaluation is unreliable on this data.
- **Segmentation:** Three segments (Families, Mainstream Leisure, Loyal Short-Stay Regulars). K-means and Ward clustering agree at ARI 0.95. Silhouette is 0.36, which is modest separation.
- **Sentiment:** Default lexicon tools label about 92% of reviews positive, against 74% by star rating. A TF-IDF + logistic regression model reached macro-F1 0.71, versus 0.48 for standard VADER.
- **Pricing:** August ADR is about 63% above January after controlling for booking characteristics. The naive price-elasticity regression gives an implausible *positive* coefficient. This is diagnosed as reverse causality (hotels raise prices when they expect high demand), so it was not used for pricing decisions.
- **Synthesis:** The Families segment is 10.4% of bookings overall but 18.3% in August, which explains the summer peak found separately by forecasting and pricing. Two independent analyses also land on a ~34.5% cancellation rate for top-quintile prices.

## Repository Structure

```
.
├── scripts/
│   ├── 01_data_cleaning.py
│   ├── 02_eda_and_charts.py
│   ├── 03_time_series_forecasting.py
│   ├── 04_customer_segmentation.py
│   ├── 05_sentiment_analysis.py
│   ├── 06_revenue_pricing_analytics.py
│   └── 07_synthesis_analysis.py
├── data/            # raw + processed CSVs (see Data Sources)
├── charts/          # generated figures
├── reports/         # weekly .docx reports + final capstone report
└── README.md
```

## Tech Stack

Python 3 · pandas · NumPy · statsmodels · scikit-learn · NLTK · TextBlob · WordCloud · SciPy · matplotlib · seaborn

## Setup

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn nltk textblob wordcloud
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4','vader_lexicon','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
mkdir -p charts
```

## Data Sources

| Dataset | Used in | Source |
|---|---|---|
| Hotel Booking Demand (119,390 bookings, 2015-2017) | Modules 1, 2, 3, 5, 6 | Antonio, Almeida & Nunes (2019), *Data in Brief*; TidyTuesday mirror: `https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv` |
| TripAdvisor Hotel Reviews (20,491 reviews) | Module 4 | Public CSV from the `MainakRepositor/Datasets` GitHub repository |

Download both CSVs into the working directory as `hotels.csv` and `reviews.csv`. Check each dataset's original licence before redistributing it.

## How to Run

Run the scripts in order from the folder that contains the CSVs and a `charts/` directory:

```bash
python 01_data_cleaning.py              # hotels.csv -> hotels_clean.csv
python 02_eda_and_charts.py
python 03_time_series_forecasting.py
echo '{"k": 3}' > k_choice.json          # k=3 chosen from validity metrics (default in the script is 4)
python 04_customer_segmentation.py       # -> hotels_segmented.csv
python 05_sentiment_analysis.py          # reads reviews.csv
python 06_revenue_pricing_analytics.py
python 07_synthesis_analysis.py          # needs hotels_segmented.csv
```

## Limitations

- **Occupancy and RevPAR are proxies.** The dataset has no room-capacity field, so capacity was assumed from observed daily arrivals.
- **Price elasticity was not causally identified.** Historical price and demand are jointly determined by the hotel's own pricing system. Scenario analysis uses assumed elasticities and is illustrative only.
- **The two datasets are not linked.** Sentiment findings describe hotel reviews in general, not the two hotels in the booking data.
- **Segments are booking-level, not guest-level.** There is no guest ID.
- **Review labels are star ratings,** not human-annotated sentiment.
- **Scope is limited:** two Portuguese hotels, 2015-2017.

## References

- Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41-49.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM-14*.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

## Author

**Abhi** · B.Tech CSE (Data Science), ABES Engineering College  
[LinkedIn / email]
