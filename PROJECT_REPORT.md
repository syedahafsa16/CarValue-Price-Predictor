# CarValue — Project Report
## Used Car Price Prediction Web Application

**Author:** Syeda Hafsa
**Date:** 2026
**Domain:** Machine Learning · Web Application Development · Data Science  
**Tech Stack:** Python · scikit-learn · Streamlit · pandas · joblib

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Description](#3-dataset-description)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Selection & Training](#7-model-selection--training)
8. [Model Evaluation](#8-model-evaluation)
9. [Application Architecture](#9-application-architecture)
10. [UI/UX Design Decisions](#10-uiux-design-decisions)
11. [Buyer Mode Deal Analysis Logic](#11-buyer-mode-deal-analysis-logic)
12. [Limitations](#12-limitations)
13. [Future Work](#13-future-work)
14. [Conclusion](#14-conclusion)

---

## 1. Executive Summary

**CarValue** is an end-to-end machine learning web application designed to address a practical, high-friction problem in the Indian used-car market: pricing uncertainty. Sellers frequently under-price or over-price their vehicles due to a lack of objective market data, while buyers struggle to assess whether a seller's asking price represents fair value.

This project delivers a complete solution — from raw data ingestion and model training to a polished, dual-mode Streamlit interface — that provides instant, data-driven price estimates. The final model is a Gradient Boosting Regressor trained on 1,380 real used-car transactions, achieving an **R² of 0.66** and a **Mean Absolute Error of Rs. 5.03 L** on a held-out test set.

---

## 2. Problem Statement

### Background

The Indian used-car market is large, fragmented, and largely opaque. Prices for the same vehicle can vary by 20–40% across listings depending on the platform, city, and the negotiating position of each party. This asymmetry of information disadvantages both sellers (who may leave money on the table) and buyers (who may overpay).

### Goal

Build a predictive model and user-facing web application that:
- Accepts structured vehicle details as input
- Predicts a fair market price based on historical transaction data
- Presents the prediction in a way that is immediately actionable for both sellers and buyers
- In Buyer Mode, compares the model's estimate against a seller's asking price and communicates the result objectively and respectfully

### Success Criteria

| Criterion | Target |
|---|---|
| Prediction accuracy (R²) | ≥ 0.60 |
| Mean Absolute Error | ≤ Rs. 6 L |
| UI usability | Single-page, no instructions needed |
| Dual-mode support | Seller and Buyer modes both functional |

All success criteria were met.

---

## 3. Dataset Description

### Source

The dataset consists of used-car listings scraped from Indian automotive platforms and consolidated into a single file (`UsedCars_Combined.csv`) from eight city-level files.

### Schema

| Column | Data Type | Description |
|---|---|---|
| `Id` | Integer | Row identifier (excluded from modelling) |
| `year` | Integer | Manufacturing year of the vehicle |
| `brand` | String | Car manufacturer (31 unique brands) |
| `full_model_name` | String | Complete model descriptor (excluded — too high cardinality) |
| `model_name` | String | Abbreviated model name (excluded — too high cardinality) |
| `price` | Float | Transaction price in INR (**target variable**) |
| `distance_travelled(kms)` | Float | Odometer reading in kilometres |
| `fuel_type` | String | Petrol, Diesel, or alternative fuel |
| `city` | String | City where the vehicle was listed |

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total rows | 1,725 |
| Missing values | None (after loading) |
| Price — minimum | Rs. 62,500 |
| Price — maximum | Rs. 1.47 Cr |
| Price — mean | Rs. 14.95 L |
| Price — median | Rs. 8.75 L |
| Price — std. dev. | Rs. 16.72 L |
| Distance — mean | 53,848 km |
| Distance — maximum | 7,90,000 km |
| Year range | 1990 – 2021 |
| Number of brands | 31 |
| Number of cities | 15 |

---

## 4. Exploratory Data Analysis

### Price Distribution

The price distribution is heavily right-skewed, with the majority of vehicles falling in the Rs. 3 L – Rs. 20 L range and a long tail extending to Rs. 1.47 Cr (ultra-luxury vehicles). This skew is expected in any used-car market and motivated the use of a tree-based model rather than a linear one, as tree ensembles handle skewed targets without requiring a log-transform.

**Key price observations:**
- 25th percentile: Rs. 5.45 L
- 50th percentile: Rs. 8.75 L
- 75th percentile: Rs. 18.25 L
- Top 1% threshold: above Rs. 80 L (primarily BMWs, Mercedes-Benz, Audi, Jaguar, Land Rover)

### Brand Analysis

Hyundai (297 records), Maruti Suzuki (275), and Honda (153) dominate the dataset as volume brands, while Mercedes-Benz (131), BMW (111), and Audi (98) have significant representation at the premium end. This distribution reflects the actual market composition in major Indian cities.

### Fuel Type Distribution

| Fuel Type | Count | % |
|---|---|---|
| Diesel | 922 | 53.4% |
| Petrol | 788 | 45.7% |
| CNG + 1 / Petrol + 1 / Hybrid | 15 | 0.9% |

Diesel vehicles command a notable price premium at higher mileages due to better fuel economy — a pattern the model captures through the joint interaction of `fuel_type` and `distance_travelled`.

### Year / Age Analysis

Vehicles range from 1990 to 2021. The majority were manufactured between 2012 and 2019. Price declines sharply in the first 3–5 years (depreciation curve) and more gradually thereafter, making `car_age` a strong predictor.

### Distance Correlation

Higher odometer readings generally correlate with lower prices, but the relationship is non-linear — the first 50,000 km produce a sharper price drop than subsequent kilometres. Gradient Boosting naturally captures this non-linearity without requiring feature transformation.

---

## 5. Data Preprocessing

### Steps Applied in `train_model.py`

**1. Fuel type normalisation**

The five fuel categories were simplified to three:

```
"Petrol + 1"  →  "CNG/Hybrid"
"CNG + 1"     →  "CNG/Hybrid"
"Hybrid"      →  "CNG/Hybrid"
```

This prevents rare categories from creating spurious, noisy splits in the tree.

**2. Invalid row removal**

- Rows with `price ≤ 0` removed (no negative or zero prices are valid)
- Rows with negative `distance_travelled` removed

**3. Distance outlier clipping**

The top 1% of distance values (above ~2.5 lakh km) were clipped to their 99th-percentile value. This prevents extreme outlier vehicles from pulling the model's distance-response in an unrepresentative direction without discarding the rows entirely.

**4. Car age derivation**

```python
car_age = REFERENCE_YEAR (2022) - year
car_age = max(0, car_age)
```

`car_age` is used instead of raw `year` because it directly represents depreciation time and is more interpretable to the model.

---

## 6. Feature Engineering

### Selected Features

| Feature | Rationale |
|---|---|
| `car_age` | Primary driver of depreciation; more stable than calendar year |
| `distance_travelled(kms)` | Second strongest price signal after age |
| `brand` | Strong price signal; brand perception and segment (hatchback vs. luxury) |
| `fuel_type` | Diesel commands premium at volume; Hybrid/CNG affects segment positioning |
| `city` | Regional demand and supply differences affect price |

### Deliberately Excluded Features

| Feature | Reason for Exclusion |
|---|---|
| `Id` | Row identifier — no predictive value |
| `full_model_name` | Extremely high cardinality (hundreds of unique strings); would cause severe overfitting without extensive encoding |
| `model_name` | Similar cardinality issue; also inconsistently abbreviated across records |
| `year` (raw) | Replaced by the more semantically useful `car_age` |

---

## 7. Model Selection & Training

### Algorithm Choice

**Gradient Boosting Regressor** was selected after considering the following alternatives:

| Algorithm | Pros | Cons | Verdict |
|---|---|---|---|
| Linear Regression | Interpretable, fast | Cannot capture non-linear relationships, sensitive to outliers | Rejected |
| Random Forest | Robust, handles mixed types | Slower inference, slightly lower accuracy than GB on this dataset | Runner-up |
| **Gradient Boosting** | **Strong accuracy on tabular data, handles mixed features, robust to outliers** | **Slower training than RF** | **Selected** |
| XGBoost / LightGBM | State-of-the-art performance | Additional dependency, diminishing returns on a 1,700-row dataset | Out of scope |

### Scikit-learn Pipeline

A `Pipeline` with a `ColumnTransformer` ensures that preprocessing steps are applied consistently at both training and inference time, preventing data leakage:

```
Pipeline
└── ColumnTransformer (preprocessor)
    ├── StandardScaler          → [car_age, distance_travelled(kms)]
    └── OrdinalEncoder          → [brand, fuel_type, city]
            handle_unknown='use_encoded_value', unknown_value=-1
└── GradientBoostingRegressor (regressor)
```

Using `OrdinalEncoder` rather than `OneHotEncoder` was a deliberate choice: with 31 brands and 15 cities, one-hot encoding would produce 50+ sparse dimensions on a 1,380-sample training set, risking overfitting. Ordinal encoding with an unknown-handling fallback is more appropriate at this dataset size.

### Train / Test Split

```python
train_test_split(X, y, test_size=0.20, random_state=42)
# 1,380 training samples | 345 test samples
```

### Hyperparameters

The hyperparameters were selected through manual tuning with a focus on preventing overfitting on a relatively small dataset:

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 400 | Sufficient trees to converge; higher values yielded no improvement |
| `learning_rate` | 0.08 | Lower rate with more estimators → smoother regularisation |
| `max_depth` | 5 | Balanced expressiveness without deep overfitting |
| `min_samples_leaf` | 5 | Prevents splits on very small groups |
| `subsample` | 0.85 | Stochastic gradient boosting reduces variance |

---

## 8. Model Evaluation

### Metrics on Held-Out Test Set (n = 345)

| Metric | Value | Interpretation |
|---|---|---|
| **R²** | **0.6601** | Model explains 66% of price variance |
| **MAE** | **Rs. 5,03,075** | Average absolute prediction error is Rs. ~5 L |

### Contextualising the R² Score

An R² of 0.66 is reasonable and practically useful for used-car pricing with 5 input features. The remaining 34% of variance is largely attributable to factors the dataset does not capture: vehicle condition, service history, accident records, negotiating context, and cosmetic state. No model with only the 5 available features could close this gap without additional data.

### Contextualising the MAE

Rs. 5.03 L as a mean absolute error is equivalent to roughly 35–57% of the **median** vehicle price (Rs. 8.75 L). For a buyer evaluating a Rs. 8 L car, the model's estimate might vary by ±Rs. 3–7 L from the true fair value. This is why the UI presents a price **range** (±60% of MAE) alongside the point estimate, and explicitly communicates that the output is a benchmark, not a valuation.

For higher-value vehicles (Rs. 25–50 L), a ±Rs. 5 L error is proportionally much smaller and more actionable.

---

## 9. Application Architecture

### Two-File Architecture

The application is split into two concerns:

```
train_model.py   →   data preprocessing + model training + artifact export
app.py           →   artifact loading + UI + real-time inference
```

This separation means:
- The model only needs to be retrained when the data changes
- The app loads the serialised pipeline once at startup (cached via `@st.cache_resource`)
- Inference is a single `pipeline.predict(row)` call — no preprocessing code is duplicated

### Artifact Bundle (`model_artifacts.pkl`)

The `joblib`-serialised artifact file contains:

```python
{
    "pipeline":       <sklearn Pipeline>,   # complete preprocessing + model
    "brands":         [...],                # sorted list for UI dropdown
    "fuel_types":     [...],
    "cities":         [...],
    "year_min":       1990,
    "year_max":       2021,
    "reference_year": 2022,
    "mae":            503075.0,             # surfaced in model info panel
    "r2":             0.6601,
    "n_train":        1380,
    "n_test":         345,
}
```

This bundle ensures the UI dropdowns always reflect exactly the categories the model was trained on, eliminating encoding mismatches.

### Inference Flow

```
User input (year, distance, brand, fuel_type, city)
    │
    ▼
predict_price()
    │  car_age = reference_year - year
    │  DataFrame row constructed
    │
    ▼
pipeline.predict(row)
    │  StandardScaler transforms [car_age, distance]
    │  OrdinalEncoder transforms [brand, fuel_type, city]
    │  GradientBoostingRegressor produces prediction
    │
    ▼
max(0.0, float(price))    ← floor at zero
    │
    ▼
format_inr()              ← converts to Lakh / Crore display notation
```

---

## 10. UI/UX Design Decisions

### Dual-Mode Architecture

Rather than a single form with an optional "comparison" section, the application uses explicit **Seller Mode** and **Buyer Mode** buttons. This framing:
- Immediately communicates the two distinct use cases
- Sets user expectations about what the output means
- Keeps each mode's interface focused — Buyer Mode adds the asking-price input, Seller Mode adds the metric chips; neither clutters the other

### Typography

**Plus Jakarta Sans** (Google Fonts) was chosen for its:
- High x-height and excellent screen readability at small sizes
- Modern geometric character matching the clean, data-forward aesthetic
- Robust weight range (400–800) for clear typographic hierarchy

### Result Card Design

Price results are displayed in large gradient cards with:
- A small ALL-CAPS eyebrow label (e.g. *Estimated Market Selling Price*)
- A 3 rem, 800-weight price figure as the focal point
- A muted subline for range or vehicle details
- Soft box shadows to lift the card off the page

This hierarchy ensures the price is the first thing the eye lands on.

### Metric Chips (Seller Mode)

After the price card, five small chips summarise the input parameters (Brand, Year, Distance, Fuel, Car Age). This serves as a confirmation that the model received the correct inputs, reducing user anxiety about prediction accuracy.

### Optional Photo Upload

The photo upload feature is intentionally scoped to display only. It is placed in a collapsed expander to avoid visual noise for users who do not need it, and its label clearly states it does not affect the prediction. This prevents the UI from implying capabilities the application does not yet have.

---

## 11. Buyer Mode Deal Analysis Logic

### Three-Tier Classification

```python
diff = asking_price - predicted_market_value
pct  = (diff / predicted) * 100

if diff < 0:              # asking below market
    → "Great Value"       / "Below Estimated Market Price"

elif pct <= 8:            # within 8% above market
    → "Aligned with Market" / "Fair Market Value"

else:                     # more than 8% above market
    → "Above Our Market Benchmark" / "Above Estimated Market Price"
```

The 8% fair-value threshold was chosen because it sits comfortably within the model's own MAE (~35% of median price), meaning differences smaller than 8% are likely within prediction uncertainty rather than a genuine over-price.

### Tone & Language Principles

All three tier descriptions were written to be:

1. **Objective** — phrased as comparisons to a market benchmark, not personal judgements about the seller
2. **Contextual** — the "Above Benchmark" tier explicitly acknowledges that low mileage, full service history, or recent upgrades can legitimately justify a higher price
3. **Actionable** — each tier gives the buyer a clear next step (proceed with confidence / inspect before proceeding / use as a negotiation reference point)
4. **Non-accusatory** — words like "overpriced", "bad deal", or "rip-off" are never used

### Comparison Table

Rather than embedding numbers inside prose sentences, the deal card presents a clean three-row table:

```
Our market estimate:      Rs. X
Seller's asking price:    Rs. Y
───────────────────────────────
Difference:               Rs. Z above / below market (N%)
```

This format is scannable at a glance and unambiguous.

---

## 12. Limitations

| Limitation | Impact | Mitigation in Current Version |
|---|---|---|
| No vehicle condition input | Condition can swing price ±20% | Price range (±60% MAE) displayed; disclaimer shown |
| No model-level granularity | "Honda" could mean a City (Rs. 8 L) or an Accord (Rs. 25 L) | User can reference brand + year to narrow segment |
| Dataset size (1,725 rows) | Limits generalisation for rare brands | OrdinalEncoder gracefully handles unseen categories |
| India-specific only | Not valid for other markets | Scope clearly stated in UI and documentation |
| Static dataset (up to 2021) | Market conditions change over time | Reference year fixed at 2022; easily retrained with fresh data |
| No image-based valuation | Uploaded photo does not affect prediction | Clearly labelled as display-only |

---

## 13. Future Work

### Short-Term

- **Vehicle condition input** — A 4-point condition selector (Excellent / Good / Fair / Poor) with an empirical price multiplier derived from the dataset
- **Model-name granularity** — Group `full_model_name` into ~50 logical segments using fuzzy string matching to capture within-brand price variation
- **SHAP explainability** — Display a SHAP waterfall chart per prediction so users understand which features drove the estimate

### Medium-Term

- **Dataset expansion** — Periodic re-scraping of listing platforms to keep training data current
- **Multi-city demand index** — Weight predictions by current city-level supply/demand ratio from live listings
- **Confidence interval visualisation** — Plot a distribution of similar historical transactions to show where the predicted price falls

### Long-Term

- **Image-based condition scoring** — Integrate a fine-tuned vision model to estimate condition (Clean / Minor Damage / Significant Damage) from an uploaded photo and adjust the price prediction accordingly
- **REST API** — Expose the model via a FastAPI endpoint to support third-party integrations (dealer platforms, comparison sites)

---

## 14. Conclusion

CarValue demonstrates a complete, production-style machine learning workflow: from raw data exploration through preprocessing, model training, evaluation, artifact serialisation, and delivery via a polished web interface. The project achieves all defined success criteria and is structured for easy extension.

The most important design decisions were:
1. **Separating training from inference** via a serialised artifact bundle, which keeps the app fast and the code clean
2. **Framing the UI around user roles** (Seller vs. Buyer) rather than model inputs, which dramatically simplifies the user experience
3. **Communicating uncertainty honestly** by displaying price ranges and clear disclaimers rather than presenting a single number as ground truth
4. **Using objective, respectful language** in the deal analysis to position the tool as an informational aid rather than an arbiter of negotiation

This project is immediately deployable to [Streamlit Community Cloud](https://streamlit.io/cloud) at no cost, making it accessible as both a portfolio demonstration and a genuinely useful tool for the Indian used-car market.

---

*Report prepared as part of a data science and machine learning portfolio.*  
*All predictions are indicative market benchmarks and should not be treated as professional vehicle valuations.*
