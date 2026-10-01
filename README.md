# 🚗 CarValue — Used Car Price Predictor

> An AI-powered web application that estimates fair market prices for used cars in India, built with Python, scikit-learn, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

**CarValue** is a full-stack machine learning application that helps users in the Indian used-car market make informed decisions about pricing. It presents two distinct user experiences:

- **Seller Mode** — Enter your vehicle details and receive an estimated fair market selling price with a realistic price range.
- **Buyer Mode** — Enter the details of a car you are considering purchasing. Optionally provide the seller's asking price to receive an objective market comparison that helps guide negotiation.

The prediction model is trained on 1,725 real used-car transactions from eight major Indian cities and achieves an **R² of 0.66** and a **Mean Absolute Error of Rs. 5.03 L** on the held-out test set.

---

## Features

| Feature | Description |
|---|---|
| **Seller Mode** | Predicts fair market selling price + typical high/low range |
| **Buyer Mode** | Predicts market value; optional asking-price comparison |
| **Deal Analysis** | Polite, objective three-tier verdict (Great Value / Aligned with Market / Above Benchmark) |
| **Vehicle Photo Upload** | Optional display-only photo upload (does not affect prediction) |
| **Modern UI** | Plus Jakarta Sans typography, gradient result cards, metric chips |
| **Model Info Panel** | Expandable section with R², MAE, and methodology notes |

---

## Project Structure

```
used_cars_data/
│
├── app.py                   # Streamlit front-end (UI + prediction calls)
├── train_model.py           # Data preprocessing, model training, artifact export
│
├── model_artifacts.pkl      # Saved scikit-learn pipeline + UI metadata (generated)
│
├── UsedCars_Combined.csv    # Primary training dataset (1,725 rows)
├── Dataset.csv              # Extended dataset with engineered features
├── UsedCars_Banglore.csv    # City-level raw data
├── UsedCars_Chennai.csv
├── UsedCars_Delhi.csv
├── UsedCars_Hyderabad.csv
├── UsedCars_Kolkata.csv
├── UsedCars_Mumbai.csv
├── UsedCars_Pune.csv
│
└── README.md                # This file
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Indian used-car listings (combined city datasets) |
| Total records | 1,725 |
| Target variable | `price` (INR) |
| Price range | Rs. 62,500 – Rs. 1.47 Cr |
| Median price | Rs. 8.75 L |
| Year range | 1990 – 2021 |
| Cities covered | Bangalore, Chennai, Delhi, Dehradun, Hyderabad, Kolkata, Mumbai, Pune (and surrounding areas) |
| Brands | 31 (Hyundai, Maruti Suzuki, Honda, Mercedes-Benz, Toyota, BMW, Audi, Mahindra, Ford, Volkswagen, and more) |

### Columns Used for Prediction

| Column | Type | Notes |
|---|---|---|
| `year` | Numerical | Manufacturing year → converted to `car_age` |
| `distance_travelled(kms)` | Numerical | Odometer reading; top 1% clipped |
| `brand` | Categorical | 31 unique brands |
| `fuel_type` | Categorical | Petrol, Diesel, CNG/Hybrid |
| `city` | Categorical | 15 cities/regions |

> Columns excluded: `Id`, `full_model_name`, `model_name` — IDs and high-cardinality free-text fields not suitable as stable prediction features.

---

## Model

### Algorithm

**Gradient Boosting Regressor** (scikit-learn `GradientBoostingRegressor`)

Chosen for its strong performance on tabular data with mixed feature types and its robustness to outliers compared to linear models.

### Preprocessing Pipeline

```
ColumnTransformer
├── StandardScaler          → car_age, distance_travelled(kms)
└── OrdinalEncoder          → brand, fuel_type, city
        (unknown_value=-1 for unseen categories)
```

### Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 400 |
| `learning_rate` | 0.08 |
| `max_depth` | 5 |
| `min_samples_leaf` | 5 |
| `subsample` | 0.85 |
| `random_state` | 42 |

### Performance (80/20 train-test split)

| Metric | Score |
|---|---|
| R² | 0.66 |
| MAE | Rs. 5.03 L (~$6,000 USD) |
| Training samples | 1,380 |
| Test samples | 345 |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/carvalue.git
cd carvalue

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install pandas scikit-learn streamlit numpy joblib

# 4. Train the model (generates model_artifacts.pkl)
python train_model.py

# 5. Launch the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

---

## Usage

### Seller Mode

1. Click **Seller Mode** at the top of the page.
2. Select your car's **brand**, **manufacturing year**, **fuel type**, and **city**.
3. Drag the **distance travelled** slider to your odometer reading.
4. Optionally upload a photo of the vehicle.
5. Click **Estimate Price**.
6. The app displays the predicted fair market price and a typical high/low range.

### Buyer Mode

1. Click **Buyer Mode** at the top of the page.
2. Enter the same vehicle details as above.
3. Check **"I have an asking price I'd like to compare"** and enter the seller's asking price.
4. Click **Evaluate This Car**.
5. The app shows the estimated market value and a three-tier deal analysis:
   - 🟢 **Great Value** — asking price is below the market benchmark
   - 🟡 **Aligned with Market** — asking price is within ~8% of benchmark
   - 🟠 **Above Our Market Benchmark** — asking price exceeds the benchmark; provides context for negotiation

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `scikit-learn` | ML pipeline, preprocessing, and model |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `joblib` | Model serialisation |

Install all at once:

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

---

## Limitations & Disclaimer

- Predictions are **indicative market benchmarks**, not professional valuations.
- The model does not account for vehicle **condition**, **accident history**, **service records**, or **optional extras** — all of which can meaningfully affect the actual transaction price.
- The training dataset covers the **Indian used-car market only**. The model is not suitable for other markets.
- Always **inspect the vehicle in person** and verify service history before making any purchase decision.

---

## Future Improvements

- [ ] Integrate a vehicle condition rating input (Excellent / Good / Fair / Poor)
- [ ] Add model-level predictions (not just brand) using a refined feature encoding strategy
- [ ] Expand the dataset with more recent listings via web scraping
- [ ] Implement SHAP values for per-prediction explainability
- [ ] Add a price trend visualisation by brand and year

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Author

Built as a portfolio project demonstrating end-to-end ML application development — from raw data exploration and model training to a polished, user-facing web interface.

*Feel free to fork, star, or open issues. Contributions are welcome.*
