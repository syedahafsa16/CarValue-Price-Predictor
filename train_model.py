"""
CarValue — Model Training Script
Trains a GradientBoostingRegressor on the UsedCars_Combined.csv dataset and
saves the pipeline + metadata as model_artifacts.pkl.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "UsedCars_Combined.csv"
ARTIFACT_PATH = "model_artifacts.pkl"

# ── 1. Load & clean ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Simplify rare fuel types
df["fuel_type"] = df["fuel_type"].replace(
    {"Petrol + 1": "CNG/Hybrid", "CNG + 1": "CNG/Hybrid", "Hybrid": "CNG/Hybrid"}
)

# Drop rows with missing target or obviously wrong values
df = df.dropna(subset=["price"])
df = df[df["price"] > 0]
df = df[df["distance_travelled(kms)"] >= 0]

# Clip extreme outliers for distance (top 1 %)
upper_dist = df["distance_travelled(kms)"].quantile(0.99)
df["distance_travelled(kms)"] = df["distance_travelled(kms)"].clip(upper=upper_dist)

# Derive car age from year (relative to last year in dataset)
REFERENCE_YEAR = 2022
df["car_age"] = REFERENCE_YEAR - df["year"]
df["car_age"] = df["car_age"].clip(lower=0)

# ── 2. Feature selection ─────────────────────────────────────────────────────
FEATURES = ["car_age", "distance_travelled(kms)", "brand", "fuel_type", "city"]
TARGET = "price"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ── 3. Collect domain knowledge for UI dropdowns ─────────────────────────────
brands = sorted(df["brand"].unique().tolist())
fuel_types = sorted(df["fuel_type"].unique().tolist())
cities = sorted(df["city"].unique().tolist())
year_min = int(df["year"].min())
year_max = int(df["year"].max())

# ── 4. Build preprocessing + model pipeline ──────────────────────────────────
cat_features = ["brand", "fuel_type", "city"]
num_features = ["car_age", "distance_travelled(kms)"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_features,
        ),
    ]
)

model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.08,
    max_depth=5,
    min_samples_leaf=5,
    subsample=0.85,
    random_state=42,
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

# ── 5. Train / evaluate ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"[OK] Model trained on {len(X_train)} samples, evaluated on {len(X_test)} samples")
print(f"  MAE  : Rs.{mae:,.0f}")
print(f"  R2   : {r2:.4f}")

# ── 6. Save artifacts ─────────────────────────────────────────────────────────
artifacts = {
    "pipeline": pipeline,
    "brands": brands,
    "fuel_types": fuel_types,
    "cities": cities,
    "year_min": year_min,
    "year_max": year_max,
    "reference_year": REFERENCE_YEAR,
    "mae": mae,
    "r2": r2,
    "n_train": len(X_train),
    "n_test": len(X_test),
}

joblib.dump(artifacts, ARTIFACT_PATH)
print(f"[OK] Artifacts saved to {ARTIFACT_PATH}")
