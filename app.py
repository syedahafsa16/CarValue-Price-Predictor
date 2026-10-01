"""
CarValue — Used Car Price Prediction
Streamlit app with Seller Mode and Buyer Mode.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarValue — Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
 <link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Base typography ─────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, "Segoe UI", system-ui, sans-serif !important;
}
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 780px;
}

/* ── Brand header ─────────────────────────────────────────────────────────── */
.brand-header { text-align: center; margin-bottom: 0.6rem; }
.brand-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #0f172a;
    line-height: 1.1;
}
.brand-title span { color: #2563eb; }
.brand-subtitle {
    color: #64748b;
    font-size: 1rem;
    font-weight: 500;
    margin-top: 0.25rem;
    letter-spacing: 0.01em;
}

/* ── Mode banner ─────────────────────────────────────────────────────────── */
.mode-banner {
    border-radius: 12px;
    padding: 0.85rem 1.2rem;
    margin: 0.9rem 0 1.6rem 0;
    font-size: 0.93rem;
    font-weight: 500;
    line-height: 1.5;
}
.mode-banner-seller {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 4px solid #2563eb;
    color: #1e3a6e;
}
.mode-banner-buyer {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-left: 4px solid #16a34a;
    color: #14532d;
}

/* ── Section heading ─────────────────────────────────────────────────────── */
.section-heading {
    font-size: 0.78rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.6rem 0 0.7rem 0;
}

/* ── Result card — shared base ───────────────────────────────────────────── */
.result-card {
    border-radius: 20px;
    padding: 2rem 2.2rem;
    text-align: center;
    margin-top: 1.4rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 130px; height: 130px;
    border-radius: 50%;
    opacity: 0.08;
    background: currentColor;
}

/* Seller card */
.result-card-seller {
    background: linear-gradient(145deg, #eff6ff 0%, #dbeafe 100%);
    border: 1.5px solid #93c5fd;
    box-shadow: 0 4px 24px rgba(37, 99, 235, 0.10);
}
/* Buyer market-value card */
.result-card-market {
    background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1.5px solid #cbd5e1;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
}
/* Deal analysis cards */
.result-card-great {
    background: linear-gradient(145deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1.5px solid #86efac;
    box-shadow: 0 4px 24px rgba(22, 163, 74, 0.10);
}
.result-card-fair {
    background: linear-gradient(145deg, #fefce8 0%, #fef9c3 100%);
    border: 1.5px solid #fde047;
    box-shadow: 0 4px 24px rgba(202, 138, 4, 0.10);
}
.result-card-above {
    background: linear-gradient(145deg, #fff7ed 0%, #ffedd5 100%);
    border: 1.5px solid #fdba74;
    box-shadow: 0 4px 24px rgba(234, 88, 12, 0.08);
}

/* ── Result card content ─────────────────────────────────────────────────── */
.rc-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.rc-price {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -2px;
    color: #0f172a;
    line-height: 1;
}
.rc-range {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.55rem;
    font-weight: 500;
}

/* ── Deal badge ──────────────────────────────────────────────────────────── */
.deal-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.8rem;
}
.badge-great  { background: #dcfce7; color: #15803d; }
.badge-fair   { background: #fef9c3; color: #a16207; }
.badge-above  { background: #ffedd5; color: #c2410c; }

.deal-headline {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.deal-subline {
    font-size: 0.88rem;
    color: #475569;
    line-height: 1.6;
    margin-top: 0.6rem;
}

/* ── Price comparison table ──────────────────────────────────────────────── */
.compare-table {
    margin: 1.1rem auto 0 auto;
    border-collapse: collapse;
    font-size: 0.92rem;
    width: auto;
    min-width: 280px;
}
.compare-table td {
    padding: 0.38rem 1rem;
    vertical-align: middle;
}
.compare-table .ct-label {
    color: #64748b;
    font-weight: 500;
    text-align: right;
    white-space: nowrap;
}
.compare-table .ct-value {
    font-weight: 700;
    color: #0f172a;
    text-align: left;
}
.compare-table .ct-diff {
    font-weight: 600;
    text-align: left;
}
.compare-table tr.ct-divider td {
    border-top: 1px solid #e2e8f0;
    padding-top: 0.55rem;
}
.ct-below { color: #16a34a; }
.ct-above { color: #ea580c; }
.ct-even  { color: #7c3aed; }

/* ── Metric chips ────────────────────────────────────────────────────────── */
.metric-band {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.1rem;
    justify-content: center;
    flex-wrap: wrap;
}
.metric-chip {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.55rem 1.1rem;
    text-align: center;
    min-width: 100px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.mc-val { font-size: 1rem; font-weight: 700; color: #0f172a; }
.mc-lbl { font-size: 0.7rem; color: #94a3b8; margin-top: 0.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }

/* ── Disclaimer text ─────────────────────────────────────────────────────── */
.disclaimer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 1rem;
    line-height: 1.55;
}

/* ── Footer ──────────────────────────────────────────────────────────────── */
.cv-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #e2e8f0;
    font-weight: 500;
}

/* ── Streamlit widget label polish ───────────────────────────────────────── */
label[data-testid="stWidgetLabel"] p,
.stSelectbox label p,
.stSlider label p,
.stNumberInput label p,
.stCheckbox label p {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #374151 !important;
}

/* Streamlit primary button polish */
.stButton > button[kind="primary"] {
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.01em !important;
}
.stButton > button[kind="secondary"] {
    border-radius: 12px !important;
    font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Load artifacts ────────────────────────────────────────────────────────────
ARTIFACT_PATH = "model_artifacts.pkl"

@st.cache_resource(show_spinner="Loading prediction model…")
def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        st.error(
            "Model artifacts not found. Please run `python train_model.py` first."
        )
        st.stop()
    return joblib.load(ARTIFACT_PATH)


artifacts = load_artifacts()
pipeline = artifacts["pipeline"]
brands = artifacts["brands"]
fuel_types = artifacts["fuel_types"]
cities = artifacts["cities"]
year_min = artifacts["year_min"]
year_max = artifacts["year_max"]
ref_year = artifacts["reference_year"]
model_mae = artifacts["mae"]
model_r2 = artifacts["r2"]

# ── Helper ────────────────────────────────────────────────────────────────────

def format_inr(value: float) -> str:
    """Format a number as Indian Rupees with lakh/crore notation."""
    v = int(round(value))
    if v >= 10_000_000:
        return f"Rs. {v/10_000_000:.2f} Cr"
    elif v >= 100_000:
        return f"Rs. {v/100_000:.2f} L"
    else:
        return f"Rs. {v:,}"


def predict_price(year: int, distance: float, brand: str, fuel_type: str, city: str) -> float:
    car_age = ref_year - year
    car_age = max(0, car_age)
    row = pd.DataFrame(
        {
            "car_age": [car_age],
            "distance_travelled(kms)": [distance],
            "brand": [brand],
            "fuel_type": [fuel_type],
            "city": [city],
        }
    )
    price = pipeline.predict(row)[0]
    return max(0.0, float(price))


# ── Brand header ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="brand-header">
    <div class="brand-title">Car<span>Value</span></div>
    <div class="brand-subtitle">AI-powered used car price estimation</div>
</div>
<div style="text-align:center;margin-bottom:1.2rem;">
    <span style="display:inline-block;background:#f1f5f9;border-radius:999px;
                 padding:0.28rem 0.9rem;font-size:0.75rem;font-weight:600;
                 color:#64748b;letter-spacing:0.05em;">
        INDIA USED CAR MARKET &nbsp;&bull;&nbsp; 1,725 TRANSACTIONS
    </span>
</div>
""",
    unsafe_allow_html=True,
)

# ── Mode selection ────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "seller"

col_a, col_b = st.columns(2)
with col_a:
    if st.button(
        "🏷️  Seller Mode\nFind out what your car is worth",
        use_container_width=True,
        type="primary" if st.session_state.mode == "seller" else "secondary",
    ):
        st.session_state.mode = "seller"
        st.rerun()

with col_b:
    if st.button(
        "🔍  Buyer Mode\nEvaluate any car before you buy",
        use_container_width=True,
        type="primary" if st.session_state.mode == "buyer" else "secondary",
    ):
        st.session_state.mode = "buyer"
        st.rerun()

mode = st.session_state.mode

if mode == "seller":
    st.markdown(
        """
<div class="mode-banner mode-banner-seller">
    <strong>Seller Mode</strong> &mdash; Enter your vehicle details below.
    CarValue will estimate the <em>fair market selling price</em> based on
    real transaction data from major Indian cities.
</div>
""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<div class="mode-banner mode-banner-buyer">
    <strong>Buyer Mode</strong> &mdash; Enter the details of the car you are
    considering. Optionally provide the seller&rsquo;s asking price and
    CarValue will give you an objective market comparison.
</div>
""",
        unsafe_allow_html=True,
    )

# ── Shared form ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Vehicle Details</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    sel_brand = st.selectbox("Brand", options=brands, index=brands.index("Hyundai") if "Hyundai" in brands else 0)
with c2:
    sel_year = st.number_input(
        "Manufacturing Year",
        min_value=year_min,
        max_value=year_max,
        value=2018,
        step=1,
    )

c3, c4 = st.columns(2)
with c3:
    sel_fuel = st.selectbox("Fuel Type", options=fuel_types, index=0)
with c4:
    sel_city = st.selectbox("City / Location", options=cities, index=0)

sel_distance = st.slider(
    "Distance Travelled (km)",
    min_value=0,
    max_value=300_000,
    value=40_000,
    step=500,
    format="%d km",
)

# ── Buyer-only: asking price ──────────────────────────────────────────────────
asking_price = None
if mode == "buyer":
    st.markdown('<div class="section-heading">Seller\'s Asking Price <span style="font-size:0.7rem;font-weight:500;text-transform:none;letter-spacing:0;color:#94a3b8;">(optional)</span></div>', unsafe_allow_html=True)
    use_asking = st.checkbox("I have an asking price I'd like to compare against the market estimate")
    if use_asking:
        asking_price = st.number_input(
            "Asking Price (Rs.)",
            min_value=10_000,
            max_value=50_000_000,
            value=800_000,
            step=10_000,
            format="%d",
        )

# ── Optional photo upload ─────────────────────────────────────────────────────
with st.expander("📷  Add a vehicle photo *(optional — display only)*"):
    uploaded_photo = st.file_uploader(
        "Upload a photo of the vehicle",
        type=["jpg", "jpeg", "png", "webp"],
        help="Photo is displayed alongside your result but does not affect the price estimate.",
    )
    if uploaded_photo:
        st.image(uploaded_photo, caption="Your vehicle photo", use_container_width=True)

# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("")
predict_clicked = st.button(
    "Estimate Price" if mode == "seller" else "Evaluate This Car",
    type="primary",
    use_container_width=True,
)

if predict_clicked:
    predicted = predict_price(sel_year, sel_distance, sel_brand, sel_fuel, sel_city)
    car_age_display = ref_year - sel_year

    if mode == "seller":
        # ── Seller result ────────────────────────────────────────────────────
        low  = max(0, predicted - model_mae * 0.6)
        high = predicted + model_mae * 0.6

        st.markdown(
            f"""
<div class="result-card result-card-seller">
    <div class="rc-eyebrow">Estimated Market Selling Price</div>
    <div class="rc-price">{format_inr(predicted)}</div>
    <div class="rc-range">Typical market range &nbsp;&bull;&nbsp; {format_inr(low)} &ndash; {format_inr(high)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="metric-band">
    <div class="metric-chip">
        <div class="mc-val">{sel_brand}</div>
        <div class="mc-lbl">Brand</div>
    </div>
    <div class="metric-chip">
        <div class="mc-val">{sel_year}</div>
        <div class="mc-lbl">Year</div>
    </div>
    <div class="metric-chip">
        <div class="mc-val">{sel_distance:,} km</div>
        <div class="mc-lbl">Distance</div>
    </div>
    <div class="metric-chip">
        <div class="mc-val">{sel_fuel}</div>
        <div class="mc-lbl">Fuel</div>
    </div>
    <div class="metric-chip">
        <div class="mc-val">{car_age_display} yr{"s" if car_age_display != 1 else ""}</div>
        <div class="mc-lbl">Car Age</div>
    </div>
</div>
<div class="disclaimer">
    Estimate is based on real transaction data from the Indian used-car market.<br>
    Actual price may vary with the vehicle&rsquo;s condition, service records, and local demand.
</div>
""",
            unsafe_allow_html=True,
        )

    else:  # ── Buyer result ─────────────────────────────────────────────────
        # Market value card
        st.markdown(
            f"""
<div class="result-card result-card-market">
    <div class="rc-eyebrow">Estimated Market Value</div>
    <div class="rc-price">{format_inr(predicted)}</div>
    <div class="rc-range">{sel_brand} &nbsp;&bull;&nbsp; {sel_year} &nbsp;&bull;&nbsp; {sel_distance:,} km &nbsp;&bull;&nbsp; {sel_fuel} &nbsp;&bull;&nbsp; {sel_city}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if asking_price is not None:
            diff = asking_price - predicted
            pct  = (diff / predicted) * 100 if predicted > 0 else 0
            abs_diff_fmt = format_inr(abs(diff))

            # Tier thresholds
            # below market  → diff < 0
            # fair           → 0 <= pct <= 8
            # above market   → pct > 8
            if diff < 0:
                badge_cls   = "badge-great"
                badge_label = "Below Estimated Market Price"
                card_cls    = "result-card-great"
                headline    = "Great Value"
                subline = (
                    f"The seller&rsquo;s asking price of <strong>{format_inr(asking_price)}</strong> is "
                    f"<strong>{abs_diff_fmt} below</strong> our estimated market benchmark "
                    f"({abs(pct):.1f}% under market). This looks like a competitive listing "
                    f"&mdash; still, be sure to inspect the vehicle in person and verify its service history."
                )
                diff_cls = "ct-below"
                diff_label = f"{abs_diff_fmt} below market"
            elif pct <= 8:
                badge_cls   = "badge-fair"
                badge_label = "Fair Market Value"
                card_cls    = "result-card-fair"
                headline    = "Aligned with Market"
                subline = (
                    f"The seller&rsquo;s asking price of <strong>{format_inr(asking_price)}</strong> is "
                    f"closely aligned with our estimated market value"
                    f"{f' ({abs_diff_fmt} above, {pct:.1f}%)' if diff > 0 else ''}. "
                    f"This is within the normal range of market variation and generally represents "
                    f"a fair asking price. A test drive and vehicle inspection are always recommended."
                )
                diff_cls = "ct-even"
                diff_label = (
                    f"{abs_diff_fmt} above market ({pct:.1f}%)" if diff > 0
                    else "At estimated market value"
                )
            else:
                badge_cls   = "badge-above"
                badge_label = "Above Estimated Market Price"
                card_cls    = "result-card-above"
                headline    = "Above Our Market Benchmark"
                subline = (
                    f"The seller&rsquo;s asking price of <strong>{format_inr(asking_price)}</strong> is "
                    f"<strong>{abs_diff_fmt} above</strong> our estimated market benchmark "
                    f"({pct:.1f}% over market). You may find this estimate a helpful reference "
                    f"point during your conversation with the seller. Factors such as low mileage, "
                    f"a full service history, or recent upgrades can sometimes justify a higher asking price."
                )
                diff_cls = "ct-above"
                diff_label = f"{abs_diff_fmt} above market ({pct:.1f}%)"

            st.markdown(
                f"""
<div class="result-card {card_cls}" style="margin-top:1rem;">
    <span class="deal-badge {badge_cls}">{badge_label}</span>
    <div class="deal-headline">{headline}</div>
    <table class="compare-table">
        <tr>
            <td class="ct-label">Our market estimate:</td>
            <td class="ct-value">{format_inr(predicted)}</td>
        </tr>
        <tr>
            <td class="ct-label">Seller&rsquo;s asking price:</td>
            <td class="ct-value">{format_inr(asking_price)}</td>
        </tr>
        <tr class="ct-divider">
            <td class="ct-label">Difference:</td>
            <td class="ct-diff {diff_cls}">{diff_label}</td>
        </tr>
    </table>
    <div class="deal-subline">{subline}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="disclaimer">'
            "Always inspect the vehicle in person and verify its service history before making a purchase decision.<br>"
            "This estimate is a market reference, not a professional valuation."
            "</div>",
            unsafe_allow_html=True,
        )

# ── Model info expander ───────────────────────────────────────────────────────
with st.expander("About the prediction model"):
    st.markdown(
        f"""
**Algorithm:** Gradient Boosting Regressor
**Training data:** 1,380 real used-car transactions across major Indian cities
**Features used:** Brand, Manufacturing Year, Fuel Type, City, Distance Travelled
**Model accuracy (held-out test set):**
- R\u00b2 score: `{model_r2:.2f}` &mdash; explains {model_r2*100:.0f}% of price variance
- Mean Absolute Error: `{format_inr(model_mae)}` average prediction deviation

*Estimates are indicative market benchmarks. Vehicle condition, service history,
accident records, and optional extras can meaningfully affect the actual transaction price.*
"""
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='cv-footer'>"
    "CarValue &nbsp;&mdash;&nbsp; Used Car Price Estimator &nbsp;&bull;&nbsp; "
    "Powered by machine learning on Indian used-car market data"
    "</div>",
    unsafe_allow_html=True,
)
