import streamlit as st
import joblib
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MPG Predictor", page_icon="⛽", layout="centered")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("mpg_model.pkl")

model = load_model()

# ── Dropdown options (edit these to match your training data exactly) ─────────
MAKES = [
    "Acura", "Audi", "BMW", "Buick", "Cadillac", "Chevrolet", "Chrysler",
    "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Jeep", "Kia",
    "Land Rover", "Lexus", "Lincoln", "Mazda", "Mercedes-Benz", "Mercury",
    "Mini", "Mitsubishi", "Nissan", "Pontiac", "Porsche", "Ram", "Saturn",
    "Scion", "Subaru", "Suzuki", "Toyota", "Volkswagen", "Volvo",
]

FUEL_TYPES = ["Petrol", "Diesel", "Electric", "Hybrid", "CNG", "LPG"]

CYLINDERS = [2, 3, 4, 5, 6, 8, 10, 12, 16]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⛽ Estimated MPG Predictor")
st.markdown("Select your car details below to get an estimated fuel efficiency (MPG).")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        make        = st.selectbox("Make", MAKES)
        fuel_type   = st.selectbox("Fuel Type", FUEL_TYPES)
        cylinders   = st.selectbox("Cylinders", CYLINDERS, index=2)   # default 4

    with col2:
        displacement = st.selectbox("Displacement (L)", [round(x * 0.1, 1) for x in range(5, 101)], index=15)  # default 2.0
        year         = st.selectbox("Year", list(range(2025, 1989, -1)), index=7)   # default 2018

    submitted = st.form_submit_button("Predict MPG", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Encode categoricals as integer indices.
    # ⚠️  If your model used a LabelEncoder or Pipeline, adjust this section —
    #     see the DEPLOY_GUIDE.md for details.
    make_encoded  = MAKES.index(make)
    fuel_encoded  = FUEL_TYPES.index(fuel_type)

    features = np.array([[make_encoded, fuel_encoded, cylinders, displacement, year]])

    try:
        prediction = model.predict(features)[0]
        st.success(f"### Estimated Fuel Efficiency: **{prediction:.1f} MPG**")

        # Visual efficiency indicator
        if prediction < 20:
            st.warning("🔴 Low efficiency")
        elif prediction < 35:
            st.info("🟡 Average efficiency")
        else:
            st.success("🟢 High efficiency")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "This usually means the feature encoding here doesn't match what the "
            "model was trained on. Check the encoding section in DEPLOY_GUIDE.md."
        )

st.markdown("---")
st.caption("Model loaded from model.pkl · Built with Streamlit")
