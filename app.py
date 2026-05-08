import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelSense AI - MPG Predictor", page_icon="⛽", layout="centered")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("mpg_model.pkl")

model = load_model()

# ── Dropdown options (must match training data exactly) ───────────────────────
MAKES = ['aston martin', 'audi', 'bentley', 'bmw', 'buick', 'cadillac', 'chevrolet',
         'chrysler', 'ford', 'genesis', 'gmc', 'honda', 'hyundai', 'infiniti', 'jaguar',
         'jeep', 'kia', 'land rover', 'mazda', 'mercedes-benz', 'mini', 'mitsubishi',
         'nissan', 'porsche', 'ram', 'roush performance', 'subaru', 'toyota',
         'volkswagen', 'volvo']

FUEL_TYPES = ['electricity', 'gas']

CLASSES = ['large car', 'midsize car', 'midsize station wagon', 'minicompact car',
           'minivan', 'small pickup truck', 'small sport utility vehicle',
           'small station wagon', 'standard pickup truck',
           'standard sport utility vehicle', 'subcompact car', 'two seater']

DRIVES = ['awd', 'fwd', 'rwd']

TRANSMISSIONS = ['automatic', 'manual']

CYLINDERS = [2, 3, 4, 5, 6, 8, 10, 12, 16]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⛽ FuelSense AI")
st.markdown("Predict your car's fuel efficiency (MPG) using AI.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        make         = st.selectbox("Make", [m.title() for m in MAKES])
        fuel_type    = st.selectbox("Fuel Type", ["Electricity", "Gas"])
        car_class    = st.selectbox("Vehicle Class", [c.title() for c in CLASSES])
        drive        = st.selectbox("Drive", ["AWD", "FWD", "RWD"])

    with col2:
        cylinders    = st.selectbox("Cylinders", CYLINDERS, index=2)
        displacement = st.selectbox("Displacement (L)", [round(x * 0.1, 1) for x in range(5, 101)], index=15)
        year         = st.selectbox("Year", list(range(2025, 1989, -1)), index=7)
        transmission = st.selectbox("Transmission", ["Automatic", "Manual"])

    submitted = st.form_submit_button("Predict MPG", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Build a dict with all 326 features set to 0
    feature_names = model.feature_names_in_
    input_dict = {f: 0 for f in feature_names}

    # Set numeric features
    input_dict['cylinders']     = cylinders
    input_dict['displacement']  = displacement
    input_dict['year']          = year

    # Set one-hot encoded features
    make_key = f"make_{make.lower()}"
    fuel_key = f"fuel_type_{fuel_type.lower()}"
    class_key = f"class_{car_class.lower()}"
    drive_key = f"drive_{drive.lower()}"

    if make_key in input_dict:
        input_dict[make_key] = 1
    if fuel_key in input_dict:
        input_dict[fuel_key] = 1
    if class_key in input_dict:
        input_dict[class_key] = 1
    if drive_key in input_dict:
        input_dict[drive_key] = 1

    # Transmission (manual = 1, automatic = 0)
    if transmission == "Manual":
        input_dict['transmission_m'] = 1

    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])[feature_names]

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"### ⛽ Estimated Fuel Efficiency: **{prediction:.1f} MPG**")

        if prediction < 20:
            st.warning("🔴 Low efficiency")
        elif prediction < 35:
            st.info("🟡 Average efficiency")
        else:
            st.success("🟢 High efficiency")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Powered by Random Forest · Built with Streamlit")
