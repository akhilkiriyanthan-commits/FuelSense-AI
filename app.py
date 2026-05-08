import streamlit as st
import joblib
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FuelSense AI - MPG Predictor", page_icon="⛽", layout="centered")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("mpg_model.pkl")

model = load_model()

# ── All 326 feature names in exact order from training ────────────────────────
FEATURE_NAMES = ['cylinders', 'displacement', 'year', 'class_large car', 'class_midsize car', 'class_midsize station wagon', 'class_minicompact car', 'class_minivan', 'class_small pickup truck', 'class_small sport utility vehicle', 'class_small station wagon', 'class_standard pickup truck', 'class_standard sport utility vehicle', 'class_subcompact car', 'class_two seater', 'drive_awd', 'drive_fwd', 'drive_rwd', 'fuel_type_electricity', 'fuel_type_gas', 'make_aston martin', 'make_audi', 'make_bentley', 'make_bmw', 'make_buick', 'make_cadillac', 'make_chevrolet', 'make_chrysler', 'make_ford', 'make_genesis', 'make_gmc', 'make_honda', 'make_hyundai', 'make_infiniti', 'make_jaguar', 'make_jeep', 'make_kia', 'make_land rover', 'make_mazda', 'make_mercedes-benz', 'make_mini', 'make_mitsubishi', 'make_nissan', 'make_porsche', 'make_ram', 'make_roush performance', 'make_subaru', 'make_toyota', 'make_volkswagen', 'make_volvo', 'model_1500 4wd', 'model_230i coupe', 'model_430i convertible', 'model_430i coupe', 'model_430i gran coupe', 'model_430i xdrive convertible', 'model_430i xdrive coupe', 'model_430i xdrive gran coupe', 'model_440i convertible', 'model_440i coupe', 'model_440i gran coupe', 'model_440i xdrive convertible', 'model_440i xdrive coupe', 'model_440i xdrive gran coupe', 'model_6', 'model_640i convertible', 'model_640i gran coupe', 'model_640i xdrive convertible', 'model_640i xdrive gran coupe', 'model_650i convertible', 'model_650i gran coupe', 'model_650i xdrive convertible', 'model_650i xdrive gran coupe', 'model_740i xdrive', 'model_750i xdrive', 'model_840i convertible', 'model_840i coupe', 'model_840i gran coupe', 'model_840i xdrive convertible', 'model_840i xdrive coupe', 'model_840i xdrive gran coupe', 'model_a4', 'model_a4 quattro', 'model_a5 cabriolet quattro', 'model_a5 quattro', 'model_a5 sportback quattro', 'model_a6 quattro', 'model_a7 quattro', 'model_a8 l', 'model_alpina b6 xdrive gran coupe', 'model_alpina b7', 'model_alpina b8 gran coupe', 'model_atlas', 'model_atlas 4motion', 'model_bentayga', 'model_canyon 2wd', 'model_carnival', 'model_cayenne', 'model_cayenne turbo', 'model_cherokee 4wd', 'model_cherokee 4wd active drive ii', 'model_cherokee fwd', 'model_cherokee trailhawk 4wd', 'model_colorado 2wd', 'model_colorado 4wd', 'model_compass 4wd', 'model_compass fwd', 'model_cooper convertible', 'model_cooper countryman', 'model_cooper countryman all4', 'model_cooper hardtop 2 door', 'model_cooper hardtop 4 door', 'model_cooper s clubman', 'model_cooper s convertible', 'model_cooper s hardtop 2 door', 'model_cooper s hardtop 4 door', 'model_corolla', 'model_corolla hybrid', 'model_corolla xle', 'model_corolla xse', 'model_corvette', 'model_corvette z06', 'model_corvette zr1', 'model_cx-5 2wd', 'model_cx-5 4wd', 'model_db9', 'model_defender 110', 'model_defender 110 mhev', 'model_defender 90', 'model_defender 90 mhev', 'model_discovery', 'model_discovery mhev', 'model_discovery sport', 'model_e-pace', 'model_e-pace p300', 'model_eclipse cross 2wd', 'model_eclipse cross 4wd', 'model_eclipse cross es 2wd', 'model_eclipse cross es 4wd', 'model_elantra', 'model_elantra gt', 'model_elantra se', 'model_encore gx fwd', 'model_envision awd', 'model_envision fwd', 'model_equinox awd', 'model_equinox fwd', 'model_equus', 'model_escalade 2wd', 'model_escalade 4wd', 'model_escalade esv 2wd', 'model_escalade esv 4wd', 'model_evoque', 'model_f-pace', 'model_f-pace 30t', 'model_f-type convertible', 'model_f-type coupe', 'model_f-type p450 awd r-dynamic convertible', 'model_f-type p450 awd r-dynamic coupe', 'model_f-type p450 rwd convertible', 'model_f-type p450 rwd coupe', 'model_f-type r awd convertible', 'model_f-type r awd coupe', 'model_f-type r convertible', 'model_f-type r coupe', 'model_f-type s awd convertible', 'model_f-type s awd coupe', 'model_f-type s convertible', 'model_f-type s coupe', 'model_f-type v8 s convertible', 'model_forester awd', 'model_g70 awd', 'model_g70 rwd', 'model_g80 awd', 'model_g80 rwd', 'model_gle350 4matic', 'model_gle450 4matic', 'model_grand cherokee srt8', 'model_gt-r', 'model_gv70 awd', 'model_gv80 awd', 'model_hr-v 2wd', 'model_hr-v 4wd', 'model_i-miev', 'model_ilx', 'model_insight', 'model_insight touring', 'model_jetta', 'model_john cooper works convertible', 'model_john cooper works gp', 'model_john cooper works hardtop', 'model_john cooper works hardtop 2 door', 'model_k900', 'model_kona awd', 'model_kona fwd', 'model_m4 competition coupe', 'model_m4 competition m xdrive convertible', 'model_m4 competition m xdrive coupe', 'model_m4 convertible', 'model_m4 coupe', 'model_m5 cs sedan', 'model_m6 convertible', 'model_m8 competition convertible', 'model_m8 competition coupe', 'model_m8 competition gran coupe', 'model_m850i xdrive convertible', 'model_m850i xdrive coupe', 'model_m850i xdrive gran coupe', 'model_mdx awd', 'model_mdx fwd', 'model_mirage', 'model_mulsanne', 'model_murano crosscabriolet', 'model_mustang', 'model_mustang convertible', 'model_odyssey', 'model_outlander 2wd', 'model_outlander 4wd', 'model_pacifica', 'model_palisade awd', 'model_patriot 4wd', 'model_patriot fwd', 'model_q5', 'model_q7', 'model_qx50', 'model_qx50 awd', 'model_r8', 'model_r8 spyder', 'model_range rover evoque', 'model_range rover evoque mhev', 'model_range rover velar', 'model_range rover velar p300', 'model_rapide s', 'model_rdx awd', 'model_rdx fwd', 'model_ridgeline awd', 'model_ridgeline fwd', 'model_rlx', 'model_rlx hybrid', 'model_s4', 'model_s5', 'model_s5 cabriolet', 'model_s5 sportback', 'model_s60 awd', 'model_s80 awd', 'model_santa fe awd', 'model_santa fe fwd', 'model_santa fe sport awd', 'model_santa fe sport fwd', 'model_santa fe sport ultimate awd', 'model_santa fe sport ultimate fwd', 'model_santa fe ultimate awd', 'model_santa fe ultimate fwd', 'model_sedona', 'model_sedona sx', 'model_sedona sxl', 'model_seltos awd', 'model_seltos fwd', 'model_sierra c15 2wd', 'model_sierra k15 4wd', 'model_silverado c15 2wd', 'model_silverado k15 4wd', 'model_sorento 2wd', 'model_sorento 4wd', 'model_sorento awd', 'model_sorento fe awd', 'model_sorento fwd', 'model_soul', 'model_soul eco dynamics', 'model_sportage awd', 'model_sportage fe awd', 'model_sportage fe fwd', 'model_sportage fwd', 'model_sportage hybrid awd', 'model_sportage hybrid fwd', 'model_sq5', 'model_stage 3 mustang', 'model_stinger awd', 'model_stinger rwd', 'model_suburban c1500 2wd', 'model_suburban k1500 4wd', 'model_tahoe c1500 2wd', 'model_tahoe k1500 4wd', 'model_telluride awd', 'model_telluride fwd', 'model_tlx awd', 'model_tlx awd a-spec', 'model_tlx fwd a-spec', 'model_trailblazer awd', 'model_trailblazer fwd', 'model_trax', 'model_tt coupe quattro', 'model_tt roadster quattro', 'model_tucson awd', 'model_tucson fwd', 'model_tucson hybrid', 'model_tucson hybrid blue', 'model_v60 awd', 'model_vanquish', 'model_veloster', 'model_volt', 'model_wrx', 'model_x2 m35i', 'model_x2 xdrive28i', 'model_xc40 awd', 'model_xc60 awd', 'model_xc70 awd', 'model_xe', 'model_xe 30t', 'model_xe awd', 'model_xe awd 30t', 'model_xf', 'model_xf 30t', 'model_xf awd', 'model_xf awd 30t', 'model_xf sportbrake awd', 'model_xj', 'model_xjl', 'model_xt5', 'model_xt5 awd', 'model_yukon c1500 2wd', 'model_yukon c1500 xl 2wd', 'model_yukon k1500 4wd', 'model_yukon k1500 xl 4wd', 'model_z4 sdrive28i', 'transmission_m']

# ── Dropdown options ──────────────────────────────────────────────────────────
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
    # Build a dict with all features set to 0
    input_dict = {f: 0 for f in FEATURE_NAMES}

    # Set numeric features
    input_dict['cylinders']    = cylinders
    input_dict['displacement'] = displacement
    input_dict['year']         = year

    # Set one-hot encoded features
    make_key  = f"make_{make.lower()}"
    fuel_key  = f"fuel_type_{fuel_type.lower()}"
    class_key = f"class_{car_class.lower()}"
    drive_key = f"drive_{drive.lower()}"

    for key in [make_key, fuel_key, class_key, drive_key]:
        if key in input_dict:
            input_dict[key] = 1

    if transmission == "Manual":
        input_dict['transmission_m'] = 1

    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]

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
