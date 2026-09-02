import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Predictive Maintenance",
    page_icon="🛰️",
    layout="centered",
)

MODEL_PATH = "satellite maintanence_model.joblib"  # matches the filename saved in the notebook

# ------------------------------------------------------------------
# Load model (cached so it's not reloaded on every interaction)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# The notebook label-encoded 'Satellite Type' with sklearn's LabelEncoder,
# fit on values ['H', 'L', 'M'] -> encoder.classes_ is sorted alphabetically,
# so H=0, L=1, M=2. Update this mapping if your training data differed.
SATELLITE_TYPE_MAP = {"H (High)": 0, "L (Low)": 1, "M (Medium)": 2}

FEATURE_ORDER = [
    "Satellite Type",
    "Internal Tenperature(°C)",
    "Payload Temperature (°C)",
    "Reaction Wheel Speed(RPM)",
    "Reaction Wheel Torque(Nm)",
    "Component OPerating Hours",
]

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🛰️ Satellite Predictive Maintenance")
st.write(
    "Enter the current telemetry readings for a satellite component to predict "
    "the likelihood of failure."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        sat_type_label = st.selectbox("Satellite Type", list(SATELLITE_TYPE_MAP.keys()))
        internal_temp = st.number_input(
            "Internal Temperature (°C)", value=298.0, step=0.1, format="%.2f"
        )
        payload_temp = st.number_input(
            "Payload Temperature (°C)", value=308.0, step=0.1, format="%.2f"
        )

    with col2:
        rpm = st.number_input(
            "Reaction Wheel Speed (RPM)", value=1500.0, step=1.0, format="%.1f"
        )
        torque = st.number_input(
            "Reaction Wheel Torque (Nm)", value=40.0, step=0.1, format="%.2f"
        )
        operating_hours = st.number_input(
            "Component Operating Hours", value=100.0, step=1.0, format="%.1f"
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame(
        [[
            SATELLITE_TYPE_MAP[sat_type_label],
            internal_temp,
            payload_temp,
            rpm,
            torque,
            operating_hours,
        ]],
        columns=FEATURE_ORDER,
    )

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    failure_prob = proba[1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ Predicted: **Failure likely** ({failure_prob:.1%} probability)")
    else:
        st.success(f"✅ Predicted: **No failure expected** ({failure_prob:.1%} probability of failure)")

    st.progress(min(max(failure_prob, 0.0), 1.0))

    with st.expander("Input summary"):
        st.dataframe(input_df, use_container_width=True)

    with st.expander("Raw probabilities"):
        st.write(
            {
                "No Failure (0)": f"{proba[0]:.4f}",
                "Failure (1)": f"{proba[1]:.4f}",
            }
        )

st.divider()
st.caption(
    "Model: XGBoost classifier trained on satellite telemetry data "
    "(internal temperature, payload temperature, reaction wheel speed/torque, "
    "and component operating hours). Note: during training this model showed "
    "high recall but low precision for the failure class, so expect frequent "
    "false alarms — treat 'Failure likely' predictions as a prompt for "
    "further inspection, not a certainty."
)
