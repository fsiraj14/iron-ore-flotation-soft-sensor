import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flotation Plant Quality Dashboard", layout="wide")

st.title("⚙️ Flotation Plant Quality Prediction Dashboard")
st.write("This application predicts the final **% Silica Concentrate** (impurity level) in the flotation process using real-time telemetry inputs.")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('model_lasso.joblib')

try:
    model = load_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Sidebar Inputs for features
st.sidebar.header("Real-Time Telemetry Inputs")

pct_iron_feed = st.sidebar.slider("% Iron Feed", 40.0, 70.0, 55.0, 0.1)
amina_flow = st.sidebar.slider("Amina Flow (m³/h)", 200.0, 800.0, 500.0, 1.0)
ore_pulp_ph = st.sidebar.slider("Ore Pulp pH", 8.0, 12.0, 10.0, 0.1)
col_02_air_flow = st.sidebar.slider("Column 2 Air Flow", 200.0, 350.0, 250.0, 1.0)
col_06_air_flow = st.sidebar.slider("Column 6 Air Flow", 200.0, 350.0, 250.0, 1.0)
col_07_air_flow = st.sidebar.slider("Column 7 Air Flow", 200.0, 350.0, 250.0, 1.0)
col_04_level = st.sidebar.slider("Column 4 Level", 100.0, 800.0, 450.0, 1.0)
col_07_level = st.sidebar.slider("Column 7 Level", 100.0, 800.0, 450.0, 1.0)

# Interactive Lags (1 hour & 2 hours ago)
st.sidebar.subheader("Historical Lag Measurements (1h and 2h Lags)")
amina_flow_lag1 = st.sidebar.number_input("Amina Flow (1h Lag)", value=500.0)
amina_flow_lag2 = st.sidebar.number_input("Amina Flow (2h Lag)", value=500.0)

silica_target_lag1 = st.sidebar.number_input("Silica % (1h Lag)", value=2.0)
silica_target_lag2 = st.sidebar.number_input("Silica % (2h Lag)", value=2.0)

# Prepare input array (Must match training feature alignment)
# List of expected features: 
# 8 original features + 18 lag features = 26 features
features_list = [
    'Pct_Iron_Feed', 'Amina_Flow', 'Ore_Pulp_pH', 
    'Flotation_Column_02_Air_Flow', 'Flotation_Column_06_Air_Flow', 
    'Flotation_Column_07_Air_Flow', 'Flotation_Column_04_Level', 
    'Flotation_Column_07_Level'
]

# Mocking other features with default values for prediction validation
input_data = {
    'Pct_Iron_Feed': pct_iron_feed,
    'Amina_Flow': amina_flow,
    'Ore_Pulp_pH': ore_pulp_ph,
    'Flotation_Column_02_Air_Flow': col_02_air_flow,
    'Flotation_Column_06_Air_Flow': col_06_air_flow,
    'Flotation_Column_07_Air_Flow': col_07_air_flow,
    'Flotation_Column_04_Level': col_04_level,
    'Flotation_Column_07_Level': col_07_level,
}

# Add corresponding Lags
for f in features_list:
    if f == 'Amina_Flow':
        input_data[f'{f}_lag_1'] = amina_flow_lag1
        input_data[f'{f}_lag_2'] = amina_flow_lag2
    else:
        # Fill rest of lag features with current state for baseline simplicity
        input_data[f'{f}_lag_1'] = input_data[f]
        input_data[f'{f}_lag_2'] = input_data[f]

input_data['Pct_Silica_Concentrate_lag_1'] = silica_target_lag1
input_data['Pct_Silica_Concentrate_lag_2'] = silica_target_lag2

input_df = pd.DataFrame([input_data])

# Prediction Action
if st.button("🔮 Predict Silica Concentration"):
    prediction = model.predict(input_df)[0]
    st.metric(label="Predicted % Silica Concentrate", value=f"{prediction:.3f} %", delta=f"{prediction - 2.0:.2f} % deviation from Target (2%)")
    
    if prediction > 3.0:
        st.warning("⚠️ High silica content detected! Adjust amina reagent flows.")
    else:
        st.success("✅ Concentrate meets quality thresholds.")
