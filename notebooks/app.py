import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flotation Plant Optimization & Quality Dashboard", layout="wide")

# Design Header
st.write("# ⚙ ️ Smart Flotation Plant Control & Quality Optimizer")
st.write("### Business Problem: Real-Time Silica Minimization & Reagent Optimization")
st.write("In froth flotation, silica is an impurity. Operators must keep the final Pct Silica Concentrate as low as possible (target < 2.0%).")
st.write("* **The Challenge:** Adjusting reagents like Amina Flow is expensive, and changes take hours to physically affect the system due to process delay.")
st.write("* **The Solution:** This optimizer predicts silica content using Option A (Hourly Mean with Lags), then dynamically calculates the optimal operating parameters to minimize impurities.")

# Load Models safely
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_A_path = os.path.join(current_dir, 'model_lasso.joblib')
    model_A = joblib.load(model_A_path) if os.path.exists(model_A_path) else None
    return model_A

try:
    model_A = load_models()
    if model_A is not None:
        st.sidebar.success("🏆 Option A Champion Model (Lasso) Loaded!")
    else:
        st.sidebar.warning("⚠️ Option A Model File 'model_lasso.joblib' missing.")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")

# Split Dashboard layout into Control Panel and Advisor Output
col_inputs, col_outputs = st.columns([1, 2])

with col_inputs:
    st.subheader("📥 Current Plant Readings")

    # Operator Inputs
    pct_iron_feed = st.slider("% Iron Feed (Current)", 40.0, 70.0, 55.0, 0.1)
    amina_flow_current = st.slider("Amina Flow Current (m³/h)", 200.0, 800.0, 500.0, 10.0)
    ore_pulp_ph = st.slider("Ore Pulp pH", 8.0, 12.0, 10.0, 0.1)
    col_02_air = st.slider("Col 2 Air Flow", 200.0, 350.0, 250.0, 5.0)
    col_06_air = st.slider("Col 6 Air Flow", 200.0, 350.0, 250.0, 5.0)
    col_07_air = st.slider("Col 7 Air Flow", 200.0, 350.0, 250.0, 5.0)
    col_04_level = st.slider("Col 4 Level", 100.0, 800.0, 450.0, 10.0)
    col_07_level = st.slider("Col 7 Level", 100.0, 800.0, 450.0, 10.0)

    st.subheader("🕒 Process History (Lags)")
    amina_flow_lag1 = st.number_input("Amina Flow (1h Lag)", value=500.0)
    amina_flow_lag2 = st.number_input("Amina Flow (2h Lag)", value=500.0)
    silica_lag1 = st.number_input("Silica % (1h Lag)", value=2.2)
    silica_lag2 = st.number_input("Silica % (2h Lag)", value=2.1)

with col_outputs:
    st.subheader("🔮 Predictive Quality Inspection & Control Advisor")

    if model_A is None:
        st.info("Please ensure the 'model_lasso.joblib' file is deployed in the repo root directory to run predictions.")
    else: 
        # Construct standard features map
        def build_feature_row(amina_flow_val):
            features = {
                'Pct_Iron_Feed': pct_iron_feed,
                'Amina_Flow': amina_flow_val,
                'Ore_Pulp_pH': ore_pulp_ph,
                'Flotation_Column_02_Air_Flow': col_02_air,
                'Flotation_Column_06_Air_Flow': col_06_air,
                'Flotation_Column_07_Air_Flow': col_07_air,
                'Flotation_Column_04_Level': col_04_level,
                'Flotation_Column_07_Level': col_07_level
            }
            # Populate Lag Columns
            for f in list(features.keys()):
                if f == 'Amina_Flow':
                    features[f'{f}_lag_1'] = amina_flow_lag1
                    features[f'{f}_lag_2'] = amina_flow_lag2
                else:
                    features[f'{f}_lag_1'] = features[f]
                    features[f'{f}_lag_2'] = features[f]

            features['Pct_Silica_Concentrate_lag_1'] = silica_lag1
            features['Pct_Silica_Concentrate_lag_2'] = silica_lag2
            return pd.DataFrame([features])

        # Current predictions
        current_features_df = build_feature_row(amina_flow_current)
        predicted_silica = model_A.predict(current_features_df)[0]

        # Display Current State Metric
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric(
                label="Predicted % Silica Concentrate (Current Status)",
                value=f"{predicted_silica:.3f} %",
                delta=f"{predicted_silica - 2.0:.2f} % deviation (Target: 2%)",
                delta_color="inverse"
            )

        with col_metric2:
            status = "🔴 CRITICAL (Above Target)" if predicted_silica > 2.0 else "🟢 NOMINAL (Under Target)"
            st.subheader(status)

        st.markdown("--- ")
        st.subheader("💡 Operator Decision Support & Action Recommendation Engine")

        # Run optimization across hypothetical Amina Flow rates [200, 800]
        amina_scan = np.linspace(200, 800, 61)
        predictions_scan = []

        for flow in amina_scan:
            test_f_df = build_feature_row(flow)
            predictions_scan.append(model_A.predict(test_f_df)[0])

        # Find the flow rate that minimizes silica impurity under 2.0% with minimal chemical usage
        valid_flows = [flow for flow, pred in zip(amina_scan, predictions_scan) if pred < 2.0]

        if len(valid_flows) > 0:
            optimal_flow = min(valid_flows)
            opt_prediction = predictions_scan[np.where(amina_scan == optimal_flow)[0][0]]
            flow_change = optimal_flow - amina_flow_current

            st.success("### 🛠️ Optimization Suggestion Found!")
            st.markdown(
                "* **Action Recommended:** Adjust **Amina Flow** to **" + f"{optimal_flow:.1f}" + " m³/h**.\n" 
                "* **Delta Adjustment:** ` " + f"{flow_change:+.1f}" + " m³/h ` relative to current setpoint.\n" 
                "* **Target Impurity Forecast:** This adjustment is projected to bring Silica Concentrate down to **" + f"{opt_prediction:.3f}%" + "**." 
            )
        else:
            best_idx = np.argmin(predictions_scan)
            best_flow = amina_scan[best_idx]
            best_pred = predictions_scan[best_idx]
            flow_change = best_flow - amina_flow_current

            st.warning("### ⚠️ Hard Physical Constraints Encountered!")
            st.markdown(
                "Under current feed specifications, a target below 2.0% cannot be achieved purely with Amina flow.\n" 
                "* **Emergency Setpoint Recommendation:** Adjust **Amina Flow** to **" + f"{best_flow:.1f}" + " m³/h** (Change: `" + f"{flow_change:+.1f}" + " m³/h`).\n" 
                "* **Expected Outcome:** Minimizes Silica Concentrate to **" + f"{best_pred:.3f}%" + "**.\n" 
                "* **Secondary Action:** Adjust pH controller or reduce feed tonnage rate to alter inlet kinetics." 
            )

        # Optional interactive sensitivity visualization
        st.markdown("### 📊 Sensitivity Curve: Amina Flow vs. Silica Concentrate Output")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(amina_scan, predictions_scan, label="Projected Impurity Path", color="#1f77b4", linewidth=2.5)
        ax.axhline(2.0, color="red", linestyle="--", label="Upper Quality Limit (2.0%)")
        ax.axvline(amina_flow_current, color="orange", linestyle=":", label=f"Current Flow ({amina_flow_current} m³/h)")
        ax.set_xlabel("Amina Flow Setpoint (m³/h)")
        ax.set_ylabel("Predicted % Silica Concentrate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
