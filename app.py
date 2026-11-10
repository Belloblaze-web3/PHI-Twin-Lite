from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from generate_data import FAULTS, extract_features, simulate_one

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

st.set_page_config(page_title="PHI-Twin | Landing Gear Digital Twin", layout="wide")

st.title("PHI-Twin: Landing Gear Digital Twin")
st.caption("Physics Hybrid Integrity Lab — synthetic landing-gear condition-monitoring demonstration")

if not MODEL_PATH.exists():
    st.error("Model file not found. Train it first by running: python train.py")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(MODEL_PATH))

with st.sidebar:
    st.header("Simulation controls")
    selected_name = st.selectbox("Injected synthetic scenario", list(FAULTS.values()))
    selected_fault_id = next(key for key, value in FAULTS.items() if value == selected_name)
    run = st.button("Simulate landing event", type="primary", use_container_width=True)
    st.divider()
    st.caption("Scenarios and readings are generated from a simplified physics model. This is not certified aircraft diagnostic software.")

if "event" not in st.session_state or run:
    st.session_state.event = simulate_one(selected_fault_id)
    st.session_state.event_fault_id = selected_fault_id

t, displacement, pressure, zeta, wn = st.session_state.event
features = extract_features(displacement, pressure, zeta, wn)
X = pd.DataFrame([features]).reindex(columns=model.feature_names_in_)
prediction = int(model.predict(X)[0])
probabilities = model.predict_proba(X)[0]
class_ids = [int(item) for item in model.classes_]
prob_by_id = dict(zip(class_ids, probabilities))
confidence = float(np.max(probabilities))
health = float(prob_by_id.get(0, 0.0) * 100)

col1, col2, col3 = st.columns(3)
col1.metric("Predicted condition", FAULTS.get(prediction, str(prediction)))
col2.metric("Model confidence", f"{confidence * 100:.1f}%")
col3.metric("Normal-class probability", f"{health:.1f}%")
st.caption(f"Synthetic scenario injected: **{FAULTS[st.session_state.event_fault_id]}**. A scenario label and model prediction are distinct; mismatches can occur.")

left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=displacement, name="Strut displacement", line={"color": "#2563eb"}))
    fig.update_layout(title="Simulated oleo strut response", xaxis_title="Time (s)", yaxis_title="Displacement (m)", template="plotly_white", margin={"l": 10, "r": 10, "t": 50, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
with right:
    pressure_fig = go.Figure()
    pressure_fig.add_trace(go.Scatter(x=t, y=pressure, name="Pressure proxy", line={"color": "#0f766e"}))
    pressure_fig.update_layout(title="Simulated hydraulic pressure proxy", xaxis_title="Time (s)", yaxis_title="Pressure (MPa proxy)", template="plotly_white", margin={"l": 10, "r": 10, "t": 50, "b": 10})
    st.plotly_chart(pressure_fig, use_container_width=True)

st.subheader("Extracted physics features")
st.dataframe(pd.DataFrame({"Value": features}).T, use_container_width=True)

st.subheader("Class probabilities")
prob_df = pd.DataFrame({"Condition": [FAULTS.get(cid, str(cid)) for cid in class_ids], "Probability": probabilities * 100}).sort_values("Probability", ascending=False)
st.bar_chart(prob_df.set_index("Condition"))

if prediction == 0:
    st.success("The model classified this synthetic event as Normal.")
else:
    st.warning(f"The model classified this synthetic event as {FAULTS.get(prediction, prediction)}. In real operations, decisions require validated sensor data and qualified engineering review.")

st.info("All data here are simulated. The health index is only the model's probability for the Normal class; no remaining-useful-life estimate is provided because the model is not trained for RUL.")
