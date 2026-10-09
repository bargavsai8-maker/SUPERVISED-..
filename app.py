
import streamlit as st
import joblib
import numpy as np

st.title("Breast Cancer Classification")
st.write("Educational machine-learning project")

model = joblib.load("breast_cancer_model (3).pkl")

st.info("Model loaded successfully!")

st.warning(
    "For educational purposes only. "
    "This app is not a medical diagnostic tool."
)
