import streamlit as st
import joblib
import numpy as np

st.title("Breast Cancer Classification")
st.write("Educational machine-learning project")

try:
    model = joblib.load("breast_cancer_model (3).pkl")
    st.success("Model loaded successfully!")

    st.warning(
        "For educational purposes only. "
        "This app is not a medical diagnostic tool."
    )

    st.header("Enter Tumor Features")

    feature_names = [
        "Mean Radius",
        "Mean Texture",
        "Mean Perimeter",
        "Mean Area",
        "Mean Smoothness",
    ]

    values = []
    for name in feature_names:
        value = st.number_input(
            name, min_value=0.0, value=1.0
        )
        values.append(value)

    if st.button("Predict"):
        if hasattr(model, "n_features_in_") and model.n_features_in_ != len(values):
            st.error(
                f"This model requires {model.n_features_in_} "
                "features. The sample form has only 5."
            )
        else:
            prediction = model.predict([values])[0]
            st.write("Model output:", prediction)
            st.info(
                "Educational output only; not a medical diagnosis."
            )

except Exception as e:
    st.error(f"Model loading or prediction error: {e}")
