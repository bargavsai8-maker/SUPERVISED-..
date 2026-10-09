import streamlit as st
import joblib

st.title("Breast Cancer Classification")
st.write("Educational machine-learning project")

try:
    model = joblib.load("breast_cancer_model (3).pkl")
    st.success("Model loaded successfully!")

    st.warning(
        "Educational purposes only. "
        "Not a medical diagnostic tool."
    )

    feature_names = [
        "Mean Radius", "Mean Texture", "Mean Perimeter",
        "Mean Area", "Mean Smoothness", "Mean Compactness",
        "Mean Concavity", "Mean Concave Points",
        "Mean Symmetry", "Mean Fractal Dimension",
        "Radius Error", "Texture Error", "Perimeter Error",
        "Area Error", "Smoothness Error", "Compactness Error",
        "Concavity Error", "Concave Points Error",
        "Symmetry Error", "Fractal Dimension Error",
        "Worst Radius", "Worst Texture", "Worst Perimeter",
        "Worst Area", "Worst Smoothness", "Worst Compactness",
        "Worst Concavity", "Worst Concave Points",
        "Worst Symmetry", "Worst Fractal Dimension"
    ]

    st.header("Enter Tumor Features")
    values = []

    for name in feature_names:
        values.append(
            st.number_input(
                name, min_value=0.0, value=1.0,
                key=name, format="%.6f"
            )
        )

    if st.button("Predict"):
        if getattr(model, "n_features_in_", 30) != 30:
            st.error("The model feature count is not 30.")
        else:
            prediction = model.predict([values])[0]

            if str(prediction).upper() in ["M", "1", "MALIGNANT"]:
                st.error("Model output: Malignant")
            else:
                st.success("Model output: Benign")

            st.caption(
                "Educational output only. "
                "Not a medical diagnosis."
            )

except Exception as e:
    st.error(f"Error: {e}")
