import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load(r"C:\Users\zhang\Desktop\combine.pkl")

# Define feature names
feature_names = [
    "TG", "TC", "HDL_C", "Hemoglobin", "CRP", "Amylase", "BUN", "Albumin",
    "Blood_Glucose", "Serum sodium", "Calcium", "ALT","L1 Muscle Density", "L1 VAT Density", 
    "L1 VAT CSA", "L2 VAT Density", "L2 IMAT Density", "L3 Muscle Density", "L3 SAT Density", "L3 VAT Density", 
    "L3 IMAT Density", "L4 Muscle Density", "L4 Muscle CSA", "L4 SAT Density", "L4 VAT Density", "L4 VAT CSA", "L5 Muscle CSA", 
    "L5 SAT Density", "L5 VAT Density", "L5 IMAT Density", "liver Mean Density", "liver Median Density",
      "L/S mean", "L/S median",  "Balthazar CT"
]

# Streamlit user interface
st.title("Organ Failure Predictor")

# Initialize feature inputs
inputs = {}

# Continuous variable inputs
for feature in feature_names:
    inputs[feature] = st.number_input(
        label=f"{feature}:",
        value=0.0,
        key=feature,
        format="%f",
        step=0.01,
        help=f"Enter the value for {feature}."
    )

# Create DataFrame from inputs
feature_values = [inputs[feat] for feat in feature_names]
features = pd.DataFrame([feature_values], columns=feature_names)

# Ensure DataFrame has correct column order and types
assert all(features.columns == feature_names), "Feature columns do not match expected order."

if st.button("Predict"):
    try:
        # Predict class and probabilities
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display prediction results
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")

        # Generate advice based on prediction results
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:
            advice = (
                f"According to our model, there is a high risk of organ failure. "
                f"The model predicts that the probability of organ failure is {probability:.1f}%. "
                "While this is just an estimate, it suggests that you may be at significant risk. "
                "We recommend consulting a healthcare professional for further evaluation "
                "and to ensure you receive an accurate diagnosis and necessary treatment."
            )
        else:
            advice = (
                f"According to our model, there is a low risk of organ failure. "
                f"The model predicts that the probability of not having organ failure is {probability:.1f}%. "
                "However, maintaining a healthy lifestyle is still very important. "
                "We recommend regular check-ups to monitor your health, "
                "and to seek medical advice promptly if you experience any symptoms."
            )

        st.write(advice)

        # Calculate SHAP values and display force plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        shap.force_plot(explainer.expected_value, shap_values[0], features, matplotlib=True)
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")