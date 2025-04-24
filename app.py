import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# --- App Title ---
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Use this tool to assess the risk of diabetes based on patient health metrics.")

st.divider()

# --- Model Selection ---
st.sidebar.title("⚙️ Choose Prediction Model")
model_choice = st.sidebar.selectbox(
    "Select a machine learning model:",
    ("Random Forest", "Logistic Regression", "XGBoost", "KNN")
)

model_file_map = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "KNN": "knn_model.pkl"
}

# Load model and scaler
model = joblib.load(model_file_map[model_choice])
scaler = joblib.load("scaler.pkl")

# --- User Input Form ---
st.subheader("📋 Enter Patient Data")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 120)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", 30, 180, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 0, 1000, 85)
    bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 30.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 10, 120, 35)

# --- Predict Button ---
st.divider()
if st.button("🔍 Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\n**Confidence:** {prob:.2%}")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\n**Confidence:** {1 - prob:.2%}")

    # --- Model Info Message ---
    st.divider()
    st.markdown("### 🤖 Model Insight")
    # --- Model-Specific Insight in Sidebar ---
    st.sidebar.markdown("### 🧠 Model Insight")

    if model_choice == "Random Forest":
        st.sidebar.success(
            "Random Forest prioritizes **precision**, minimizing false positives.\n\n✅ Ideal for clinical confirmation.")
    elif model_choice == "XGBoost":
        st.sidebar.warning(
            "XGBoost emphasizes **recall**, catching more diabetics.\n\n⚠️ Great for community **screening tools**.")
    elif model_choice == "Logistic Regression":
        st.sidebar.info(
            "Logistic Regression offers **explainability** and decent overall performance.\n\n📊 Good for audit trails.")
    elif model_choice == "KNN":
        st.sidebar.info(
            "KNN uses **similar historical cases** to make predictions.\n\n🔍 Simple, interpretable, but may be less stable.")
# --- Optional Footer ---
st.divider()
st.caption("Built by Oluwafemi • Powered by Scikit-Learn & Streamlit")


