import streamlit as st
import joblib

# --- Page Configuration ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# --- Load Model and Scaler ---
model_choice = st.selectbox(
    'Choose Prediction Model:',
    ("Logistic Regression", "Random Forest", "XGBoost", "KNN")
)

model_file_map = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "KNN": "knn_model.pkl"
}

model = joblib.load(model_file_map[model_choice])
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Risk Prediction App")

st.write("""
Welcome! Please complete all sections below to predict your diabetes risk.  
All fields are required except insulin (if unknown).
""")

st.write("---")

# --- BMI Calculator inside Expander ---
with st.expander("🧮 Step 1: Calculate Your Body Mass Index (BMI)"):
    weight = st.number_input('Enter your weight (in kilograms)', min_value=0.0, step=0.1)
    height_cm = st.number_input('Enter your height (in centimeters)', min_value=0.0, step=0.1)

    bmi = None
    if weight > 0 and height_cm > 0:
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)
        st.success(f"✅ Your BMI is: **{bmi:.2f}**")

        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        elif 30 <= bmi < 35:
            category = "Obesity Class I (Moderate)"
        elif 35 <= bmi < 40:
            category = "Obesity Class II (Severe)"
        else:
            category = "Obesity Class III (Very Severe)"

        st.info(f"💬 BMI Category: **{category}**")

st.write("---")

# --- DPF Calculator inside Expander ---
with st.expander("🧬 Step 2: Calculate Your Diabetes Pedigree Function (DPF)"):
    st.write("""
    **What is DPF?**  
    It estimates your genetic risk of diabetes based on family history.  
    👉🏽 Required to proceed with prediction.
    """)
    parents_diabetes = st.selectbox('How many of your parents have diabetes?', [0, 1, 2])
    siblings_diabetes = st.selectbox('How many of your siblings have diabetes?', [0, 1, '2 or more'])

    parent_score = 0.3 if parents_diabetes == 1 else (0.5 if parents_diabetes == 2 else 0)
    sibling_score = 0.3 if siblings_diabetes == 1 else (0.6 if siblings_diabetes == '2 or more' else 0)
    dpf = parent_score + sibling_score

    st.success(f"✅ Your calculated DPF is: **{dpf:.2f}**")

st.write("---")

# --- Main Prediction Form ---
st.header("📝 Step 3: Complete the Risk Prediction Form")

with st.form("prediction_form"):
    pregnancies = st.number_input('Number of pregnancies', min_value=0)
    glucose = st.number_input('Glucose level (mg/dL)', min_value=0)
    blood_pressure = st.number_input('Blood pressure (mmHg)', min_value=0)

    st.write("""
    **How to measure Skin Thickness:**  
    Measured with a skinfold caliper at the back of the upper arm (triceps).  
    Units: **millimeters (mm)**. Normal range: **10–50mm**.
    """)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0)
    insulin = st.number_input('2-Hour Serum Insulin (μU/mL)', min_value=0)

    bmi_input = st.number_input('BMI', value=bmi if bmi else 0.0)
    dpf_input = st.number_input('Diabetes Pedigree Function (DPF)', value=dpf if dpf else 0.0)
    age = st.number_input('Age', min_value=0)

    submitted = st.form_submit_button("🔎 Predict Diabetes Risk")

    if submitted:
        # Validation
        if (pregnancies == 0 or glucose == 0 or blood_pressure == 0 or
                skin_thickness == 0 or bmi_input == 0.0 or dpf_input == 0.0 or age == 0):
            st.error("🚫 Please complete all required fields before prediction.")

        else:
            if insulin == 0:
                st.warning("⚠️ Insulin value missing. Prediction will proceed but may be slightly less accurate.")
                insulin_value = -1
            else:
                insulin_value = insulin

            user_data = [[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin_value, bmi_input, dpf_input, age]]

            # Scale input
            user_data_scaled = scaler.transform(user_data)

            with st.spinner('🔄 Calculating your diabetes risk...'):
                probability = model.predict_proba(user_data_scaled)[0][1] * 100

            st.success(
                f"🩺 Your predicted risk of diabetes is **{probability:.2f}%** using the **{model_choice}** model.")

            if probability > 60:
                st.error("""
                🚨 **High Risk Detected!**  
                👉 Immediate consultation with a healthcare professional is strongly recommended.  
                👉 Adopt urgent lifestyle changes: healthy diet, regular exercise, weight control.
                """)
            else:
                st.info("""
                💬 **Moderate Risk Detected.**  
                👉 Maintain healthy lifestyle habits.  
                👉 Regular checkups are still recommended.
                """)

            st.balloons()

st.write("---")

# --- Disclaimer ---
st.caption("""
**Disclaimer:**  
This app is a predictive tool based on machine learning models.  
It is not a substitute for professional medical advice, diagnosis, or treatment.  
Always consult a licensed healthcare provider.
""")

