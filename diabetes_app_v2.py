import streamlit as st
import joblib

# --- Title and Welcome Message ---
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("Diabetes Risk Prediction App")

st.write("""
Welcome!  
Please fill in the required information carefully.  
If you don't know your BMI or Diabetes Pedigree Function (DPF), you can calculate them easily below.

**Note:** This app provides a risk estimate and does not replace professional medical advice.
""")

# --- Model Selection Section ---
st.header("Choose a Prediction Model")

model_choice = st.selectbox(
    'Select the Machine Learning Model you want to use:',
    ("Logistic Regression", "Random Forest", "XGBoost", "KNN")
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

st.write("---")

# --- BMI Calculator Section ---
st.header("Step 1: BMI Calculator (Optional)")

weight = st.number_input('Enter your weight (in kilograms)', min_value=0.0, step=0.1)
height_cm = st.number_input('Enter your height (in centimeters)', min_value=0.0, step=0.1)

bmi = None
if weight > 0 and height_cm > 0:
    height_m = height_cm / 100  # Convert cm to meters
    bmi = weight / (height_m ** 2)
    st.success(f"✅ Your calculated BMI is: **{bmi:.2f}**")

    # BMI Category Interpretation
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

    st.info(f"💬 Your BMI category is: **{category}**")

st.write("---")

# --- Diabetes Pedigree Function (DPF) Calculator Section ---
st.header("Step 2: Diabetes Pedigree Function (DPF) Calculator (Required)")

st.write("""
**What is Diabetes Pedigree Function (DPF)?**  
- DPF estimates a person's genetic risk of diabetes.
- It considers the number of immediate family members (parents, siblings) with diabetes.
- Higher DPF values suggest a higher inherited risk.
- DPF alone does **NOT** diagnose diabetes.

👉🏽 Please complete the short family history questions below to calculate your DPF.  
👉🏽 This is **required** to proceed with risk prediction.
""")

parents_diabetes = st.selectbox('How many of your parents have diabetes?', [0, 1, 2])
siblings_diabetes = st.selectbox('How many of your siblings have diabetes?', [0, 1, '2 or more'])

# Assign simple scores
parent_score = 0
sibling_score = 0

if parents_diabetes == 1:
    parent_score = 0.3
elif parents_diabetes == 2:
    parent_score = 0.5

if siblings_diabetes == 1:
    sibling_score = 0.3
elif siblings_diabetes == '2 or more':
    sibling_score = 0.6

dpf = parent_score + sibling_score

if parent_score > 0 or sibling_score > 0:
    st.success(f"✅ Your estimated Diabetes Pedigree Function (DPF) is: **{dpf:.2f}**")
else:
    st.success(f"✅ Your estimated DPF is: **{dpf:.1f}** (No family history)")

st.write("---")

# --- Main Prediction Form ---
st.header("Step 3: Prediction Form")

st.write("Now complete the form below. If you calculated BMI and DPF above, you can use them directly!")

pregnancies = st.number_input('Number of pregnancies', min_value=0)
glucose = st.number_input('Glucose level (mg/dL)', min_value=0)
blood_pressure = st.number_input('Blood pressure level (mmHg)', min_value=0)

st.write("""
**How to measure Skin Thickness:**  
- Refers to **triceps skinfold thickness**.  
- Measured with a **skinfold caliper** at the back of the upper arm.
- Units: **millimeters (mm)**.
- Typical adult range: **10mm to 50mm**.
""")

skin_thickness = st.number_input(
    'Skin Thickness (mm)',
    min_value=0,
    help="Measured with a skinfold caliper at the back of the upper arm (triceps). Typically between 10mm and 50mm."
)

insulin = st.number_input('2-Hour Serum Insulin (μU/mL)', min_value=0)

# Use calculated BMI and DPF if available, else manual
bmi_input = st.number_input('BMI', value=bmi if bmi else 0.0)
dpf_input = st.number_input('Diabetes Pedigree Function (DPF)', value=dpf if dpf else 0.0)

age = st.number_input('Age', min_value=0)

# --- Prediction Button ---
if st.button('Predict Diabetes Risk'):
    # Validation for mandatory fields (except insulin)
    if (pregnancies == 0 or glucose == 0 or blood_pressure == 0 or
            skin_thickness == 0 or bmi_input == 0.0 or dpf_input == 0.0 or age == 0):

        st.error("🚫 Please fill all required fields correctly (except insulin) before prediction.")

    else:
        # Handle missing insulin
        if insulin == 0:
            st.warning("⚠️ Insulin value missing. Prediction will proceed but may be slightly less accurate.")
            insulin_value = -1
        else:
            insulin_value = insulin

        # Prepare user input
        user_data = [[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin_value, bmi_input, dpf_input, age]]

        # Scale input data
        user_data_scaled = scaler.transform(user_data)

        # Predict
        probability = model.predict_proba(user_data_scaled)[0][1] * 100  # Probability (%)

        st.success(f"🩺 Your predicted risk of diabetes is **{probability:.2f}%** using the **{model_choice}** model.")

        # Risk Interpretation
        if probability > 60:
            st.error("""
            🚨 **High Risk Detected!**  
            👉 Your risk of having diabetes is high.  
            👉 Immediate consultation with a healthcare provider is strongly recommended.  
            👉 Adopt urgent lifestyle changes: healthy diet, regular exercise, weight management, and regular screenings.
            """)
        else:
            st.info("""
            💬 **Moderate Risk Detected.**  
            👉 Your risk of diabetes is moderate.  
            👉 Maintain healthy habits: good diet, regular physical activity, weight control.  
            👉 Regular medical checkups are still advisable.
            """)

# --- Disclaimer ---
st.write("---")
st.caption("""
**Disclaimer:**  
This app is a predictive tool based on machine learning modeling.  
It is not a substitute for professional medical advice, diagnosis, or treatment.  
Always consult a licensed healthcare provider for your medical needs.
""")
