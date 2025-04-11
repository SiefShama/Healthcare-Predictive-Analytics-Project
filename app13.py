import streamlit as st
import pandas as pd
import requests
import joblib
from io import BytesIO
import traceback

# Cache the model loading function to avoid repeated downloads
@st.cache_resource(show_spinner="Loading model from Google Drive...")
def load_model_from_drive(file_id):
    """Load a model from a public Google Drive link."""
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad responses
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error("❌ Failed to load model. See details below.")
        st.exception(e)
        return None

# Mapping from model name to Drive file ID
model_drive_ids = {
    "pipe_DD": "1-xCpVwzAhJhKuU0zgch_x5aDvEvE4B0D",
    "gnb_DD": "1-pnjrekA5J-tD3sUUxFqoNSiwpO-zqhm",
    "svc_DD": "1-nI_HOuD7JYYj5mQfe0ALuPOZfMJuK_l",
    "knn_DD": "1-kgfXJDdFiwD6jnSz5Zk-4luzE6UY6Z3",
    "tree_DD": "1-jkof1CesquCSRHcrdDVwSVxiH2GU_a_",
    "rf_DD": "1-x_hd7qXWGBVjlhMV9ZkTK3p-A1R2sMH",
    "xgb_DD": "101OGOHoLmxWm1rnwgO3Dqq6Q6UVGTyuW",
    "mlp_DD": "100o9SlDylmRe3wnYearEmo1f4p6TA3Ni",
    "logr_DD": "1-zgV0vl8g1qWsQkrqB7nzZS_a_sXnvev"
}

# Page title
st.set_page_config(page_title="Diabetes Risk Form", layout="centered")
st.title("🩺 Diabetes Risk Input Form")
st.markdown("Fill in the details below to record your health indicators.")

# Display model performance table
st.title("🧠 Diabetes Prediction Models Comparison")
st.markdown("Below is a summary of various ML models and their performance:")

model_data = {
    "Model": list(model_drive_ids.keys()),
    "Train Acc": [0.7895, 0.7875, 0.8002, 0.8327, 0.9825, 0.9825, 0.8128, 0.8079, 0.7897],
    "Test Acc": [0.7866, 0.7824, 0.7957, 0.7682, 0.6926, 0.7774, 0.8027, 0.8026, 0.7861],
    "R²": [0.1022, 0.0845, 0.1402, 0.0247, -0.2933, 0.0631, 0.1699, 0.1694, 0.1001],
    "Notes": [
        "Good generalization. Slight drop from train to test.",
        "Balanced performance.",
        "Strong performance.",
        "Overfitting suspected.",
        "Severe overfitting.",
        "High training score, lower generalization.",
        "Best generalization.",
        "Similar to XGB. Stable and well-balanced.",
        "Good generalization, comparable to pipe_DD."
    ]
}
df_models = pd.DataFrame(model_data)
st.dataframe(df_models, use_container_width=True)

# Model selection dropdown
model_choice = st.selectbox("🔍 Select a model to use for prediction", df_models["Model"].tolist())
st.markdown(f"✅ **You selected:** `{model_choice}`")

# Summary
st.markdown("""
🏆 **Top Performers**
- XGB & MLP: Excellent balance of accuracy and generalization.
- Logistic Regression & Pipeline: Solid results.
- Tree models: Watch for overfitting.
""", unsafe_allow_html=True)

# User input form
with st.form("diabetes_form"):
    st.subheader("Health Information")

    col1, col2 = st.columns(2)

    with col1:
        hb = st.selectbox("High Blood Pressure (HB)", [0, 1], format_func=lambda x: "Yes" if x else "No")
        cholesterol = st.selectbox("Cholesterol", [0, 1], format_func=lambda x: "High" if x else "Normal")
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10, max_value=70)
        heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
        phys_activity = st.radio("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x else "No")
        phys_health = st.slider("Poor Physical Health Days (Last 30 Days)", 0, 30)
        gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.number_input("Age", min_value=1, max_value=120)

    with col2:
        stroke = st.radio("Stroke History", [0, 1], format_func=lambda x: "Yes" if x else "No")
        gen_health = st.selectbox("General Health", [1, 2, 3, 4, 5],
                                  format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x - 1])
        chol_check = st.radio("Cholesterol Checked (Last 5 Years)", [0, 1], format_func=lambda x: "Yes" if x else "No")
        smoker = st.radio("Smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")
        fruits = st.radio("Consumes Fruits Regularly", [0, 1], format_func=lambda x: "Yes" if x else "No")
        veggies = st.radio("Consumes Vegetables Regularly", [0, 1], format_func=lambda x: "Yes" if x else "No")
        alcohol = st.radio("Heavy Alcohol Consumption", [0, 1], format_func=lambda x: "Yes" if x else "No")
        ment_health = st.slider("Poor Mental Health Days (Last 30 Days)", 0, 30)
        diff_walk = st.radio("Difficulty Walking", [0, 1], format_func=lambda x: "Yes" if x else "No")

    submitted = st.form_submit_button("Submit")

# Display and predict
if submitted:
    st.markdown("---")
    st.subheader("📋 Submitted Information")

    user_data = {
        "HB": hb,
        "Cholesterol": cholesterol,
        "BMI": bmi,
        "Heart_Disease": heart_disease,
        "PhysActivity": phys_activity,
        "PhysHlth": phys_health,
        "Gender": gender,
        "Age": age,
        "Stroke": stroke,
        "GenHlth": gen_health,
        "CholCheck": chol_check,
        "Smoker": smoker,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": alcohol,
        "MentHlth": ment_health,
        "DiffWalk": diff_walk
    }

    df = pd.DataFrame([user_data])
    st.dataframe(df)
    st.success("✅ Your input has been recorded!")

    user_input = [[
        hb, cholesterol, bmi, heart_disease, phys_activity,
        phys_health, gender, age, stroke, gen_health,
        chol_check, smoker, fruits, veggies, alcohol,
        ment_health, diff_walk
    ]]

    model_id = model_drive_ids.get(model_choice)

    if model_id:
        model = load_model_from_drive(model_id)
        if model:
            try:
                prediction = model.predict(user_input)[0]
                pred_label = "Diabetic" if prediction == 1 else "Non-Diabetic"
                st.success(f"🧾 **Prediction Result:** {pred_label}")
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(user_input)[0][1]
                    st.info(f"📊 Probability of being diabetic: **{prob:.2%}**")
            except Exception as e:
                st.error("❌ Prediction failed. See error below.")
                st.exception(e)
        else:
            st.error("❌ Model could not be loaded.")
    else:
        st.error("❌ Selected model ID is missing.")
