import streamlit as st
import pandas as pd

# Page title
st.set_page_config(page_title="Diabetes Risk Form", layout="centered")
st.title("🩺 Diabetes Risk Input Form")
st.markdown("Fill in the details below to record your health indicators.")

# Display model performance table
st.title("🧠 Diabetes Prediction Models Comparison")

st.markdown("Below is a summary of various ML models and their performance:")

model_data = {
    "Model": ["pipe_DD", "gnb_DD", "svc_DD", "knn_DD", "tree_DD", "rf_DD", "xgb_DD", "mlp_DD", "logr_DD"],
    "Train Acc": [0.7895, 0.7875, 0.8002, 0.8327, 0.9825, 0.9825, 0.8128, 0.8079, 0.7897],
    "Test Acc": [0.7866, 0.7824, 0.7957, 0.7682, 0.6926, 0.7774, 0.8027, 0.8026, 0.7861],
    "R²": [0.1022, 0.0845, 0.1402, 0.0247, -0.2933, 0.0631, 0.1699, 0.1694, 0.1001],
    "Notes": [
        "Good generalization. Slight drop from train to test. Class 0 has better recall than class 1.",
        "Balanced performance. Slightly better class 0 precision and recall.",
        "Strong performance. No classification report provided.",
        "Overfitting suspected. Drop in test accuracy and low R².",
        "Severe overfitting. Poor generalization.",
        "High training score, lower generalization than expected.",
        "Best generalization. Strong accuracy and balance across classes.",
        "Similar to XGB. Stable and well-balanced performance.",
        "Good generalization, comparable to `pipe_DD`."
    ]
}

df_models = pd.DataFrame(model_data)
st.dataframe(df_models, use_container_width=True)

# Model selection dropdown
model_choice = st.selectbox("🔍 Select a model to use for prediction", df_models["Model"].tolist())
st.markdown(f"✅ **You selected:** `{model_choice}`")

# Top performers summary
st.markdown("""
<br>

🏆 **Top Classifier Performers**
- **XGB and MLP classifiers** show the **best overall performance**, combining high accuracy (~80%) with good precision/recall balance and generalization.
- **Logistic Regression** and **Pipeline model** (`pipe_DD`) also deliver solid results and appear stable.
- **Tree-based models** like `tree_DD` and `rf_DD` exhibit signs of **overfitting**—excellent training scores but a noticeable drop in test performance.
""", unsafe_allow_html=True)


# User input form
with st.form("diabetes_form"):
    st.subheader("Health Information")

    col1, col2 = st.columns(2)

    with col1:
        hb = st.selectbox("High Blood Pressure (HB)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        cholesterol = st.selectbox("Cholesterol", [0, 1], format_func=lambda x: "High" if x == 1 else "Normal")
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10, max_value=70)
        heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        phys_activity = st.radio("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        phys_health = st.slider("Poor Physical Health Days (Last 30 Days)", 0, 30)
        gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.number_input("Age", min_value=1, max_value=120)

    with col2:
        stroke = st.radio("Stroke History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        gen_health = st.selectbox("General Health", [1, 2, 3, 4, 5],
                                  format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x - 1])
        chol_check = st.radio("Cholesterol Checked (Last 5 Years)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        smoker = st.radio("Smoker", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        fruits = st.radio("Consumes Fruits Regularly", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        veggies = st.radio("Consumes Vegetables Regularly", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        alcohol = st.radio("Heavy Alcohol Consumption", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        ment_health = st.slider("Poor Mental Health Days (Last 30 Days)", 0, 30)
        diff_walk = st.radio("Difficulty Walking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Submit")

# Display result at bottom
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

