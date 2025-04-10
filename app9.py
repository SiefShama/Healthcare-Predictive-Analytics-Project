import streamlit as st
import pandas as pd

# Page title
st.set_page_config(page_title="Diabetes Risk Form", layout="centered")
st.title("🩺 Diabetes Risk Input Form")
st.markdown("Fill in the details below to record your health indicators.")

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

