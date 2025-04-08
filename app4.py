import streamlit as st
import joblib
import os

# Define the path where models are stored
load_path = "/content/drive/MyDrive/Project_3_Healthcare/saved_models"

# Define model names
model_names_D = ["pipe_D", "regressor_D", "gnb_D", "svc_D", "knn_D", "tree_D", "rf_D", "xgb_D", 
                 "mlp_D", "logr_D", "treer_D", "rfr_D", "xgbr_D", "mlpr_D", "knnr_D"]

# Load all models
loaded_models_D = {name: joblib.load(f"{load_path}/{name}.pkl") for name in model_names_D}

# Main Title
st.title("Healthcare Prediction & Visualization Tool")

# Toggle between Plots and ML Prediction using a Radio Button
mode = st.radio("Choose Mode", ["Visualizations", "ML Predictions"], horizontal=True)

if mode == "Visualizations":
    st.subheader("📊 Visualizations")
    st.write("Here we will display plots and graphs.")  # Replace with actual visualizations
    # TODO: Add your visualization code here

else:
    st.subheader("🧠 Machine Learning Predictions")

    # Select Model
    selected_model_name = st.selectbox("Select a ML Model", options=model_names_D)
    st.write(f"Selected Model: `{selected_model_name}`")

    # User Input Form
    st.markdown("### 📝 Health Data Form")

    # Create input fields
    diabetes_state = st.selectbox("Is the individual diabetic?", options=["Yes", "No"])
    hb = st.selectbox("Does the individual follow healthy behavior?", options=["Yes", "No"])
    cholesterol = st.selectbox("Does the individual have high cholesterol?", options=["Yes", "No"])
    bmi = st.number_input("Enter BMI", min_value=13.0, max_value=41.0, step=0.1)
    heart_disease = st.selectbox("Does the individual have heart disease?", options=["Yes", "No"])
    phys_activity = st.checkbox("Does the individual engage in physical activity?")
    phys_health = st.number_input("Days of bad physical health (past month)", min_value=0, max_value=12, step=1)
    gender = st.radio("Select Gender", options=["Male", "Female"])
    age = st.number_input("Enter Age", min_value=1, max_value=95, step=1)
    stroke = st.selectbox("Has the individual had a stroke?", options=["Yes", "No"])
    gen_health = st.radio("General Health (1=Excellent, 5=Poor)", options=[1, 2, 3, 4, 5])
    chol_check = st.checkbox("Had a cholesterol check in the past 5 years?")
    smoker = st.checkbox("Is the individual a smoker?")
    fruits = st.checkbox("Consumes fruits regularly?")
    veggies = st.checkbox("Consumes vegetables regularly?")
    heavy_alcohol = st.checkbox("Consumes alcohol heavily?")
    mental_health = st.number_input("Days of bad mental health (past month)", min_value=0, max_value=5, step=1)
    diff_walk = st.checkbox("Difficulty walking?")

    # On Submit
    if st.button("Predict"):
        st.write("📡 Running prediction with selected model...")

        # Convert inputs to model-friendly format
        input_data = [
            1 if diabetes_state == "Yes" else 0,
            1 if hb == "Yes" else 0,
            1 if cholesterol == "Yes" else 0,
            bmi,
            1 if heart_disease == "Yes" else 0,
            1 if phys_activity else 0,
            phys_health,
            1 if gender == "Male" else 0,
            age,
            1 if stroke == "Yes" else 0,
            gen_health,
            1 if chol_check else 0,
            1 if smoker else 0,
            1 if fruits else 0,
            1 if veggies else 0,
            1 if heavy_alcohol else 0,
            mental_health,
            1 if diff_walk else 0,
        ]

        # Reshape input for prediction
        input_data = [input_data]

        # Load selected model
        model = loaded_models_D[selected_model_name]

        # Predict
        prediction = model.predict(input_data)

        # Show result
        st.success(f"🩺 Prediction: {prediction[0]}")
