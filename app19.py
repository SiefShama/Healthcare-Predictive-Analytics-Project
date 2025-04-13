import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import os
import requests
import joblib
from io import BytesIO

# ------------------ Caching Functions ------------------
@st.cache_data
def load_data(dataset_version):
    file_path = os.path.join(os.getcwd(), 'data', 'Diabetic_DB_Start.csv') if dataset_version == 'Start' else os.path.join(os.getcwd(), 'data', 'Diabetic_DB_Mil2.csv')
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found at {file_path}")
        return pd.DataFrame()

@st.cache_resource(show_spinner="Loading model from Google Drive...")
def load_model_from_drive(file_id):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error("❌ Failed to load model.")
        st.exception(e)
        return None

# ------------------ Plotting Section ------------------
def plotting_section():
    st.title("📊 Diabetes Dataset Visualizations")
    dataset_version = st.sidebar.selectbox("Choose dataset version:", ["Start", "Cleaned"])
    Diabetic_DB = load_data(dataset_version)
    
    # Define the dataset schema
    data1 = {
        "Column Name": [
            "Diabetes_State", "HB", "Cholesterol", "BMI", "Heart_Disease", "PhysActivity",
            "PhysHlth", "Gender", "Age", "Stroke", "GenHlth", "CholCheck", "Smoker",
            "Fruits", "Veggies", "HvyAlcoholConsump", "MentHlth", "DiffWalk", "ID", "Race", "Weight"
        ],
        "Data Type": [
            "int64", "float64", "float64", "float64", "float64", "float64",
            "int64", "float64", "int64", "float64", "float64", "float64", "float64",
            "float64", "float64", "float64", "float64", "float64", "int64", "object", "object"
        ],
        "Description": [
            "Indicates if the individual has diabetes (`1` = Yes, `0` = No).",
            "High blood pressure level.",
            "Total cholesterol level.",
            "Body Mass Index (BMI).",
            "Indicates history of heart disease or heart attack.",
            "Whether the individual exercises regularly (`1` = Yes, `0` = No).",
            "Number of days in the past month with poor physical health.",
            "Gender (`1` = Male, `0` = Female).",
            "Age group classification.",
            "Whether the individual has had a stroke (`1` = Yes, `0` = No).",
            "Self-reported general health (`1` = Excellent, ... , `5` = Poor).",
            "Whether cholesterol was checked in the past 5 years (`1` = Yes, `0` = No).",
            "Whether the individual is a smoker (`1` = Yes, `0` = No).",
            "Whether the individual consumes fruits regularly (`1` = Yes, `0` = No).",
            "Whether the individual consumes vegetables regularly (`1` = Yes, `0` = No).",
            "Heavy alcohol consumption (`1` = Yes, `0` = No).",
            "Number of days in the past month with poor mental health.",
            "Difficulty walking or climbing stairs (`1` = Yes, `0` = No).",
            "Unique patient identifier.",
            "Patient's racial background.",
            "Weight of the patient (missing for most records)."
        ],
        "Potential Use in Analysis": [
            "Target variable for diabetes prediction models.",
            "Risk factor for diabetes and heart disease.",
            "High levels may indicate risk of diabetes or cardiovascular disease.",
            "Used to assess obesity, a risk factor for diabetes.",
            "Useful for assessing comorbidities with diabetes.",
            "Physical activity helps in managing diabetes.",
            "Indicator of overall health and chronic conditions.",
            "Used for demographic analysis in health studies.",
            "Age is a major risk factor for diabetes.",
            "Stroke risk increases with diabetes.",
            "Subjective health assessment for predictive models.",
            "Preventive health measure for cardiovascular risks.",
            "Smoking is a risk factor for diabetes complications.",
            "Affects diet-related diabetes risk.",
            "Indicator of a healthy diet.",
            "Excessive drinking can increase diabetes risk.",
            "Mental health impacts overall well-being.",
            "Indicator of mobility issues related to diabetes.",
            "Used for tracking individual records.",
            "Useful for studying disparities in diabetes prevalence.",
            "Can be used for BMI calculations if data is available."
        ]
    }
    

    # Define the dataset schema
    data2 = {
        "Column Name": [
            "Diabetes_State", "HB", "Cholesterol", "BMI", "Heart_Disease", "PhysActivity",
            "PhysHlth", "Gender", "Age", "Stroke", "GenHlth", "CholCheck", "Smoker",
            "Fruits", "Veggies", "HvyAlcoholConsump", "MentHlth", "DiffWalk"
        ],
        "Data Type": [
            "int64", "int64", "int64", "int64", "int64", "int64",
            "int64", "int64", "int64", "int64", "int64", "int64", "int64",
            "int64", "int64", "int64", "int64", "int64"
        ],
        "Description": [
            "Indicates if the individual has diabetes (`1` = Yes, `0` = No).",
            "High blood pressure level.",
            "Total cholesterol level.",
            "Body Mass Index (BMI).",
            "Indicates history of heart disease or heart attack.",
            "Whether the individual exercises regularly (`1` = Yes, `0` = No).",
            "Number of days in the past month with poor physical health.",
            "Gender (`1` = Male, `0` = Female).",
            "Age group classification.",
            "Whether the individual has had a stroke (`1` = Yes, `0` = No).",
            "Self-reported general health (`1` = Excellent, ... , `5` = Poor).",
            "Whether cholesterol was checked in the past 5 years (`1` = Yes, `0` = No).",
            "Whether the individual is a smoker (`1` = Yes, `0` = No).",
            "Whether the individual consumes fruits regularly (`1` = Yes, `0` = No).",
            "Whether the individual consumes vegetables regularly (`1` = Yes, `0` = No).",
            "Heavy alcohol consumption (`1` = Yes, `0` = No).",
            "Number of days in the past month with poor mental health.",
            "Difficulty walking or climbing stairs (`1` = Yes, `0` = No)."
        ],
        "Potential Use in Analysis": [
            "Target variable for diabetes prediction models.",
            "Risk factor for diabetes and heart disease.",
            "High levels may indicate risk of diabetes or cardiovascular disease.",
            "Used to assess obesity, a risk factor for diabetes.",
            "Useful for assessing comorbidities with diabetes.",
            "Physical activity helps in managing diabetes.",
            "Indicator of overall health and chronic conditions.",
            "Used for demographic analysis in health studies.",
            "Age is a major risk factor for diabetes.",
            "Stroke risk increases with diabetes.",
            "Subjective health assessment for predictive models.",
            "Preventive health measure for cardiovascular risks.",
            "Smoking is a risk factor for diabetes complications.",
            "Affects diet-related diabetes risk.",
            "Indicator of a healthy diet.",
            "Excessive drinking can increase diabetes risk.",
            "Mental health impacts overall well-being.",
            "Indicator of mobility issues related to diabetes."
        ]
    }

    # Convert to DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Display with Streamlit
    if dataset_version == "Start":
        st.title("📋 Dataset Schema (Start Version)")
        st.dataframe(df1)
        
        
        
    else:
        st.title("📋 Dataset Schema (CLeaned Version)")
        st.dataframe(df2)
    
    categorical_columns = [
        "Diabetes_State", "Gender", "Stroke", "GenHlth",
        "CholCheck", "Smoker", "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk",
        "HB","Cholesterol", "Heart_Disease", "PhysActivity"
    ]
    numerical_columns = [ "BMI", "MentHlth", "Age", "PhysHlth"]

    plot_type = st.sidebar.selectbox("Choose plot type:", ["Categorical", "Numerical"])

    if plot_type == "Categorical":
        selected_col = st.selectbox("Select a categorical column:", categorical_columns)
        count_data = Diabetic_DB[selected_col].value_counts().reset_index()
        count_data.columns = [selected_col, "Count"]
        fig = go.Figure(data=[go.Bar(x=count_data[selected_col], y=count_data["Count"], marker=dict(color=count_data["Count"], colorscale="RdBu"))])
        fig.update_layout(title=f"Distribution of {selected_col}", xaxis_title=selected_col, yaxis_title="Count")
        st.plotly_chart(fig)
        if dataset_version == "Start":
            
            st.markdown("### 🔍 **Overall Dataset Insights**")
            st.markdown("""

            Imbalances: Many target variables (e.g., Stroke, Heart_Disease, Diabetes_State) show strong class imbalance, which could affect predictive modeling and need handling during training.
            Balanced Features: Variables like Gender, Smoker, HB, and Cholesterol are more evenly distributed, contributing well to feature diversity.
            Lifestyle Factors: Variables like PhysActivity, Fruits, Veggies, and HvyAlcoholConsum suggest respondents report relatively healthy behaviors.

            """, unsafe_allow_html=True)
            if selected_col =="Diabetes_State":
                st.markdown("#### 🔷 **Distribution of Diabetes_State**")
                st.markdown("""
                0: Majority (~220,000+ samples) do not have diabetes.
                1: Smaller portion (~100,000 samples) have diabetes.
                Comment: There's class imbalance here. If this is the target variable for a model, class balancing techniques might be needed (e.g., SMOTE, class weights).

                """, unsafe_allow_html=True)
                
            elif selected_col == "Gender":
                st.markdown("#### 🔷 **Distribution of Gender**")
                st.markdown("""
                0 and 1 (probably Male and Female or vice versa): Both classes are well represented.
                Comment: The gender variable appears balanced, which is good for modeling to avoid gender bias.
                """, unsafe_allow_html=True)
                
            elif selected_col == "Stroke":
                st.markdown("#### 🔷 **Distribution of Stroke**")
                st.markdown("""
                0: Vast majority (~240,000) did not experience stroke.
                1: Very small portion (~10,000) had a stroke.
                Comment: Very imbalanced, potential challenge for binary classification tasks. Important for precision-recall metrics over accuracy.

                """, unsafe_allow_html=True)
                
            elif selected_col == "GenHlth":
                st.markdown("#### 🔷 **Distribution of GenHlth (General Health)**")
                st.markdown("""
                Values from 1 (Excellent) to 5 (Poor):
                2 and 3 (Good to Very Good) are most common.
                Value 5 (Poor) is the least.
                Comment: Shows a realistic distribution of self-rated health with a skew towards better health.

                """, unsafe_allow_html=True)
                
            elif selected_col == "CholCheck":
                st.markdown("#### 🔷 **Distribution of CholCheck (Cholesterol Check)**")
                st.markdown("""
                1 (Yes) dominates — most people had their cholesterol checked.
                0 (No) is very low.
                Comment: Indicates good health awareness in the dataset sample.

                """, unsafe_allow_html=True)
                
            elif selected_col == "Smoker":
                st.markdown("#### 🔷 **Distribution of Smoker**")
                st.markdown("""
                Slightly more non-smokers than smokers.
                Comment: The distribution is relatively balanced; might serve well in exploring lifestyle impacts on health.

                """, unsafe_allow_html=True)
                
                
            elif selected_col == "Fruits":
                st.markdown("#### 🔷 **Distribution of Fruits**")
                st.markdown("""
                Majority consume fruits (1).
                Minority do not (0).
                Comment: Positive lifestyle behavior; this can serve as a health feature indicating awareness.

                """, unsafe_allow_html=True)
                
                
            elif selected_col == "Veggies":
                st.markdown("#### 🔷 **Distribution of Veggies**")
                st.markdown("""
                Similar to Fruits — majority consume vegetables.
                Comment: Another lifestyle indicator with more healthy responders.
                """, unsafe_allow_html=True)
                
            elif selected_col == "HvyAlcoholConsump":
                st.markdown("#### 🔷 **Distribution of HvyAlcoholConsum (Heavy Alcohol Consumption)**")
                st.markdown("""
                0 (No): Strong majority.
                1 (Yes): Very few.
                Comment: Imbalance again; very few heavy drinkers, so feature may have limited discriminative power in models unless highly correlated.

                """, unsafe_allow_html=True)
                  
            elif selected_col == "DiffWalk":
                st.markdown("#### 🔷 **Distribution of DiffWalk (Difficulty Walking)**")
                st.markdown("""
                0: Majority have no difficulty.
                1: Minority face difficulty.
                Comment: May correlate with age, disability, or chronic disease indicators — useful for deeper analysis.
                """, unsafe_allow_html=True)
            
            elif selected_col == "HB":
                st.markdown("#### 🔷 **Distribution of HB (High Blood Pressure)**")
                st.markdown("""
                Reasonably balanced:
                Many have high blood pressure (1).
                Many don’t (0).
                Comment: Potentially a strong predictor of both heart disease and diabetes.
                """, unsafe_allow_html=True)
                
            elif selected_col == "Cholesterol":
                st.markdown("#### 🔷 **Distribution of Cholesterol**")
                st.markdown("""
                Fairly balanced:
                Slightly more individuals without high cholesterol (0).
                Many with high cholesterol (1).
                Comment: Useful for cardiovascular risk models.
                """, unsafe_allow_html=True)
                
            elif selected_col == "Heart_Disease":
                st.markdown("#### 🔷 **Distribution of Heart_Disease**")
                st.markdown("""
                0: Vast majority without heart disease.
                1: Small minority with heart disease.
                Comment: Another imbalanced target if used in prediction. Oversampling may be needed.
                """, unsafe_allow_html=True)
                
            elif selected_col == "PhysActivity":
                st.markdown("#### 🔷 **Distribution of PhysActivity (Physical Activity)**")
                st.markdown("""
                Majority engage in physical activity (1).
                Minority do not (0).
                Comment: Important lifestyle variable; inverse relationship with many chronic diseases.
                """, unsafe_allow_html=True)
                  

        else:
            st.markdown("### 🔍 **Overall Dataset Insights**")
            st.markdown("""
            The dataset reflects a generally healthy population, with most individuals free from diabetes, heart disease, or stroke. However, a notable minority has these conditions, highlighting key risk groups.
            Lifestyle habits are mixed — many show poor physical activity, low fruit/vegetable intake, and some smoke or drink heavily, all contributing to health risks.
            Health awareness is moderate, with many having checked their cholesterol, but gaps remain. General health perception is mostly positive, though some report poor health and mobility issues, possibly linked to aging or chronic illness.
            These insights emphasize the need for targeted health promotion, early screening, and lifestyle interventions.

            """, unsafe_allow_html=True)
            

            if selected_col =="Diabetes_State":
                st.markdown("#### 🔷 **Distribution of Diabetes_State**")
                st.markdown("""
                - This column represents whether a person has diabetes (1) or not (0).  
                - The bar plot likely shows that the majority of individuals do not have diabetes, with a smaller proportion being diabetic.  
                - This imbalance suggests that diabetes cases are less frequent in the dataset, which is expected in a general population dataset.

                """, unsafe_allow_html=True)
                
            elif selected_col == "Gender":
                st.markdown("#### 🔷 **Distribution of Gender**")
                st.markdown("""
                The dataset has a mix of male (0) and female (1) respondents.
                If there's an imbalance, it could indicate sampling bias where one gender is more represented than the other.
                Understanding gender distribution helps in assessing how health conditions vary between males and females.
                
                """, unsafe_allow_html=True)
                
            elif selected_col == "Stroke":
                st.markdown("#### 🔷 **Distribution of Stroke**")
                st.markdown("""
                The majority of individuals are expected to have 0 (no stroke), while a smaller percentage has 1 (had a stroke).
                The small number of stroke cases aligns with real-world data, as strokes are less frequent in the general population.
                This variable is crucial for analyzing cardiovascular risks.
                """, unsafe_allow_html=True)
                
            elif selected_col == "GenHlth":
                st.markdown("#### 🔷 **Distribution of GenHlth (General Health)**")
                st.markdown("""
                This is likely an ordinal scale (e.g., 1 = Excellent, 5 = Poor).
                If the bar plot is skewed towards the lower values (1 or 2), it suggests that most people rate their health as good.
                A significant number of people in the higher categories (4 or 5) could indicate a population with existing health concerns.
                """, unsafe_allow_html=True)
                
            elif selected_col == "CholCheck":
                st.markdown("#### 🔷 **Distribution of CholCheck (Cholesterol Check)**")
                st.markdown("""
                This binary variable (0 = No, 1 = Yes) indicates whether individuals have checked their cholesterol levels.
                If the majority fall into 1, it suggests a health-conscious population.
                If many haven't checked (0), it might indicate a need for better health screening programs.

                """, unsafe_allow_html=True)
                
            elif selected_col == "Smoker":
                st.markdown("#### 🔷 **Distribution of Smoker**")
                st.markdown("""
                The dataset likely shows a mix of smokers (1) and non-smokers (0).
                A high percentage of smokers suggests a public health concern, as smoking is linked to diabetes and cardiovascular diseases.
                This variable is crucial for understanding lifestyle risk factors.
                """, unsafe_allow_html=True)
                
                
            elif selected_col == "Fruits":
                st.markdown("#### 🔷 **Distribution of Fruits**")
                st.markdown("""
                This variable is likely binary (1 = Eats fruits regularly, 0 = Does not).
                If most people fall into 0, it suggests poor dietary habits.
                Higher fruit consumption (more 1s) is associated with better overall health.
                """, unsafe_allow_html=True)
                
                
            elif selected_col == "Veggies":
                st.markdown("#### 🔷 **Distribution of Veggies**")
                st.markdown("""
                Like the fruits variable, this indicates vegetable consumption.
                If more people report 0, it suggests a lack of proper nutrition, which could be a risk factor for diabetes.
                A higher number of 1s would be a positive indicator of healthy eating habits.
                """, unsafe_allow_html=True)
                
            elif selected_col == "HvyAlcoholConsump":
                st.markdown("#### 🔷 **Distribution of HvyAlcoholConsum (Heavy Alcohol Consumption)**")
                st.markdown("""
                This variable indicates individuals who consume alcohol heavily (1) versus those who do not (0).
                If the majority are 0, it suggests that most individuals do not engage in heavy drinking.
                Higher numbers in 1 could indicate a health concern, as excessive alcohol is linked to various diseases, including diabetes and heart conditions.
                """, unsafe_allow_html=True)
                  
            elif selected_col == "DiffWalk":
                st.markdown("#### 🔷 **Distribution of DiffWalk (Difficulty Walking)**")
                st.markdown("""
                This variable indicates if a person has mobility issues (1) or not (0).
                A high count in 1 could indicate a population with chronic illnesses or aging individuals.
                Mobility issues are often associated with diabetes complications or heart diseases.
                """, unsafe_allow_html=True)
                            
            elif selected_col == "HB":
                st.markdown("#### 🔷 **Distribution of HB (High Blood Pressure)**")
                st.markdown("""
                Hemoglobin levels typically follow a normal distribution, but the histogram might show a skewed pattern due to missing values.
                A peak in a normal range (e.g., 12–17 g/dL) suggests most individuals have healthy hemoglobin levels.
                If there are outliers (extremely low or high values), it could indicate anemia or polycythemia, both of which can be linked to diabetes complications.
                """, unsafe_allow_html=True)
                
            elif selected_col == "Cholesterol":
                st.markdown("#### 🔷 **Distribution of Cholesterol**")
                st.markdown("""
                The histogram likely shows a right-skewed distribution, meaning most people have cholesterol in the normal range but some have very high values.
                High cholesterol is a major risk factor for diabetes and heart disease.
                If many individuals have high cholesterol levels, it highlights a significant health risk in the population.

                """, unsafe_allow_html=True)
                
            elif selected_col == "Heart_Disease":
                st.markdown("#### 🔷 **Distribution of Heart_Disease**")
                st.markdown("""
                The values might be binary or continuous, representing risk scores.
                If binary (0 = No, 1 = Yes), a majority 0 indicates fewer heart disease cases, while a significant number of 1s highlights a concerning trend.
                If continuous, a right-skewed distribution suggests a high-risk population.
                """, unsafe_allow_html=True)
                
            elif selected_col == "PhysActivity":
                st.markdown("#### 🔷 **Distribution of PhysActivity (Physical Activity)**")
                st.markdown("""
                This variable may show a peak at 0 (no activity) and then a gradual decrease at higher activity levels.
                A high number of 0s suggests a sedentary lifestyle, which is a risk factor for diabetes.
                If the histogram shows a spread-out distribution, it means some individuals are highly active, while others are not.
                """, unsafe_allow_html=True)
            
            
    elif plot_type == "Numerical":
        selected_col = st.selectbox("Select a numerical column:", numerical_columns)
        data = Diabetic_DB[selected_col].dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Histogram", marker_color='royalblue', opacity=0.7, histnorm='probability density'))
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        y_vals = kde(x_vals)
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='KDE', line=dict(color='crimson')))
        fig.update_layout(title=f"Histogram with KDE for {selected_col}", xaxis_title=selected_col, yaxis_title="Density", barmode='overlay')
        st.plotly_chart(fig)
        if dataset_version == "Start":
            st.markdown("### 🧠 **Insights and Recommendations**")
            st.markdown("""

            1. **Right-Skewed Distributions:**
               - **MentHlth** and **PhysHlth** exhibit **strong right skew**, indicating most individuals report **no unhealthy days** in the past month.
               - These features highlight a population skewed toward better health, with a minority experiencing chronic or persistent health issues.

            2. **BMI (Body Mass Index):**
               - Displays a **realistic population distribution**, concentrated between **20–35**, with a notable **peak near 27–28**.
               - Presence of **extreme outliers (BMI > 50)** may affect model training unless addressed.
               - A majority fall in the **overweight or obese** category, aligning with common chronic disease risk trends.

            3. **Age Variable:**
               - Appears **categorical or bucketed** rather than continuous.
               - **Multimodal distribution** implies groupings like age bands (e.g., 18–24, 25–29, etc.).
               - This needs clarification as it directly affects encoding strategies in modeling.
            """, unsafe_allow_html=True)



            if selected_col =="BMI":
                st.markdown("#### 🔷 **Distribution of BMI (Body Mass Index)**")
                st.markdown("""
                Distribution: Right-skewed (positively skewed).
                Peak: Most data points are concentrated between 20 and 35 BMI, with a prominent peak around 27-28.
                Outliers: Few individuals have BMI above 50, extending even to 90+.
                Comment:
                    The distribution is realistic and typical for population-based health surveys.
                    BMI values above 30 typically indicate obesity — a major risk factor for diabetes and heart disease.
                    The presence of outliers may affect model performance if not handled (e.g., with normalization, clipping, or log transformation).

                """, unsafe_allow_html=True)
            elif selected_col == "MentHlth":
                st.markdown("#### 🔷 **Distribution of MentHlth (Number of Mentally Unhealthy Days in Last 30)**")
                st.markdown("""
                Distribution: Strong right skew.
                Peak at 0: A huge spike at 0 days (indicating no mentally unhealthy days for most respondents).
                Spread: Gradual decline up to 30 days. Minor peaks near 5, 10, and 30.
                Comment:
                    Most individuals report no or few mentally unhealthy days — common in health self-reports.
                    The spike at 30 days might indicate chronic or severe mental health issues for a small group.
                    This skewed nature may benefit from transformation or binning in modeling.

                """, unsafe_allow_html=True)
            elif selected_col == "Age":
                st.markdown("#### 🔷 **Distribution of Age**")
                st.markdown("""
                Distribution: Multimodal (multiple peaks).
                Strange Binning: Appears categorical or grouped by age brackets (e.g., 5-year intervals). Peaks at fixed intervals (e.g., 18, 25, 30... up to 80+).
                Peak: Highest concentration around early 20s, then steady peaks through older ages.
                Comment:
                    Age might be encoded as buckets or group codes rather than actual age (e.g., 1 = 18-24, 2 = 25-29, etc.).
                    Clarify if these are codes or exact ages — affects feature engineering.
                    Age plays a crucial role in health outcome prediction, but should be continuous or properly one-hot encoded if categorical.

                """, unsafe_allow_html=True)
            elif selected_col == "PhysHlth":
                st.markdown("#### 🔷 **Distribution of PhysHlth (Number of Physically Unhealthy Days in Last 30)**")
                st.markdown("""
                Distribution: Strong right skew — very similar to MentHlth.
                Peak at 0: Majority of the respondents report 0 physically unhealthy days.
                Tail: Long tail reaching to 30 with minor spikes along the way.
                Comment:
                    Shows that many people are in good physical health.
                    Spikes at values like 10 and 30 may be due to rounding or special cases (chronic illness).
                    Like MentHlth, this feature might benefit from grouping (e.g., 0 days, 1-5, 6-15, 16-30) in classification models or visualizations.
     
                """, unsafe_allow_html=True)
            
        else:
            st.markdown("### 🧠 **Insights and Recommendations**")
            st.markdown("""            
            The data shows trends typical of health-focused populations. 
            Most individuals have normal to slightly high BMI, with a significant portion potentially obese — reinforcing obesity’s link to diabetes. Physical and mental health are generally good, though a minority experience chronic issues, indicating potential risk groups. 
            The age distribution appears skewed toward older individuals, aligning with higher risks for chronic diseases like diabetes and heart disease. 
            These patterns suggest strong potential for predicting health outcomes using lifestyle and age-related features.

            """, unsafe_allow_html=True)
            
            if selected_col =="BMI":
                st.markdown("#### 🔷 **Distribution of BMI (Body Mass Index)**")
                st.markdown("""
                The BMI histogram is likely skewed toward higher values, as obesity is common in diabetes datasets.
                A peak in the 18.5–24.9 range indicates normal weight, while peaks in the 25–30+ range suggest overweight or obesity.
                If many individuals fall into BMI > 30, it reinforces the link between obesity and diabetes.
                """, unsafe_allow_html=True)
            elif selected_col == "MentHlth":
                st.markdown("#### 🔷 **Distribution of MentHlth (Number of Mentally Unhealthy Days in Last 30)**")
                st.markdown("""
                This histogram might be right-skewed, with most individuals reporting 0–5 mentally unhealthy days.
                If there is a significant number of people with 10+ mentally unhealthy days, it suggests mental health challenges in the population.
                Poor mental health is linked to chronic disease management, including diabetes.
                """, unsafe_allow_html=True)
            elif selected_col == "Age":
                st.markdown("#### 🔷 **Distribution of Age**")
                st.markdown("""
                The age column is encoded, possibly in categories (e.g., 1 = young adults, 9 = middle-aged, etc.).
                The bar chart might show peaks in certain age ranges, indicating a higher representation of certain age groups.
                If older age groups dominate, it could suggest a bias towards older populations, who are more likely to have health conditions.

                """, unsafe_allow_html=True)
            elif selected_col == "PhysHlth":
                st.markdown("#### 🔷 **Distribution of PhysHlth (Number of Physically Unhealthy Days in Last 30)**")
                st.markdown("""
                The distribution is likely skewed towards 0 (indicating good physical health for most people).
                A small number of individuals report higher values, suggesting they experienced many days of poor health.
                This variable could be important in predicting diabetes and heart disease, as prolonged poor physical health is often linked to chronic conditions.
                     
                """, unsafe_allow_html=True)
            

# ------------------ Prediction Section ------------------
def prediction_section():
    st.title("🩺 Diabetes Risk Prediction Form")
    st.markdown("Fill in the details below to record your health indicators.")
    
    model_type = st.sidebar.selectbox("Choose Model Type:", ["Classification", "Regression"])

    model_drive_ids_C = {
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
    
    model_drive_ids_R = {
        "regressor_DD": "1-tmgxZHBXoMjnyx8iHdzBORbHS43bCVj",
        "treer_DD": "1-zct0945l0EESuVkW51f60oFQXJC2AIt",
        "rfr_DD": "102J_kkpxhcdpbGOJzSXBcH5-g-RHVhXj",
        "xgbr_DD": "10A2tkOSszRryIa1PDbxyGefVj2w1-7aE",
        "mlpr_DD": "106ZKudJcHg--Q1OeFfQ-g1CF4Ps_KGpD",
        "knnr_DD": "102fam3U0c63bHIypOD6Wh_84obw6MPjg"
    }

    if model_type == "Classification":
        model_data = {
            "Model": list(model_drive_ids_C.keys()),
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
        
        # Summary
        st.markdown("### 🏆 **Top Classifier Performers**")
        st.markdown("""
        - **XGB & MLP**: Excellent balance of accuracy and generalization.
        - **Logistic Regression** & Pipeline: Solid results.
        - **Tree models**: Watch for overfitting.
        """, unsafe_allow_html=True)
        
        model_drive_ids = model_drive_ids_C


    else:
        model_data = {
            "Model": list(model_drive_ids_R.keys()),
            "Train R²": [0.3505, 0.9632, 0.8801, 0.4369, 0.4003, 0.5194],
            "Test R²": [0.3398, -0.2958, 0.3001, 0.3928, 0.3846, 0.2679],
            "Notes": [
                "Moderate performance. Better than basic baseline.",
                "Severe overfitting. Likely memorizing training data.",
                "Overfitting. Poor test R² compared to training.",
                "Best performing regressor. Decent generalization.",
                "Very close to XGB. Consistent and generalizable.",
                "Weaker test performance. Likely impacted by local sensitivity of KNN."
            ]
        }
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
        
        model_drive_ids = model_drive_ids_R
        
        # Analysis summary
        st.markdown("### 🏆 **Top Regressor Performers**")
        st.markdown("""
        - **XGB Regressor** (`xgbr_DD`) and **MLP Regressor** (`mlpr_DD`) show **the most stable performance** across training and test sets, with R² values near 0.39.
        - **Tree-based regressors** (`treer_DD`, `rfr_DD`) suffer from severe **overfitting**.
        - **`regressor_DD`** also performs moderately well but is outperformed by `xgbr_DD`.
        """, unsafe_allow_html=True)
    
    
    model_choice = st.selectbox("🔍 Select a model to use for prediction", df_models["Model"].tolist())
    model = load_model_from_drive(model_drive_ids[model_choice])
    st.markdown(f"✅ **You selected:** `{model_choice}`")

    with st.form("diabetes_form"):
        st.subheader("Health Information")
        col1, col2 = st.columns(2)
        with col1:
            hb = st.radio("High Blood Pressure (HB)", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cholesterol = st.radio("Cholesterol", [0, 1], format_func=lambda x: "High" if x else "Normal")
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10, max_value=70)
            heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
            phys_activity = st.radio("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x else "No")
            phys_health = st.slider("Poor Physical Health Days (Last 30 Days)", 0, 30)
            gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            age = st.number_input("Age", min_value=1, max_value=120)
        with col2:
            stroke = st.radio("Stroke History", [0, 1])
            gen_health = st.selectbox("General Health", [1, 2, 3, 4, 5], format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x - 1])
            chol_check = st.radio("Cholesterol Checked", [0, 1])
            smoker = st.radio("Smoker", [0, 1])
            fruits = st.radio("Consumes Fruits", [0, 1])
            veggies = st.radio("Consumes Vegetables", [0, 1])
            alcohol = st.radio("Heavy Alcohol Use", [0, 1])
            ment_health = st.slider("Poor Mental Health Days", 0, 30)
            diff_walk = st.radio("Difficulty Walking", [0, 1])
        submitted = st.form_submit_button("Submit")

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


        if model:
            prediction = model.predict(df)[0]
            result = "🟢 Likely Healthy" if prediction == 0 else "🔴 Likely Diabetic"
            st.success(f"🧠 Model Prediction: {result}")


            
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




# ------------------ Main App Entry ------------------
def main():
    st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
    st.sidebar.title("🔎 Select Mode")
    mode = st.sidebar.radio("Choose a view:", ["📊 Visualizations", "🧠 Prediction"])

    if mode == "📊 Visualizations":
        plotting_section()
    elif mode == "🧠 Prediction":
        prediction_section()

if __name__ == "__main__":
    main()
