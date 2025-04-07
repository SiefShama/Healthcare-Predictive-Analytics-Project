import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
import os

# Load the data using relative path
@st.cache
def load_data():
    # Relative path to the file from the script's location
    file_path = os.path.join(os.getcwd(), 'data', 'Diabetic_DB_Mil2.csv')
    
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found at {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Load the data
Diabetic_DB = load_data()

# Categorical columns
categorical_columns = [
    "Diabetes_State", "PhysHlth", "Gender", "Age", "Stroke", "GenHlth",
    "CholCheck", "Smoker", "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk"
]

# Plot for categorical columns
def plot_categorical_data():
    fig = sp.make_subplots(
        rows=6, cols=2,
        subplot_titles=[f"Distribution of {col}" for col in categorical_columns],
        vertical_spacing=0.08
    )

    for i, col in enumerate(categorical_columns):
        row = i // 2 + 1
        col_pos = i % 2 + 1
        count_data = Diabetic_DB[col].value_counts().reset_index()
        count_data.columns = [col, "Count"]
        
        fig.add_trace(
            go.Bar(
                x=count_data[col],
                y=count_data["Count"],
                marker=dict(color=count_data["Count"], colorscale="RdBu"),
                name=col
            ),
            row=row, col=col_pos
        )

    fig.update_layout(
        height=2500,
        width=1000,
        title_text="Distribution of Categorical Features",
        showlegend=False
    )
    
    return fig

# Numerical columns
numerical_columns = ["HB", "Cholesterol", "BMI", "Heart_Disease", "PhysActivity", "MentHlth"]

# Plot for numerical columns
def plot_numerical_data():
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=[f"Histogram of {col}" for col in numerical_columns],
        vertical_spacing=0.1
    )

    for i, col in enumerate(numerical_columns):
        row = i // 2 + 1
        col_pos = i % 2 + 1
        
        # Drop NaNs
        data = Diabetic_DB[col].dropna()
        
        # Histogram trace
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name=f"{col} Histogram",
                marker_color='royalblue',
                opacity=0.7,
                histnorm='probability density'
            ),
            row=row, col=col_pos
        )

        # KDE line using scipy
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        y_vals = kde(x_vals)

        # KDE trace
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=f"{col} KDE",
                line=dict(color='crimson')
            ),
            row=row, col=col_pos
        )

    fig.update_layout(
        height=1200,
        width=1000,
        title_text="Histograms with KDE for Numerical Features",
        showlegend=False
    )

    return fig

# Streamlit App Layout
def main():
    st.title("Diabetes Dataset Visualizations")
    st.sidebar.header("Choose the Graphs to Display")

    # Display categorical and numerical data visualizations
    st.subheader("Categorical Feature Distributions")
    st.plotly_chart(plot_categorical_data())

    st.subheader("Numerical Feature Histograms & KDEs")
    st.plotly_chart(plot_numerical_data())

# Run the app
if __name__ == "__main__":
    main()
