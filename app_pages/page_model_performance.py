from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st

from sklearn import set_config
from sklearn.utils import estimator_html_repr

from src.data_management import load_house_data, load_pkl_file
sns.set_style("whitegrid")

def page_model_performance_body():
    st.write("## Model Performance")

    # Load pipeline and data
    pipeline = load_pkl_file(
        "outputs/ml_pipeline/predict_saleprice/extra_trees_pipeline.pkl"
    )
    df = load_house_data()

    # Data Preparation notes (text)
    st.write("### Data Preparation")
    st.info(
        "* The dataset was split 80/20 into train and test sets. \n \n"
        "* Numerical variables were imputed using the median value.  \n \n"
        "* Categorical variables were imputed using the mode value, and encoded with an Ordinal Encoder.  \n \n"
        "* Standard Scaler was used for feature scaling.  \n \n"
        "* Feature selection was performed using SelectFromModel.  \n \n"
    )

    # Present ML pipeline steps (text, Ml piepline image)
    st.write("### Pipeline Steps")
    display_pipeline_steps(pipeline)

    # Feature importance (graph)

    # Model performance: R² and RMSE on train and test sets (table), confirmation (text)

    # Predicted vs Actual Sale Price plot (graph)

    # Considerations and conclusions after the pipeline is trained (text)

def display_pipeline_steps(pipeline):
    set_config(display='diagram')
    # Streamlit function to render HTML to page
    st.components.v1.html(
    estimator_html_repr(pipeline),
    height=500,
    scrolling=True
    )
