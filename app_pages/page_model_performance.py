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
    st.write("### Feature Importance")
    st.write("" \
    "The most important features for predicting the sale price of a house were: "
    "OverallQuality, GrLivArea, KitchenQual, YearBuilt, GarageArea, 1stFlrSF and TotalBsmtSF. \n \n"
    "The model was trained on these features. The diagram below displays the relative importance "
    "of each feature. \n \n")

    st.image("outputs/plots/feature_importance.png")

    # Model performance: R² and RMSE on train and test sets (table), confirmation (text)
    st.write(
        "### Model Performance  \n \n"
        "The model results from the train and test set are as follows:"
    )
    st.markdown("""
        | Metric | Train Set | Test Set |
        |---|---|---|
        | **R² Score** | 0.943 | 0.880 |
        | **Mean Absolute Error** | 12,313.615 | 18,615.871 |
        | **Mean Squared Error** | 337,456,356.691 | 923,834,197.056 |
        | **Root Mean Squared Error** | 18,369.985 | 30,394.641 |

        The R2 score from the test set is 0.88, which is well above the 0.75 defined in our rationale.

        The R2 score from the train set is 0.943, with a variance of 0.063 from the test set.
    """)

    # Predicted vs Actual Sale Price plot (graph)
    st.write(
        "### Predicted vs. Actual Sale Price  \n \n"
        "After training, we parsed 20 percent of the housing data to the model, withholding the sale price. \n \n"
        "The graph below shows the model's predicted sale price against the actual sale price on this test set." \
    )

    st.image("outputs/plots/regression_evaluation.png")

    # Considerations and conclusions after the pipeline is trained (text)

def display_pipeline_steps(pipeline):
    set_config(display='diagram')
    # Streamlit function to render HTML to page
    st.components.v1.html(
    estimator_html_repr(pipeline),
    height=500,
    scrolling=True
    )
