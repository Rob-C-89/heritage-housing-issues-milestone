import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")

from src.data_management import load_pkl_file, load_house_data

def page_sale_price_predictor_body():
    # Load predict sale price files
    inherited_predictions = pd.read_csv(f"outputs/ml_pipeline/predict_saleprice/inherited_house_predictions.csv")
    pipeline = load_pkl_file("outputs/ml_pipeline/predict_saleprice/extra_trees_pipeline.pkl")


    # State BR2
    st.write("# Sale Price Predictor")
    st.write("The client wants to predict the sale price of the four inherited properties and any other house in Ames, "
             "Iowa. A trained regression model provides an objective, data-driven prediction, giving the client a "
             "reliable basis for pricing decisions.")

    # Inherited property predictions
    st.write("## Predicted sale price for the inherited properties")
    st.write(
        "The client supplied the features of the four inherited properties. Using the prediction ML model, "
        "the following sales prices have been predicted.  \n"
    )
    st.info("**Please note**  \n"
        "The figures displayed below are a prediction, not a guarantee of final sales price. See the Model Performance "
        "page for more information."
    )
    display_inherited_predictions(inherited_predictions)

    # Live interface for predicting sales price
    st.write("## Predict Sales Price Interface")
    st.info("Input house features in the below widgets to see a predicted sale price for any given property.")

    X_live = draw_input_widgets()

    #if st.button("Run sales price prediction"):



    # House price display

def display_inherited_predictions(inherited_predictions):
    st.dataframe(
        inherited_predictions
        .set_index(inherited_predictions.columns[0])
        .style.format({
            'PredictedSalePrice': '${:,.2f}',
            'GarageArea': '{:.0f}',
            'TotalBsmtSF': '{:.0f}',
        })
    )

def draw_input_widgets():
    df = load_house_data()
    percentageMin, percentageMax = 0.4, 2.0

    X_live = pd.DataFrame([], index=[0])

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)

    with col1:
        feature = "OverallQual"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
        X_live[feature] = st_widget

    with col2:
        feature = "GrLivArea"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col3:
        feature = "KitchenQual"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
        X_live[feature] = st_widget

    with col4:
        feature = "YearBuilt"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col5:
        feature = "GarageArea"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col6:
        feature = "1stFlrSF"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col7:
        feature = "TotalBsmtSF"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    return X_live