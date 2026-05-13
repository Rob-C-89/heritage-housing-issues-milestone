import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")

def page_sale_price_predictor_body():
    # Load predict sale price files
    inherited_predictions = pd.read_csv(f"outputs/ml_pipeline/predict_saleprice/inherited_house_predictions.csv")

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

    # Widgets to predict sale price on any given property

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