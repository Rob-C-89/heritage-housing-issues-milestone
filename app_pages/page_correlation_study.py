import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_house_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def page_correlation_study_body():

    # Load data
    df = load_house_data()

    # Correlation study introduction
    st.write("## Housing Correlation Study")
    st.write("The client wants to understand how house attributes correlate with Sale Price in Ames, Iowa, "
             "presented through data visualisations.  \n"
             "Visualising which attributes most strongly drive Sale Price in this specific dataset allows "
             "her to make informed, evidence-based decisions about how to present and position her "
             "inherited properties for sale."
             )
    
    # Checkbox: inspect data
    if st.checkbox("Inspect House Data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
            f"The first 10 rows can be found below for illustration purposes.")

        st.write(df.head(10))

    # Target visualisation
    st.write("### Target Visualisation")
    st.write("It is helpful to visualise the target Sales Price, to better understand the nature "
             "and distribution of this variable.\n"
             "We see that the majority of properties sold for between 100,000 and 200,000.")

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='hist', y='SalePrice', bins=75, grid=True, title='Sale Price Distribution', ax=ax)
    st.pyplot(fig)

    # Display most correlated values
    

    # Checkbox: heatmap

    # Checkbox: scatterplots

    # Checkbox: box plots
    