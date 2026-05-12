import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_house_data

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")

def page_correlation_study_body():

    # Load data
    df = pd.read_csv(f"outputs/datasets/cleaned/house_prices_encoded.csv")
    spearman_corr = df.corr(method='spearman', numeric_only=True)['SalePrice'].sort_values(key=abs, ascending=False)
    top_features = spearman_corr.index[:10]

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

    # Correlation study overview
    st.write("### Correlation Summary")
    st.write(
        f" A correlation study was conducted to investigate the relationship between house features "
        f"and sale price.  \n"
        f"* Two features, OverallQual and GrLivArea, show a strong correlation with SalesPrice.  \n"
        f"* Seven features (YearBuilt, GarageArea, TotalBsmtSF, GarageYrBlt, 1stFlrSF, YearRemodAdd "
        f"and OpenPorchSF) show a moderate correlation.  \n"
        f"* There are a number of features showing a weak but potentially useful level of correlation "
        f"for the purposes of Machine Learning."
    )

    st.write("## Data Visulisation")
    
    # Heatmap
    st.write("### Heatmap Displaying Correlation Results")

    plot_heatmap(df, spearman_corr)

    # Checkbox: scatterplots
    st.write("### Scatter Plots")
    st.write(
        "Scatter plots for each of the top features have been created. These diagrams illustrate the " \
        "relationship between the features and the sale price, helping to visualise the levels of correlation."
    )

    if st.checkbox("Display scatterplots"):
        plot_scatter(df, top_features)


    # Checkbox: box plots
    st.write("### Box Plots")
    st.write(
        "To explore the relationship of the categorical variables with Sale Price, "
        "we have included 4 box plots."
    )
    if st.checkbox("Display box plots"):
        plot_box()

def plot_heatmap(df, spearman_corr):
    top_features = spearman_corr.index[:10]
    df_corr = df[top_features].corr(method='spearman')

    mask = np.zeros_like(df_corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots()
    sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, ax=ax)
    ax.set_title('Top Feature Correlations with Sale Price')
    st.pyplot(fig)

def plot_scatter(df, top_features):
    for feature in top_features[1:]:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feature, y='SalePrice')
        ax.set_title(f'{feature} vs Sale Price')
        st.pyplot(fig)

def plot_box():
    categorical_features = ['BsmtFinType1', 'KitchenQual', 'GarageFinish', 'BsmtExposure']
    df = load_house_data()

    for feature in categorical_features:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=feature, y='SalePrice')
        ax.set_title(f'{feature} vs Sale Price')
        st.pyplot(fig)
    