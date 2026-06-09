import numpy as np
import streamlit as st
from src.data_management import load_house_data

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

sns.set_style("whitegrid")

def page_correlation_study_body():

    # Load data
    df = pd.read_csv(f"outputs/datasets/collection/house_prices.csv")
    df_encoded = pd.read_csv(f"outputs/datasets/cleaned/house_prices_encoded.csv")
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

    st.write("---")

    st.write("## Data Understanding")
    st.write("Before conducting the correlation study, the data was explored and visualised to gain an "
             "understanding of the dataset as a whole. This process gives insight into the nature and spread of the variables, "
             "the extent of missing values, any alerts concerning the data, and general familiarity with the dataset."
            )   
    
    st.write("### Data Inspection")
    st.write("The first 10 rows of the dataset can be found below for illustration purposes.")

    # Checkbox: inspect data
    if st.checkbox("Inspect House Data"):
        st.write(f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        st.write(df.head(10))

    # Profile Report
    st.write("A Profile Report has been created, to visualise and help us examine the data. This report also includes "
             "a profile overview, which is helpful in drawing attention to alerts such as missing values. \n"
             "The full report can be viewed below by clicking on the checkbox. \n"
             "In summary, the report contains 18 alerts, highlighting the presence of missing values and zero values. The "
             "total missing cell count is 10.2%."
            )
    
    # Checkbox: Profile Report
    if st.checkbox("View Profile Report"):
        pandas_report = ProfileReport(df=df, minimal=True)
        st_profile_report(pandas_report)

    # Target visualisation
    st.write("### Target Visualisation")
    st.write("It is helpful to visualise the target Sales Price, to better understand the nature "
             "and distribution of this variable.\n"
             "From the profile report and this graph, we see that: \n"
             "* The majority of properties sold for between 100,000 and 200,000. \n"
             "* The mean value is $180,921. \n"
             "* The minimum is $34,900. \n"
             "* The maximum is $755,000. \n"
             "* There are no missing or zero values. \n"
             "* We also see a number of outliers (houses with a particularly high or low "
             "sale price compared to the rest of the dataset)."
             )

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='hist', y='SalePrice', bins=75, grid=True, title='Sale Price Distribution', ax=ax)
    st.pyplot(fig)

    st.write("---")

    # Correlation study overview
    st.write("### Correlation Summary")
    st.write(
        " A correlation study was conducted to investigate the relationship between house features "
        "and sale price.  \n"
        "* Two features, OverallQual and GrLivArea, show a strong correlation with SalesPrice.  \n"
        "* Seven features (YearBuilt, GarageArea, TotalBsmtSF, GarageYrBlt, 1stFlrSF, YearRemodAdd "
        "and OpenPorchSF) show a moderate correlation.  \n"
        "* There are a number of features showing a weak but potentially useful level of correlation "
        "for the purposes of Machine Learning."
    )

    st.write("---")

    st.write("## Data Visualisation")
    
    # Heatmap
    st.write("### Heatmap")
    st.write("The heatmap below helps to visualise the correlation between the top "
             "features and the sale price. The darker the colour, the stronger the "
             "correlation. \n \n"
             "It also illustrates the correlation between the features themselves,"
             "which is useful for understanding the dataset as a comprehensive whole."
            )


    plot_heatmap(df_encoded, spearman_corr)

    st.write("---")

    # Checkbox: scatterplots
    st.write("### Scatter Plots")
    st.write(
        "Scatter plots for each of the top features have been created. These diagrams illustrate the " \
        "relationship between the features and the sale price, helping to visualise the levels of correlation."
        "We can see a clear linear relationship with many of the features and sale price."
    )

    if st.checkbox("Display scatterplots"):
        plot_scatter(df_encoded, top_features)

    st.write("---")

    # Checkbox: box plots
    st.write("### Box Plots")
    st.write(
        "To explore the relationship of the categorical variables with Sale Price, "
        "we have included 4 box plots."
        "These plots are helpful for visualising the spread, or skewness, of the features."
    )
    if st.checkbox("Display box plots"):
        plot_box()

def plot_heatmap(df_encoded, spearman_corr):
    top_features = spearman_corr.index[:10]
    df_corr = df_encoded[top_features].corr(method='spearman')

    mask = np.zeros_like(df_corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots()
    sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, ax=ax)
    ax.set_title('Top Feature Correlations with Sale Price')
    st.pyplot(fig)

def plot_scatter(df_encoded, top_features):
    for feature in top_features[1:]:
        fig = px.scatter(
            df_encoded,
            x=feature,
            y='SalePrice',
            labels={'x': feature, 'y': 'SalePrice'},
            title=f'{feature} vs Sale Price'
        )
        st.plotly_chart(fig)
        
def plot_box():
    categorical_features = ['BsmtFinType1', 'KitchenQual', 'GarageFinish', 'BsmtExposure']
    df = load_house_data()

    for feature in categorical_features:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=feature, y='SalePrice')
        ax.set_title(f'{feature} vs Sale Price')
        st.pyplot(fig)
    