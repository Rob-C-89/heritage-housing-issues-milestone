import numpy as np
import streamlit as st
from src.data_management import load_house_data

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

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
             "We see that the majority of properties sold for between 100,000 and 200,000."
             "We also see a number of outliers (houses with a particularly high or low "
             "sale price, compared to the rest of the dataset)."
             "From the profile report generated in the data exploration process, and from this graph, we see that: \n"
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

    st.write("---")

    st.write("## Data Visualisation")
    
    # Heatmap
    st.write("### Heatmap")
    st.write(
        "The heatmap below helps to visualise the correlation between the top "
        "features and the sale price. \n\n"
        "Each cell shows a correlation coefficient between -1 and 1. "
        "A value closer to 1 indicates a strong positive relationship — "
        "as one variable increases, so does the other. A value closer to zero indicates "
        "litle to no relationship. \n\n"
        "The bolder the colour, the stronger the correlation (as shown in the key to the "
        "right of the graph). \n \n"
        )
    st.info(
        "The first column on the left can be read to understand the correlation between the features "
        "and Sale Price - this is the most significant column for understanding the "
        "relationships between the features and the target variable. "
        "The other columns show the relationship between the features themselves."
        )


    plot_heatmap(df, spearman_corr)

    st.write("---")

    # Checkbox: scatterplots
    st.write("### Scatter Plots")
    st.write(
        "Scatter plots for each of the top features have been created. These diagrams illustrate the " \
        "relationship between the features and the sale price, helping to visualise the levels of correlation."
        "We can see a clear linear relationship with many of the features and sale price."
    )
    st.info(
        "* A distinct linear relationship can be seen between Sale Price and OverallQual, GrLivArea, TotalBsmtSF, "
        "and 1stFlrSF. As the value increases on one axis, it increases on the other, suggesting significant correlation. \n"
        "* YearBuilt, GarageArea, GarageYrBuilt and YearRemodAdd display a weaker linear relationship - there is a general "
        "upward trend, but it is less distinct. \n"
        "* OpenPorchSF shows a weak positive relationship, with a dense vertical cluster on the left of the plot. This suggests "
        "it is a minor factor in determining sale price, and on the border of considering inclusion."
    )

    if st.checkbox("Display scatterplots"):
        plot_scatter(df, top_features)

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

    st.write("---")

    st.write('## Considerations and Conclusions')
    st.write(
        "**Considerations** \n\n"
        "* The correlation study was conducted on the raw dataset prior to data cleaning "
        "and feature engineering. Missing values were present and ignored in several variables at the "
        "time of analysis, which may have slightly influenced the correlation coefficients. \n\n"
        "* Spearman correlation was selected over Pearson as the dataset contains skewed "
        "distributions and ordinal variables, for which Spearman is the more appropriate "
        "method. \n\n"
        "* Correlation measures linear relationships - non-linear "
        "relationships between variables and Sale Price may not be fully captured "
        "by the correlation coefficients alone. \n\n"
    )
    st.write(
        "**Conclusions** \n\n"
        "* The correlation study strongly supports the hypothesis that certain house attributes "
        "show a measurable and statistically significant positive correlation with "
        "Sale Price in Ames, Iowa. \n\n"
        "* OverallQual and GrLivArea are the strongest predictors of Sale Price, "
        "with Spearman correlation coefficients of 0.81 and 0.73 respectively. \n\n"
        "* The findings from this study were used to inform the "
        "modelling phase and develop the Sale Price predictor."
    )

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
        fig = px.scatter(
            df,
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
    