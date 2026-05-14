import streamlit as st

def page_hypothesis_validation_body():
    st.write("## Hypothesis and Validation")

    st.write("### Hypothesis 1 - House attributes correlate with Sale Price")

    st.info(
        "**Statement**\n\n"
        "Certain features are expected to show a measurable positive correlation "
        "with Sale Price."
    )

    st.write("**Validation Approach**")
    st.write(
        "* Compute Spearman correlation coefficients between all house "
        "features and Sale Price.\n"
        "* Produce a correlation heatmap to identify the strongest relationships at a glance.\n"
        "* Generate scatter plots of the highest-correlated continuous variables against Sale Price.\n"
        "* Hypothesis is confirmed if key features show a statistically significant "
        "correlation with Sale Price."
    )

    st.success(
        "**Confirmed** ✅\n\n"
        "The correlation analysis identified several attributes with strong positive "
        "correlations with Sale Price.  \n"
        "Two features, OverallQual and GrLivArea, show a strong correlation with SalesPrice (0.73 to 0.81).  \n"
        "Seven features (YearBuilt, GarageArea, TotalBsmtSF, GarageYrBlt, 1stFlrSF, YearRemodAdd and OpenPorchSF)"
        "showed a moderate correlation of 0.48 to 0.65."
    )

    st.write("---")

    st.write("### Hypothesis 2 - House attributes can be used to predict Sale Price")

    st.info(
        "**Statement**\n\n"
        "A machine learning regression model trained on the Ames housing dataset will "
        "accurately predict Sale Price, using the most correlated features as inputs."
    )

    st.write("**Validation Approach**")
    st.write(
        "* Select features based on findings from Hypothesis 1 correlation analysis.\n"
        "* Split the Ames dataset into training and test sets to evaluate model generalisation.\n"
        "* Train and compare regression models.\n"
        "* Evaluate model performance using R² (target ≥ 0.75) and RMSE as primary metrics.\n"
        "* Apply the final model to predict Sale Price for the four inherited properties.\n"
        "* Hypothesis is confirmed if the model achieves the R² target on unseen test data, "
        "demonstrating reliable predictive power beyond a simple baseline mean estimate."
    )

    st.success(
        "**Confirmed** ✅\n\n"
        "The model was trained on 7 of the 9 important features identified in the correlation study.  \n"
        "The Extra Trees Regressor achieved an R² of 0.880 on the test set, exceeding "
        "the 0.75 target, with a Mean Absolute Error of approximately $18,600."
    )