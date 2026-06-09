import seaborn as sns
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
        "* Feature selection was performed using SelectFromModel. The model was trained "
        "on 7 out of the 9 features identified in the correlation study. These features "
        "were selected by the Extra Trees Regressor based on their ability to reduce "
        "prediction error. \n \n"
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

    # Explanation of technical terms
    st.info(
        "**R² Score:** Measures how well the model explains the variance in sale price. "
        "A score of 1.0 is a perfect prediction. A score of 0.88 means the model explains "
        "88% of the variation in sale price. \n\n"

        "**Mean Absolute Error (MAE):** The average difference between predicted and actual "
        "sale prices. All errors are treated equally regardless of size. \n\n"
        "**Mean Squared Error (MSE):** The average of squared differences between predicted "
        "and actual sale prices. Large errors are penalised heavily due to squaring. \n\n"
        "**Root Mean Squared Error (RMSE):** Similar to MAE but larger errors are penalised "
        "more heavily. A higher RMSE relative to MAE indicates the model makes occasional "
        "larger errors on certain properties."
    )
    
    st.write(
        "In practical terms, the client can expect predictions to generally be within "
        "\$18,000 - \$30,000 of the actual sale price for a typical Ames property. This range "
        "provides a reliable basis for pricing decisions on the four inherited properties, "
        "while acknowledging some uncertainty particularly at the higher end of the "
        "price range."
)


    # Predicted vs Actual Sale Price plot (graph)
    st.write("### Predicted vs. Actual Sale Price")
    st.write(
        f"After training, the model was evaluated by withholding 20% of the housing data "
        "and comparing the model's predicted sale price against the known sale price. "
        "This unseen data was not used during training, providing an honest assessment of "
        "how well the model generalises to new properties. This is known as the test set. "
    )

    st.write(
        "The scatter plots below show the predicted sale price against the actual sale price "
        "for both the train and test sets. The red line represents a perfect prediction — "
        "points closer to the red line indicate more accurate predictions. "
    )
    st.write(
        "The train set plot shows the model performs well on data it was trained on, "
        "with predictions closely following the red line across the full price range. "
        "The test set plot shows the model generalises well to unseen data. Some deviation is visible "
        "at higher sale prices, which is expected given the smaller number of high value "
        "properties in the dataset (outliers)."
)

    st.image("outputs/plots/regression_evaluation.png")

    # Considerations and conclusions after the pipeline is trained (text)
    st.write("### Considerations and Conclusions")
    st.write(
        "The Extra Trees Regressor was selected after comparing seven regression algorithms. \n \n"
        "A second model was created, selecting only the top features identified in the correlation analysis."
        "This model underperformed when compared to the first model, so it was discarded. \n \n"
        "The selected model achieved an R2 score of 0.88 on the test set, exceeding the 0.75 target set "
        "in the business requirements." \
        "We can conclude that the model has been successful in answering "
        "the predictive task it was intended for."
    )

    st.warning(
            """
            * The R2 result of the test set (0.880) compared to the train set (0.943) indicates slight overfitting,
            but not enough to warrant further changes.

            * The RMSE of \~\$30,000 is influenced by high value outliers in the dataset. The Mean Average of \~\$18,600
            is more representative of a typical error in prediction value.

            * We can conclude that the model performs well in the context of Business Requirement 2, providing
            reliable sale price predictions for the inherited properties, and for any property in the Ames area.
            """
    )

def display_pipeline_steps(pipeline):
    set_config(display='diagram')
    # Streamlit function to render HTML to page
    st.components.v1.html(
    estimator_html_repr(pipeline),
    height=500,
    scrolling=True
    )
