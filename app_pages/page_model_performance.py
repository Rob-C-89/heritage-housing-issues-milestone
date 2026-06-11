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

    # Model training notes
    st.write("### How the Model Was Trained")
    st.info(
        "**Step 1 — Data Split** \n\n"
        "The dataset of 1,460 Ames properties was split into a training set (80%) "
        "and a test set (20%). The model was trained on the training set only, "
        "with the test set held back to evaluate performance on unseen data. \n\n"
        "**Step 2 — Data Cleaning** \n\n"
        "Missing values in numerical variables were filled using the median value. "
        "Missing values in categorical variables were filled using the most frequent "
        "category (mode). \n\n"
        "**Step 3 — Algorithm Selection** \n\n"
        "Seven machine learning algorithms were evaluated using cross validation "
        "to identify the best performing model. The Extra Trees Regressor returned "
        "the highest R² score and was selected for further tuning. \n\n"
        "**Step 4 — Hyperparameter Tuning** \n\n"
        "The Extra Trees Regressor was fine-tuned using an extensive grid search "
        "to find the optimal settings. Cross validation with 5 folds was used "
        "to ensure reliable results. \n\n"
        "**Step 5 — Feature Selection** \n\n"
        "The model automatically selected the 7 most important features from the "
        "dataset to train on: OverallQual, GrLivArea, KitchenQual, YearBuilt, "
        "GarageArea, 1stFlrSF and TotalBsmtSF. \n\n"
        "**Step 6 — Evaluation** \n\n"
        "The trained model was evaluated against the held-back test set, achieving "
        "an R² score of 0.88 — exceeding the 0.75 target set in the project rationale."
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

    st.write("### Manual Testing of Model Performance")
    st.write(
        "To validate the model's performance in a real-world scenario, the first house "
        "was taken from the dataset and run through the Sale Price Predictor. "
        "The model returned a predicted sale price of \$216,187.86 against an actual "
        "sale price of \$223,500. This is an error of \$7,312, or 3.27%, demonstrating strong predictive accuracy "
        "for a typical Ames property."
)
    
    st.markdown("""
        | Feature | Value |
        | --- | --- |
        | OverallQual | 7 |
        | GrLivArea | 1786 |
        | KitchenQual | Gd |
        | YearBuilt | 2001 |
        | GarageArea | 608 |
        | 1stFlrSF | 920 |
        | TotalBsmtSF | 920 |
        | SalePrice | 223,500 |
        | **Model Predicted Price** | **216,187.86** |
""")

    st.write(
        "### Note on Alternative Model"
    )
    st.write(
        "During the modelling phase, a second model was created. \n\n"
        "This was to explore the results of a model informed by the data exploration, cleaning process, and correlation analysis." 
        "All but the top features identified in the correlation analysis "
        "were dropped, before training the model solely on these 9 features. A Gradient Boosting Regressor model was returned. \n\n"
        "The model performed well and returned an R2 score of 0.816/0.817 on the train/test sets. This suggests very consistent "
        "generalisation to unseen data, and would be perfectly acceptable given our target R2 score of 0.75. \n\n"
        "However, this underperformed when compared to the first Extra Trees Regressor model, suggesting "
        "that variables with weaker individual correlations still contributed collectively to predictive accuracy when included in the full feature set."
    )

    # Considerations and conclusions after the pipeline is trained (text)
    st.write("### Considerations and Conclusions")
    st.write(
        "The Extra Trees Regressor was selected after comparing seven regression algorithms."
    )

    st.write(
        "The selected Extra Trees Regressor model achieved an R2 score of 0.88 on the test set, exceeding the 0.75 target set "
        "in the business requirements." 
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

    st.write("### Next Steps")
    st.write(
        "The model performs well in generalising to unseen data. "
        "Next steps to improve performance might include:"
    )
    st.write(
        "* **Collect more data** — Retraining the model on a larger dataset would "
        "improve generalisation, particularly for higher value properties where the "
        "current model shows the most deviation. \n\n"
        "* **Outlier treatment** — Dropping or capping extreme Sale Price outliers "
        "and retraining to assess whether this reduces the gap between MAE and RMSE. \n\n"
        "* **Unsupervised learning** — Applying clustering to group properties into "
        "market segments (budget, mid-range, luxury) could provide the client with "
        "additional context for pricing decisions beyond a single predicted value. \n\n"
    )

def display_pipeline_steps(pipeline):
    set_config(display='diagram')
    # Streamlit function to render HTML to page
    st.components.v1.html(
    estimator_html_repr(pipeline),
    height=500,
    scrolling=True
    )
