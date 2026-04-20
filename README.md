
# Heritage Housing Issues - Data Analytics Milestone Project


## Project Overview

Heritage Housing Issues is a milestone project for the Predictive Analytics unit of Code Institute's Diploma in Full Stack Web Development.

The purpose of this project is to build a Data App with a Machine Learning User Interface (UI) combining: (1) Python packages for Machine Learning, Data Analysis and Data Visualisations; and (2) Streamlit for fast Machine Learning prototyping. This project allows the user to perform critical data analysis to generate useful insights, and deliver data-driven recommendations.


## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). 
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

### Business Requirement 1 (BR1)
The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.

### Business Requirement 2 (BR2)
The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Hypotheses and Validation

### Hypothesis 1 - House attributes correlate with Sale Price

**Statement**
Certain attributes are expected to show a measurable positive correlation with Sale Price.

**Validation Approach**
-   Compute Pearson and Spearman correlation coefficients between all house attributes and Sale Price.
-   Produce a correlation heatmap to identify the strongest relationships at a glance.
-   Generate scatter plots of the highest-correlated continuous variables against Sale Price.
- Hypothesis is confirmed if key attributes show a statistically significant correlation with Sale Price.

### Hypothesis 2 - House attributes can be used to predict Sale Price

**Statement**
A machine learning regression model trained on the Ames housing dataset will accurately predict Sale Price, using the most correlated features as inputs.

**Validation Approach**
-   Select features based on findings from Hypothesis 1 correlation analysis.
-   Split the Ames dataset into training and test sets to evaluate model generalisation
-   Train and compare regression models.
-   Evaluate model performance using R² (target ≥ 0.75) and RMSE as primary metrics.
-   Apply the final model to predict Sale Price for the four inherited properties.
-   Hypothesis is confirmed if the model achieves the R² target on unseen test data, demonstrating reliable predictive power beyond a simple baseline mean estimate.


## Rationales to map the business requirements to the Data Visualisations and ML tasks

### Business Requirement 1 — Correlation & Visualisation

**Requirement:** The client wants to understand how house attributes correlate with Sale Price in Ames, Iowa, presented through data visualisations.

**Rationale:** The client has strong property knowledge in her home region but lacks familiarity with the Ames, Iowa market. Visualising which attributes most strongly drive Sale Price in this specific dataset allows her to make informed, evidence-based decisions about how to present and position her inherited properties for sale, rather than relying on assumptions from another market.


| CRISP-DM Phase | Tasks |
|---|---|
| **Business Understanding** | Define correlation analysis and data visualisation as the goal for BR1. |
| **Data Understanding** | Collect data, and explore distributions and relationships between house attributes and Sale Price. |
| **Data Preparation** | Clean, encode and normalise data as necessary. |
| **Evaluation** | Confirm Hypothesis 1 via statistically significant correlation coefficients. |
| **Deployment** | Present correlation heatmaps and scatter plots to the client as visual deliverables. |

### Business Requirement 2 - Sale Price prediction

**Requirement:** The client wants to predict the sale price of her four inherited properties and any other house in Ames, Iowa.

**Rationale:** Correlation analysis identifies relationships but cannot produce a price estimate. A trained regression model provides an objective, data-driven prediction, giving the client a reliable basis for pricing decisions. 

| CRISP-DM Phase | Task |
|---|---|
| **Business Understanding** | Define Sale Price prediction as the analytical goal for BR2. |
| **Data Understanding** | Identify most correlated features from BR1 to inform feature selection. |
| **Data Preparation** | Handle missing values, encode categorical variables, engineer features as required. |
| **Modelling** | Train and compare regression models (e.g. Linear Regression, Random Forest, Gradient Boosting) |
| **Evaluation** | Assess model performance using R² (target ≥ 0.75) and RMSE on unseen test data; confirm Hypothesis 2 |
| **Deployment** | Apply final model to predict Sale Price for the four inherited properties and deliver an interactive prediction interface |


## ML Business Case

### Predict house Sale Price

**Regression Model**

-   We want an ML model to predict the Sale Price of residential properties in Ames, Iowa, based on historical house sale data. The target variable is continuous and numerical. We consider a regression model. It is a supervised model with a single continuous output: the predicted Sale Price in US dollars.
    
-   Our ideal outcome is to provide the client with a reliable, data-driven predicted Sale Price for each of her four inherited properties.
    
-   The model success metrics are:
    
    -   At least R² of 0.75 on both train and test sets
    -   The ML model is considered a failure if:
        -   R² falls below 0.75 on the test set, indicating the model does not generalise reliably to unseen Ames properties
        -   RMSE is disproportionately large relative to the Sale Price range ($34,900 — $755,000), suggesting predictions are too imprecise to be actionable
-   The model output is defined as a predicted Sale Price in US dollars for any given property in Ames, Iowa. The client will input house attributes via an interactive interface (Streamlit dashboard) and receive a predicted Sale Price on the fly (not in batches). The four inherited properties will be predicted as a batch at the point of delivery.
    
-   Heuristics: Currently, the client has no approach to estimate Sale Price in Ames, Iowa.
    
-   The training data to fit the model comes from the Ames Housing Dataset, which contains approximately 1,500 house sale records.
    
    -   Train data — target: SalePrice; features: all correlated house attributes identified in BR1 correlation analysis, excluding unique identifiers and variables with excessive missing values

## Dashboard Design (Streamlit App User Interface)

### Page 1: Quick project summary

-   This page will provide a quick project summary:
    -   Project Terms & Jargon
    -   Describe Project Dataset, including a link to the dataset source
    -   State Business Requirements

### Page 2: Correlation Study and Visualisation

- This page will answer business requirement 1:
	- State BR1
	- Checkbox: data inspection on house attributes and sale value, (display the number of rows and columns in the data, and display the first ten rows of the data)
	-  Display the most correlated variables to sale price and the conclusions
	- Checkbox: Heatmap of sale price and related variables.
	- Checkbox: Scatter plots of sale price and related variables.

### Page 3: House Sale Price Predictor

- This page will answer business requirement 2:
	- State BR2
	- Inherited properties predictions:
		-   Display a table of the four inherited properties with their attributes
		-   Display the predicted Sale Price for each inherited property
		-   Display the total combined predicted Sale Price for all four properties
	- Interactive house Sale Price predictor:
		- Set of widgets inputs relating to house attributes. Each set of inputs is related to a given ML task to predict sale price.
		- Display predicted house sale price

### Page 4: Project Hypothesis and Validation

- Before the analysis, we knew we wanted this page to describe each project hypothesis, the conclusions, and how we validated each. After the data analysis, we can report that:

	- 1 - House attributes correlate with Sale Price
		-  to be confirmed. Validation TBC.
	- 2 - House attributes can be used to predict Sale Price
		- to be confirmed. Validation TBC.

### Page 5: ML Model Performance

- Data preparation notes
-   Considerations and conclusions after the pipeline is trained
-   Present ML pipeline steps
-   Feature importance
- Model performance: R² and RMSE on train and test sets 
- Predicted vs Actual Sale Price plot 
-  Confirmation of whether R² ≥ 0.75 target was achieved

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the .python-version Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)


* In case you would like to thank the people that provided support through this project.

