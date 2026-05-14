import streamlit as st

def page_summary_body():
    st.write("### Quick Project Summary")

    # Project Summary
    st.write(
        f"This project was developed for a client who has received an inheritance of four housing properties "
        f"located in Ames, Iowa. The client is looking to maximise the sales price for the properties, "
        f"utilising methods of predictive analytics and machine learning."
    )

    # Project Terms and Jargon
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* **Sale Price** - the final sale price of a house (in US dollars).\n"
        f"* **Features** - the attributes of a house used to predict its sale price.\n"
        f"* **Overall Quality (OverallQual)** - a rating of the overall material and "
        f"finish of the house, scored 1-10.\n"
        f"* **Above Grade Living Area (GrLivArea)** - the total living area in square "
        f"feet above ground level.\n"
        f"* **Total Basement Area (TotalBsmtSF)** - total square feet of basement area.\n"
        f"* **Garage Area (GarageArea)** - the size of the garage in square feet.\n"
        f"* **Year Built (YearBuilt)** - original construction date.\n"
        f"* **Kitchen Quality (KitchenQual)** - a rating of the kitchen quality "
        f"(Excellent, Good, Typical, Fair).\n"
        )
    
    # Link to README file
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Rob-C-89/heritage-housing-issues-milestone)."
        )
    
    st.write("---")

    # Business requirements
    st.success(
        f"The project has 2 business requirements:\n"

        f"* 1 - The client is interested in discovering how the house attributes "
        f"correlate with the sale price. Therefore, the client expects data visualisations "
        f"of the correlated variables against the sale price to show that.\n"

        f"* 2 - The client is interested in predicting the house sale price from her "
        f"four inherited houses and any other house in Ames, Iowa."
        )

    st.write("---")
    
    # Project source data
    st.info(
        f"**Project Source Data**\n\n"
        f"The dataset used in this project is sourced from "
        f"[Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data) "
        f"and represents records for 1460 properties from Ames, Iowa, indicating 23 features "
        f"for each property, and its respective sale price."
    )


