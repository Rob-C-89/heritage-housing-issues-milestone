import streamlit as st
from app_pages.multipage import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_correlation_study import page_correlation_study_body
from app_pages.page_sale_price_predictor import page_sale_price_predictor_body
from app_pages.page_hypothesis_validation import page_hypothesis_validation_body
#from app_pages.page_model_performance import page_model_performance_body

app = MultiPage(app_name= "Ames House Sale Predictor")

app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Correlation Study", page_correlation_study_body)
app.add_page("House Sale Price Predictor", page_sale_price_predictor_body)
app.add_page("Hypothesis and Validation", page_hypothesis_validation_body)
#app.add_page("Model Performance", page_model_performance_body)

app.run()