# dashboards/dashboard_app.py

import streamlit as st
import pandas as pd
from src import config
from src.data_processing import load_data, clean_titles_data
from src.feature_engineering import process_text_features, create_count_matrix
from src.models.content_based import ContentBasedRecommender

# Load and preprocess data
titles_df, _ = load_data()
titles_df = clean_titles_data(titles_df)
titles_df = process_text_features(titles_df)
count_matrix, _ = create_count_matrix(titles_df)
content_recommender = ContentBasedRecommender(titles_df, count_matrix)

# Streamlit app
st.title('Recommender System Dashboard')

# Input title ID
title_id = st.text_input('Enter Title ID (e.g., tm107473):')

if title_id:
    recommendations = content_recommender.get_recommendations(title_id)
    if not recommendations.empty:
        st.write('Top Recommendations:')
        st.table(recommendations)
    else:
        st.write('Title ID not found.')
