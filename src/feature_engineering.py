# src/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from src import config

def process_text_features(titles_df):
    """
    Processes text features and creates a content 'SOUP' for each title.
    """
    # Clean and preprocess text data
    for col in ['GENRE_TMDB', 'DIRECTOR', 'ACTOR', 'PRODUCER', 'WRITER']:
        titles_df[col] = titles_df[col].apply(clean_list_column)

    # Combine features into a single string
    titles_df['SOUP'] = titles_df.apply(create_soup, axis=1)
    return titles_df

def clean_list_column(x):
    """
    Cleans a list column by converting all strings to lowercase and removing spaces.
    """
    return [str.lower(i.replace(" ", "")) for i in x]

def create_soup(x):
    """
    Combines all relevant features into a single string.
    """
    # Ensure ORIGINAL_TITLE is a string
    original_title = str(x['ORIGINAL_TITLE']).replace(" ", "").lower() if pd.notnull(x['ORIGINAL_TITLE']) else ''
    return ' '.join(x['GENRE_TMDB']) + ' ' + \
           ' '.join(x['DIRECTOR']) + ' ' + \
           ' '.join(x['ACTOR']) + ' ' + \
           ' '.join(x['PRODUCER']) + ' ' + \
           original_title

def create_count_matrix(titles_df, max_features=5000, n_components=100):
    """
    Creates a TF-IDF matrix from the 'SOUP' feature with limited features and applies Truncated SVD.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(titles_df['SOUP'])

    # Apply Truncated SVD to reduce dimensions
    svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_STATE)
    tfidf_matrix_svd = svd.fit_transform(tfidf_matrix)

    return tfidf_matrix_svd, vectorizer, svd



def create_interaction_matrix(interactions_df):
    """
    Creates a user-item interaction matrix with weighted interactions using a sparse matrix.
    """

    # Map user and item IDs to numerical indices
    user_codes = interactions_df['BE_ID'].astype('category').cat.codes.values
    item_codes = interactions_df['TITLE_ID'].astype('category').cat.codes.values

    interactions_df['user_code'] = user_codes
    interactions_df['item_code'] = item_codes
    interactions_df['weight'] = interactions_df['INTERACTION_TYPE'].map(config.INTERACTION_WEIGHTS)

    # Create a sparse matrix
    interaction_matrix = coo_matrix(
        (interactions_df['weight'], (interactions_df['user_code'], interactions_df['item_code']))
    )

    # Create mappings to get back original IDs
    user_id_mapping = dict(enumerate(interactions_df['BE_ID'].astype('category').cat.categories))
    item_id_mapping = dict(enumerate(interactions_df['TITLE_ID'].astype('category').cat.categories))

    return interaction_matrix, user_id_mapping, item_id_mapping
