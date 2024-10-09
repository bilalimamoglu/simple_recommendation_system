# src/data_processing.py

import pandas as pd
import numpy as np
import os
from src import config

def load_data():
    """
    Loads the titles and interactions datasets.
    """
    titles = pd.read_csv(config.TITLES_PATH)
    interactions = pd.read_csv(config.INTERACTIONS_PATH)
    return titles, interactions

def load_processed_data():
    """
    Loads the preprocessed titles and interactions datasets.
    """
    titles = pd.read_csv(config.PROCESSED_TITLES_PATH)
    interactions = pd.read_csv(config.PROCESSED_INTERACTIONS_PATH)
    return titles, interactions

def save_processed_data(titles_df, interactions_df):
    """
    Saves the processed data to the processed data directory.
    """
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    titles_df.to_csv(config.PROCESSED_TITLES_PATH, index=False)
    interactions_df.to_csv(config.PROCESSED_INTERACTIONS_PATH, index=False)

def clean_titles_data(titles_df):
    """
    Cleans and preprocesses the titles DataFrame.
    """
    # Fill missing values and convert to string
    titles_df['ORIGINAL_TITLE'] = titles_df['ORIGINAL_TITLE'].fillna('').astype(str)
    titles_df['ORIGINAL_LANGUAGE'] = titles_df['ORIGINAL_LANGUAGE'].fillna('').astype(str)
    titles_df['GENRE_TMDB'] = titles_df['GENRE_TMDB'].fillna('[]')
    titles_df['DIRECTOR'] = titles_df['DIRECTOR'].fillna('[]')
    titles_df['ACTOR'] = titles_df['ACTOR'].fillna('[]')
    titles_df['PRODUCER'] = titles_df['PRODUCER'].fillna('[]')
    titles_df['WRITER'] = titles_df['WRITER'].fillna('[]')

    # Convert stringified lists into actual lists
    list_columns = ['GENRE_TMDB', 'DIRECTOR', 'ACTOR', 'PRODUCER', 'WRITER']
    for col in list_columns:
        titles_df[col] = titles_df[col].apply(eval_list_column)

    return titles_df


def eval_list_column(x):
    """
    Safely evaluates a string representation of a list.
    """
    try:
        return eval(x)
    except:
        return []

def clean_interactions_data(interactions_df):
    """
    Cleans and preprocesses the interactions DataFrame.
    """
    # Convert timestamps to datetime objects
    interactions_df['COLLECTOR_TSTAMP'] = pd.to_datetime(interactions_df['COLLECTOR_TSTAMP'])
    return interactions_df
