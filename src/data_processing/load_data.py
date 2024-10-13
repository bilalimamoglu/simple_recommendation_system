# src/load_data.py

import pandas as pd
import numpy as np
import os
import ast
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
    titles_df['RELEASE_DURATION_DAYS'] = pd.to_numeric(titles_df['RELEASE_DURATION_DAYS'], errors='coerce')

    # Convert stringified lists into actual lists
    list_columns = ['GENRE_TMDB', 'DIRECTOR', 'ACTOR', 'PRODUCER', 'WRITER']
    for col in list_columns:
        titles_df[col] = titles_df[col].apply(str_to_list)

    # Handle less frequent categories
    titles_df = handle_less_frequent_categories(titles_df)

    # Process release dates
    titles_df = process_release_dates(titles_df)

    # Reset index
    titles_df = titles_df.reset_index(drop=True)

    return titles_df

def str_to_list(x):
    """
    Safely evaluates a string representation of a list.
    """
    try:
        return ast.literal_eval(x)
    except:
        return ['Unknown']

def handle_less_frequent_categories(titles_df):
    """
    Replaces less frequent items in DIRECTOR, ACTOR, and PRODUCER columns with 'other'.
    """
    def get_top_items(column, min_count):
        all_items = titles_df.explode(column)[column]
        item_counts = all_items.value_counts()
        top_items = item_counts[item_counts >= min_count].index.tolist()
        return top_items

    # Thresholds
    director_min_count = 5
    actor_min_count = 10
    producer_min_count = 5

    # Get top items
    top_directors = get_top_items('DIRECTOR', director_min_count)
    top_actors = get_top_items('ACTOR', actor_min_count)
    top_producers = get_top_items('PRODUCER', producer_min_count)

    # Replace less frequent items
    def replace_less_frequent(items, top_items):
        return [item if item in top_items else 'other' for item in items]

    titles_df['DIRECTOR'] = titles_df['DIRECTOR'].apply(lambda x: replace_less_frequent(x, top_directors))
    titles_df['ACTOR'] = titles_df['ACTOR'].apply(lambda x: replace_less_frequent(x, top_actors))
    titles_df['PRODUCER'] = titles_df['PRODUCER'].apply(lambda x: replace_less_frequent(x, top_producers))

    return titles_df

def process_release_dates(titles_df):
    """
    Processes 'RELEASE_DURATION_DAYS' to obtain 'RELEASE_DATE'.
    """
    # Use the maximum date from interactions as reference
    reference_date = pd.to_datetime('2024-06-30', utc=True)

    # Remove rows with NaN in 'RELEASE_DURATION_DAYS'
    titles_df = titles_df.dropna(subset=['RELEASE_DURATION_DAYS'])

    # Identify extreme values (filter out values beyond 100 years)
    threshold_days = 365 * 100  # 100 years
    titles_df = titles_df[titles_df['RELEASE_DURATION_DAYS'] <= threshold_days]

    # Convert 'RELEASE_DURATION_DAYS' to timedelta
    titles_df['RELEASE_DURATION_DAYS'] = pd.to_timedelta(titles_df['RELEASE_DURATION_DAYS'], unit='D')

    # Calculate 'RELEASE_DATE'
    titles_df['RELEASE_DATE'] = reference_date - titles_df['RELEASE_DURATION_DAYS']

    # Ensure 'RELEASE_DATE' is valid datetime
    titles_df['RELEASE_DATE'] = pd.to_datetime(titles_df['RELEASE_DATE'], errors='coerce', utc=True)

    # Drop rows with invalid 'RELEASE_DATE' values (NaT values)
    titles_df = titles_df.dropna(subset=['RELEASE_DATE'])

    return titles_df

def clean_interactions_data(interactions_df):
    """
    Cleans and preprocesses the interactions DataFrame.
    """
    # Convert timestamps to datetime objects
    interactions_df['COLLECTOR_TSTAMP'] = pd.to_datetime(
        interactions_df['COLLECTOR_TSTAMP'], errors='coerce', utc=True
    )
    # Drop rows where 'COLLECTOR_TSTAMP' could not be parsed
    interactions_df = interactions_df.dropna(subset=['COLLECTOR_TSTAMP']).reset_index(drop=True)
    return interactions_df
