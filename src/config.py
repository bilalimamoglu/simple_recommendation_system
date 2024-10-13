# src/config.py

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model directories
MODELS_DIR = os.path.join(BASE_DIR, 'models_saved')

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Raw data paths
TITLES_PATH = os.path.join(RAW_DATA_DIR, 'titles.csv')
INTERACTIONS_PATH = os.path.join(RAW_DATA_DIR, 'title_interactions.csv')

# Processed data paths
PROCESSED_TITLES_PATH = os.path.join(PROCESSED_DATA_DIR, 'titles_processed.csv')
PROCESSED_INTERACTIONS_PATH = os.path.join(PROCESSED_DATA_DIR, 'interactions_processed.csv')

# Model parameters
TOP_K = 10  # Number of recommendations
INTERACTION_WEIGHTS = {
    'likelist_addition': 5,
    'seenlist_addition': 4,
    'clickout_provider': 3,
    'watchlist_addition': 2
}

# Other configurations
RANDOM_STATE = 42
