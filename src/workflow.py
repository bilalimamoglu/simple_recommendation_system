# src/workflow.py

import logging
import os
import pandas as pd
from surprise import Dataset, Reader
from sklearn.model_selection import StratifiedShuffleSplit
import random
from tqdm import tqdm
import numpy as np

from src.data_processing.load_data import load_data
from src.data_processing.preprocess import clean_titles_data, clean_interactions_data
from src.feature_engineering.feature_engineer import (
    process_text_features,
    create_count_matrix,
    create_weighted_count_matrix
)
from src.models.content_based import ContentBasedRecommender, WeightedContentBasedRecommender
from src.models.collaborative_filtering import CollaborativeFilteringRecommender
from src.models.hybrid_model import HybridRecommender
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f_score_at_k,
    ndcg_at_k,
    mean_average_precision,
    area_under_roc_curve
)
from app.utils import save_model

from src import config


def load_and_preprocess_data(args):
    """
    Loads and preprocesses the raw data.

    Parameters:
    - args: Parsed command-line arguments.

    Returns:
    - titles_df: Preprocessed titles DataFrame.
    - interactions_df: Preprocessed interactions DataFrame.
    """
    logging.info('Loading raw data...')
    titles_df, interactions_df = load_data()

    logging.info('Preprocessing data...')
    titles_df = clean_titles_data(titles_df)
    interactions_df = clean_interactions_data(interactions_df)

    # Save processed data
    logging.info('Saving processed data...')
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    titles_df.to_csv(config.PROCESSED_TITLES_PATH, index=False)
    interactions_df.to_csv(config.PROCESSED_INTERACTIONS_PATH, index=False)

    return titles_df, interactions_df


def perform_feature_engineering(titles_df, args):
    """
    Performs feature engineering including TF-IDF vectorization and dimensionality reduction.

    Parameters:
    - titles_df: Preprocessed titles DataFrame.
    - args: Parsed command-line arguments.

    Returns:
    - tfidf_matrix_svd: Reduced TF-IDF matrix for standard content-based recommender.
    - weighted_tfidf_matrix_svd: Reduced weighted TF-IDF matrix for weighted content-based recommender.
    """
    logging.info('Performing feature engineering...')
    titles_df = process_text_features(titles_df)

    # Create standard TF-IDF matrix
    tfidf_matrix_svd, vectorizer, svd = create_count_matrix(titles_df, max_features=args.max_features)

    # Create weighted TF-IDF matrix
    weights = {
        'ORIGINAL_TITLE': args.title_weight,
        'GENRE_TMDB': args.genre_weight,
        'DIRECTOR': args.director_weight,
        'ACTOR': args.actor_weight,
        'PRODUCER': args.producer_weight
    }
    weighted_tfidf_matrix_svd, weighted_vectorizers, weighted_svd = create_weighted_count_matrix(
        titles_df, weights, max_features=args.max_features, n_components=100
    )

    return tfidf_matrix_svd, weighted_tfidf_matrix_svd


def initialize_and_train_recommenders(titles_df, interactions_df, tfidf_matrix_svd, weighted_tfidf_matrix_svd, args):
    """
    Initializes and trains all recommender systems.

    Parameters:
    - titles_df: Preprocessed titles DataFrame.
    - interactions_df: Preprocessed interactions DataFrame.
    - tfidf_matrix_svd: Reduced TF-IDF matrix for standard content-based recommender.
    - weighted_tfidf_matrix_svd: Reduced weighted TF-IDF matrix for weighted content-based recommender.
    - args: Parsed command-line arguments.

    Returns:
    - recommenders: Dictionary containing all trained recommender instances.
    - sampled_users: List of user IDs selected for evaluation.
    """
    # Map TITLE_ID to ORIGINAL_TITLE
    title_id_to_title = dict(zip(titles_df['TITLE_ID'], titles_df['ORIGINAL_TITLE']))

    # Map interaction types to ratings
    interactions_df['rating'] = interactions_df['INTERACTION_TYPE'].map(config.INTERACTION_WEIGHTS)
    interactions_df['rating'] = interactions_df['rating'].fillna(0)

    # Ensure COLLECTOR_TSTAMP is datetime
    interactions_df['COLLECTOR_TSTAMP'] = pd.to_datetime(
        interactions_df['COLLECTOR_TSTAMP'],
        utc=True,
        errors='coerce'
    )

    # Handle NaT values
    interactions_df = interactions_df.dropna(subset=['COLLECTOR_TSTAMP']).reset_index(drop=True)
    logging.info(f"Number of data points after dropping invalid timestamps: {len(interactions_df)}")

    # Sort interactions by timestamp
    interactions_df = interactions_df.sort_values('COLLECTOR_TSTAMP').reset_index(drop=True)

    # Time-wise train-test split
    logging.info('Splitting data into train and test sets chronologically...')
    cutoff_date = pd.to_datetime('2024-06-30', utc=True)
    train_df_full = interactions_df[interactions_df['COLLECTOR_TSTAMP'] < cutoff_date].reset_index(drop=True)
    test_df = interactions_df[interactions_df['COLLECTOR_TSTAMP'] >= cutoff_date].reset_index(drop=True)

    logging.info(f"Number of training data points: {len(train_df_full)}")
    logging.info(f"Number of test data points: {len(test_df)}")

    # Apply sample percentage for training data
    if args.sample_percentage < 100.0:
        logging.info(f"Sampling {args.sample_percentage}% of training data...")
        train_df_sampled = train_df_full.sample(frac=args.sample_percentage / 100.0,
                                                random_state=config.RANDOM_STATE).reset_index(drop=True)
    else:
        train_df_sampled = train_df_full.copy()

    # Log number of data points after sampling
    logging.info(f"Number of training data points after sampling: {len(train_df_sampled)}")

    # Identify common users present in both training and test sets
    logging.info('Identifying common users present in both training and test sets...')
    common_users = set(train_df_sampled['BE_ID']).intersection(set(test_df['BE_ID']))
    logging.info(f"Number of common users for evaluation: {len(common_users)}")

    if not common_users:
        logging.error(
            "No common users found between training and test sets. Adjust the sample percentage or check your data.")
        return {}, []

    # Perform stratified sampling among common users to select 1000 users
    logging.info('Applying stratified sampling to select 1000 users for evaluation...')
    # Filter training data to include only common users
    train_common = train_df_sampled[train_df_sampled['BE_ID'].isin(common_users)].reset_index(drop=True)

    # Calculate interaction counts per user
    user_interaction_counts = train_common.groupby('BE_ID').size().reset_index(name='interaction_count')

    # Define number of strata (e.g., 5 bins)
    n_strata = 5
    try:
        user_interaction_counts['interaction_bin'] = pd.qcut(user_interaction_counts['interaction_count'], q=n_strata,
                                                             labels=False, duplicates='drop')
    except ValueError as e:
        logging.warning(f"Stratified sampling encountered an issue: {e}. Falling back to uniform sampling.")
        sampled_users = random.sample(list(common_users), min(1000, len(common_users)))
    else:
        # Initialize StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=min(1000, len(common_users)), random_state=config.RANDOM_STATE)
        for train_idx, test_idx in splitter.split(user_interaction_counts, user_interaction_counts['interaction_bin']):
            sampled_users = user_interaction_counts.iloc[test_idx]['BE_ID'].tolist()

    logging.info(f"Number of users selected for evaluation: {len(sampled_users)}")

    # Filter test set to include only interactions from sampled users
    test_df_filtered = test_df[test_df['BE_ID'].isin(sampled_users)].reset_index(drop=True)
    logging.info(f"Number of test data points after filtering sampled users: {len(test_df_filtered)}")

    # Use the Reader class to parse the DataFrame
    reader = Reader(rating_scale=(interactions_df['rating'].min(), interactions_df['rating'].max()))

    # Build dataset for collaborative filtering
    train_data = Dataset.load_from_df(
        train_df_sampled[['BE_ID', 'TITLE_ID', 'rating']],
        reader
    )

    # Create mappings for indices
    title_id_to_idx = pd.Series(titles_df.index, index=titles_df['TITLE_ID']).drop_duplicates()
    idx_to_title_id = pd.Series(titles_df['TITLE_ID'].values, index=titles_df.index)

    # Initialize recommenders dictionary
    recommenders = {}

    # Initialize and train standard content-based recommender
    logging.info('Initializing standard content-based recommender...')
    content_recommender = ContentBasedRecommender(titles_df, tfidf_matrix_svd, idx_to_title_id)
    content_recommender.train()
    recommenders['content_based'] = content_recommender

    # Initialize and train weighted content-based recommender
    logging.info('Initializing weighted content-based recommender...')
    weighted_content_recommender = WeightedContentBasedRecommender(titles_df, weighted_tfidf_matrix_svd,
                                                                   idx_to_title_id)
    weighted_content_recommender.train()
    recommenders['content_based_weighted'] = weighted_content_recommender

    # Define Algorithm Type Mappings
    user_based_algorithms = ['SVD', 'SVDpp', 'NMF']
    item_based_algorithms = ['KNNBaseline', 'KNNBasic', 'KNNWithMeans']

    # Initialize and train collaborative filtering recommenders
    collaborative_recommenders = []
    for algo in args.algorithms.split(','):
        algo = algo.strip()
        if algo in user_based_algorithms:
            user_based = True
        elif algo in item_based_algorithms:
            user_based = False
        else:
            logging.warning(f"Algorithm '{algo}' is not recognized. Skipping.")
            continue

        logging.info(
            f'Initializing {"user-based" if user_based else "item-based"} collaborative filtering recommender with algorithm: {algo}')
        collaborative_recommender = CollaborativeFilteringRecommender(
            data=train_data,  # Pass the Dataset object instead of trainset
            algorithm=algo,
            algo_params={'random_state': config.RANDOM_STATE},
            user_based=user_based
        )
        collaborative_recommender.train()
        collaborative_recommenders.append(collaborative_recommender)
        recommenders[f'collaborative_{algo}'] = collaborative_recommender

    # Initialize and train hybrid recommenders (for all CF algorithms)
    hybrid_recommenders = []
    for collaborative_recommender in collaborative_recommenders:
        logging.info('Initializing hybrid recommender...')
        hybrid_recommender = HybridRecommender(
            content_recommender=content_recommender,
            collaborative_recommender=collaborative_recommender,
            alpha=args.alpha
        )
        hybrid_recommender.train()
        hybrid_recommenders.append(hybrid_recommender)
        recommenders[f'hybrid_{collaborative_recommender.algorithm_name}'] = hybrid_recommender

    return recommenders, sampled_users


def evaluate_recommenders(recommenders, interactions_df, titles_df, sampled_users, args):
    """
    Evaluates all recommenders and aggregates the results.

    Parameters:
    - recommenders: Dictionary containing all trained recommender instances.
    - interactions_df: Preprocessed interactions DataFrame.
    - titles_df: Preprocessed titles DataFrame.
    - sampled_users: List of user IDs to evaluate.
    - args: Parsed command-line arguments.

    Returns:
    - results_df: DataFrame containing user-based evaluation metrics for all recommenders.
    - algorithm_metrics_df: DataFrame containing Prediction and Catalogue Coverage per algorithm.
    - title_metrics_dfs: Dictionary containing DataFrames for each specified title.
    """
    logging.info("Starting evaluation of recommenders...")

    # Filter test data for sampled users
    cutoff_date = pd.to_datetime('2024-06-30', utc=True)
    test_df_filtered = interactions_df[
        (interactions_df['BE_ID'].isin(sampled_users)) &
        (interactions_df['COLLECTOR_TSTAMP'] >= cutoff_date)
    ].reset_index(drop=True)

    # Prepare item popularity for coverage calculation
    all_items = set(titles_df['TITLE_ID'].unique())

    # Prepare evaluation results list
    results = []

    # Prepare a dictionary to hold coverage metrics per algorithm
    algorithm_coverage = {model_key: {'Prediction Coverage': 0.0, 'Catalogue Coverage': 0.0} for model_key in recommenders.keys()}

    # To collect all recommended items per algorithm for catalogue coverage
    recommended_items_per_algorithm = {model_key: set() for model_key in recommenders.keys()}

    # To hold metrics per title
    title_ids = ['tm107473', 'tm50355', 'ts89259']
    title_metrics = {title_id: [] for title_id in title_ids}

    for model_key, recommender in recommenders.items():
        logging.info(f"Evaluating {model_key}...")
        for user_id in tqdm(sampled_users, desc=f"Evaluating {model_key}"):
            # Get the items the user interacted with in the test set
            user_test_items = test_df_filtered[test_df_filtered['BE_ID'] == user_id]['TITLE_ID'].tolist()

            if not user_test_items:
                continue  # Skip users with no test interactions

            # Get the user's training items
            train_user_items = interactions_df[
                (interactions_df['BE_ID'] == user_id) &
                (interactions_df['COLLECTOR_TSTAMP'] < cutoff_date)
            ]['TITLE_ID'].tolist()

            # Get recommendations
            if isinstance(recommender, ContentBasedRecommender) or isinstance(recommender, WeightedContentBasedRecommender):
                identifier_type = 'title'
                if train_user_items:
                    representative_item = train_user_items[-1]
                    recommendations = recommender.get_recommendations(
                        title_id=representative_item,
                        top_k=args.top_k,
                        exclude_items=train_user_items,
                        identifier_type=identifier_type
                    )
                else:
                    recommendations = []
            elif isinstance(recommender, CollaborativeFilteringRecommender):
                identifier_type = 'user'
                recommendations = recommender.get_recommendations(
                    identifier=user_id,
                    top_k=args.top_k,
                    exclude_items=train_user_items,
                    identifier_type=identifier_type
                )
            elif isinstance(recommender, HybridRecommender):
                identifier_type = 'user'
                recommendations = recommender.get_recommendations(
                    identifier=user_id,
                    top_k=args.top_k,
                    exclude_items=train_user_items,
                    identifier_type=identifier_type
                )
            else:
                recommendations = []

            # Add to the global set for catalogue coverage per algorithm
            recommended_items_per_algorithm[model_key].update(recommendations)

            # Calculate user-based metrics
            precision = precision_at_k(recommendations, user_test_items, k=args.top_k)
            recall = recall_at_k(recommendations, user_test_items, k=args.top_k)
            f_score = f_score_at_k(precision, recall)
            ndcg = ndcg_at_k(recommendations, user_test_items, k=args.top_k)
            map_score = mean_average_precision(recommendations, user_test_items, k=args.top_k)
            auc_score = area_under_roc_curve(recommendations, user_test_items, all_items)

            # Determine Model Type and Algorithm for reporting
            if 'content_based_weighted' in model_key:
                model_type = 'Content-Based Weighted'
                algorithm = 'Weighted Cosine Similarity'
            elif 'content_based' in model_key:
                model_type = 'Content-Based'
                algorithm = 'Cosine Similarity'
            elif 'collaborative' in model_key:
                model_type = 'Collaborative Filtering'
                algorithm = model_key.replace('collaborative_', '')
            elif 'hybrid' in model_key:
                model_type = 'Hybrid'
                algorithm = 'Content + Collaborative'
            else:
                model_type = 'Unknown'
                algorithm = 'Unknown'

            # Store results
            results.append({
                'Model Type': model_type,
                'Algorithm': algorithm,
                'User ID': user_id,
                'Precision@K': precision,
                'Recall@K': recall,
                'F1-Score@K': f_score,
                'nDCG@K': ndcg,
                'MAP@K': map_score,
                'AUC': auc_score
            })

            # Per-Title Metrics
            for title_id in title_ids:
                # Check if the user interacted with this title in training
                if title_id in train_user_items:
                    # Get all interactions after the interaction with the title
                    user_interactions_sorted = interactions_df[
                        (interactions_df['BE_ID'] == user_id) &
                        (interactions_df['COLLECTOR_TSTAMP'] >= cutoff_date)
                    ].sort_values('COLLECTOR_TSTAMP')

                    # Find the interaction timestamp with the title
                    title_interactions = interactions_df[
                        (interactions_df['BE_ID'] == user_id) &
                        (interactions_df['TITLE_ID'] == title_id)
                    ]

                    if not title_interactions.empty:
                        # Assuming the latest interaction with the title
                        title_last_interaction = title_interactions['COLLECTOR_TSTAMP'].max()

                        # Count interactions after the title interaction
                        post_title_interactions = interactions_df[
                            (interactions_df['BE_ID'] == user_id) &
                            (interactions_df['COLLECTOR_TSTAMP'] > title_last_interaction)
                        ]

                        if len(post_title_interactions) >= 1:
                            # Only consider users with at least 1 interaction after the title interaction
                            precision_title = precision_at_k(recommendations, user_test_items, k=args.top_k)
                            recall_title = recall_at_k(recommendations, user_test_items, k=args.top_k)
                            f_score_title = f_score_at_k(precision_title, recall_title)
                            ndcg_title = ndcg_at_k(recommendations, user_test_items, k=args.top_k)
                            map_score_title = mean_average_precision(recommendations, user_test_items, k=args.top_k)
                            auc_score_title = area_under_roc_curve(recommendations, user_test_items, all_items)

                            # Append to title metrics
                            title_metrics[title_id].append({
                                'Model Type': model_type,
                                'Algorithm': algorithm,
                                'User ID': user_id,
                                'Precision@K': precision_title,
                                'Recall@K': recall_title,
                                'F1-Score@K': f_score_title,
                                'nDCG@K': ndcg_title,
                                'MAP@K': map_score_title,
                                'AUC': auc_score_title
                            })

    # Create DataFrame from user-based results
    results_df = pd.DataFrame(results)

    # Define numeric columns for averaging
    numeric_cols = ['Precision@K', 'Recall@K', 'F1-Score@K', 'nDCG@K', 'MAP@K', 'AUC']

    # Ensure that the metric columns are numeric
    for col in numeric_cols:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

    # Compute average metrics across all users per algorithm
    average_metrics = results_df.groupby(['Model Type', 'Algorithm'])[numeric_cols].mean().reset_index()
    logging.info('\nAverage Recommendations Evaluation Results:')
    logging.info('\n' + average_metrics.to_string(index=False))

    # Calculate Prediction Coverage and Catalogue Coverage per algorithm
    for model_key, recommended_items in recommended_items_per_algorithm.items():
        # Filter interactions for items with at least min_interactions (assuming min_interactions=1)
        prediction_coverage = calculate_prediction_coverage(interactions_df, all_items, min_interactions=1)
        # Update Prediction Coverage per algorithm
        algorithm_coverage[model_key]['Prediction Coverage'] = prediction_coverage

        # Catalogue Coverage
        catalogue_coverage = calculate_catalogue_coverage(recommended_items, all_items)
        algorithm_coverage[model_key]['Catalogue Coverage'] = catalogue_coverage

    # Create DataFrame for algorithm coverage metrics
    algorithm_metrics = []
    for model_key, coverage in algorithm_coverage.items():
        # Retrieve Model Type for the algorithm
        matching_row = average_metrics[average_metrics['Algorithm'] == model_key.split('_')[-1]]
        if not matching_row.empty:
            model_type = matching_row['Model Type'].values[0]
        else:
            model_type = 'Unknown'

        algorithm_metrics.append({
            'Model Type': model_type,
            'Algorithm': model_key,
            'Prediction Coverage': coverage['Prediction Coverage'],
            'Catalogue Coverage': coverage['Catalogue Coverage']
        })
    algorithm_metrics_df = pd.DataFrame(algorithm_metrics)

    logging.info('\nAlgorithm Coverage Evaluation Results:')
    logging.info('\n' + algorithm_metrics_df.to_string(index=False))

    # Calculate metrics per title
    title_metrics_dfs = {}
    for title_id, metrics_list in title_metrics.items():
        if metrics_list:
            title_df = pd.DataFrame(metrics_list)
            # Compute average metrics for the title
            average_title_metrics = title_df.groupby(['Model Type', 'Algorithm'])[numeric_cols].mean().reset_index()
            title_metrics_dfs[title_id] = average_title_metrics
            logging.info(f'\nAverage Metrics for Title ID {title_id}:')
            logging.info('\n' + average_title_metrics.to_string(index=False))
        else:
            logging.info(f'\nNo eligible users found for Title ID {title_id}.')
            title_metrics_dfs[title_id] = pd.DataFrame(columns=['Model Type', 'Algorithm'] + numeric_cols)

    # Return all relevant DataFrames
    return results_df, algorithm_metrics_df, title_metrics_dfs


def calculate_prediction_coverage(interactions_df, all_items, min_interactions=1):
    """
    Calculates the Prediction Coverage.

    Parameters:
    - interactions_df: DataFrame containing user-item interactions.
    - all_items: Set or list of all available item IDs.
    - min_interactions: Minimum number of interactions required for an item to be eligible.

    Returns:
    - prediction_coverage: Float representing the prediction coverage percentage.
    """
    item_interaction_counts = interactions_df['TITLE_ID'].value_counts()
    eligible_items = set(item_interaction_counts[item_interaction_counts >= min_interactions].index)
    prediction_coverage = len(eligible_items) / len(all_items)
    return prediction_coverage


def calculate_catalogue_coverage(all_recommended_items, all_items):
    """
    Calculates the Catalogue Coverage.

    Parameters:
    - all_recommended_items: Set of all unique recommended item IDs across users.
    - all_items: Set or list of all available item IDs.

    Returns:
    - catalogue_coverage: Float representing the catalogue coverage percentage.
    """
    catalogue_coverage = len(all_recommended_items) / len(all_items)
    return catalogue_coverage


def save_trained_models(recommenders):
    """
    Saves all trained recommender models to disk.

    Parameters:
    - recommenders: Dictionary containing all trained recommender instances.
    """
    for model_key, recommender in recommenders.items():
        if isinstance(recommender, WeightedContentBasedRecommender):
            model_name = 'content_based_weighted'
        elif isinstance(recommender, ContentBasedRecommender):
            model_name = 'content_based'
        elif isinstance(recommender, CollaborativeFilteringRecommender):
            model_name = f"collaborative_{recommender.algorithm_name}"
        elif isinstance(recommender, HybridRecommender):
            model_name = f"hybrid_{recommender.collaborative_recommender.algorithm_name}"
        else:
            model_name = model_key  # Default to key if type is unknown

        save_model(recommender, model_name)
