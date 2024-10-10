# app/main.py

import argparse
import logging
import os
import pandas as pd
from surprise import Dataset, Reader
from tqdm import tqdm  # Added tqdm for progress bars
import random  # Added random for sampling

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logger import setup_logging
from app.parser import parse_arguments
from app.utils import set_seed
from src import config
from src.data_processing import (
    load_data, clean_titles_data, clean_interactions_data,
    save_processed_data, load_processed_data
)
from src.feature_engineering import process_text_features, create_count_matrix
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import CollaborativeFilteringRecommender
from src.models.hybrid_model import HybridRecommender
from src.models.evaluation import (
    precision_at_k, recall_at_k, ndcg_at_k, f_score_at_k,
    calculate_diversity, calculate_novelty, estimated_ctr,
    calculate_coverage
)
from sklearn.decomposition import TruncatedSVD

setup_logging()
set_seed()

def main():
    # Parse arguments
    args = parse_arguments()
    algorithms = [alg.strip() for alg in args.algorithms.split(',')]

    # Update config with arguments
    config.TOP_K = args.top_k
    config.ALPHA = args.alpha  # Ensure ALPHA is set in config if used elsewhere

    # Load or preprocess data
    if os.path.exists(config.PROCESSED_TITLES_PATH) and os.path.exists(config.PROCESSED_INTERACTIONS_PATH):
        logging.info('Loading preprocessed data...')
        titles_df, interactions_df = load_processed_data()
    else:
        logging.info('Loading raw data...')
        titles_df, interactions_df = load_data()

        logging.info('Preprocessing data...')
        titles_df = clean_titles_data(titles_df)
        interactions_df = clean_interactions_data(interactions_df)

        save_processed_data(titles_df, interactions_df)

    logging.info('Performing feature engineering...')
    titles_df = process_text_features(titles_df)
    count_matrix, _ = create_count_matrix(titles_df, max_features=args.max_features)

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

    # Split into training and test sets chronologically
    logging.info('Splitting data into train and test sets chronologically...')
    split_index = int(len(interactions_df) * 0.8)  # 80% for training
    train_df_full = interactions_df.iloc[:split_index].reset_index(drop=True)
    test_df = interactions_df.iloc[split_index:].reset_index(drop=True)

    logging.info(f"Number of training data points: {len(train_df_full)}")
    logging.info(f"Number of test data points: {len(test_df)}")

    # Implement Sampling Strategy using sample_percentage from arguments
    logging.info('Applying sampling strategy to training data...')
    sample_frac = args.sample_percentage / 100.0
    train_df_sampled = train_df_full.sample(frac=sample_frac, random_state=config.RANDOM_STATE).reset_index(drop=True)
    logging.info(f"Number of training data points after sampling ({args.sample_percentage}%): {len(train_df_sampled)}")

    # Identify common users present in both training and test sets
    logging.info('Identifying common users present in both training and test sets...')
    common_users = set(train_df_sampled['BE_ID']).intersection(set(test_df['BE_ID']))
    logging.info(f"Number of common users for evaluation: {len(common_users)}")

    if not common_users:
        logging.error("No common users found between training and test sets. Adjust the sampling percentage or check your data.")
        return

    # Sample 1000 users from common_users for evaluation
    sample_size = 1000
    if len(common_users) >= sample_size:
        sample_users = random.sample(list(common_users), sample_size)
    else:
        sample_users = list(common_users)
        logging.warning(f"Requested sample size {sample_size} exceeds the number of common users. Evaluating on {len(sample_users)} users.")

    logging.info(f"Number of users selected for evaluation: {len(sample_users)}")

    # Filter test set to include only interactions from sampled users
    test_df_filtered = test_df[test_df['BE_ID'].isin(sample_users)].reset_index(drop=True)
    logging.info(f"Number of test data points after filtering sampled users: {len(test_df_filtered)}")

    # Use the Reader class to parse the DataFrame
    min_rating = interactions_df['rating'].min()
    max_rating = interactions_df['rating'].max()
    reader = Reader(rating_scale=(min_rating, max_rating))

    # Build trainset for collaborative filtering
    train_data = Dataset.load_from_df(
        train_df_sampled[['BE_ID', 'TITLE_ID', 'rating']],
        reader
    )

    # Initialize and train content-based recommender
    logging.info('Initializing content-based recommender...')
    content_recommender = ContentBasedRecommender(titles_df, count_matrix, algorithm='cosine')
    content_recommender.train()

    # Define Algorithm Type Mappings
    user_based_algorithms = ['SVD', 'SVDpp', 'NMF']
    item_based_algorithms = ['KNNBasic', 'KNNWithMeans', 'KNNBaseline']

    # Initialize and train collaborative filtering recommenders
    collaborative_recommenders = []
    for algo in algorithms:
        if algo in user_based_algorithms:
            user_based = True
        elif algo in item_based_algorithms:
            user_based = False
        else:
            logging.warning(f"Algorithm '{algo}' is not recognized. Skipping.")
            continue

        logging.info(f'Initializing {"user-based" if user_based else "item-based"} collaborative filtering recommender with algorithm: {algo}')
        collaborative_recommender = CollaborativeFilteringRecommender(
            data=train_data,
            algorithm=algo,
            algo_params={'sim_options': {'name': 'cosine', 'user_based': user_based}} if not user_based else {},
            user_based=user_based
        )
        collaborative_recommender.train()
        collaborative_recommenders.append(collaborative_recommender)

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

    # Prepare item features for diversity and novelty calculations
    svd = TruncatedSVD(n_components=100, random_state=config.RANDOM_STATE)
    reduced_item_features = svd.fit_transform(count_matrix)
    item_indices = titles_df['TITLE_ID'].tolist()
    item_features_dict = {item_id: reduced_item_features[idx] for idx, item_id in enumerate(item_indices)}

    # Prepare item popularity for novelty calculation
    item_popularity = interactions_df['TITLE_ID'].value_counts().to_dict()

    # Determine model types
    model_types = args.model_type.split(',') if args.model_type != 'all' else ['content', 'collaborative', 'hybrid']

    # Initialize results list
    results = []

    # Evaluate Content-Based Recommender
    if 'content' in model_types:
        logging.info("Evaluating Content-Based Model on the test set...")

        for user_id in tqdm(sample_users, desc="Evaluating Content-Based Model"):
            # Get the items the user interacted with in the test set
            user_test_items = test_df_filtered[test_df_filtered['BE_ID'] == user_id]['TITLE_ID'].tolist()

            if not user_test_items:
                continue  # Skip users with no test interactions

            # Get the user's training items
            user_train_items = train_df_sampled[train_df_sampled['BE_ID'] == user_id]['TITLE_ID'].tolist()
            if not user_train_items:
                continue  # Skip users with no training interactions

            # Assume the last item in training is the representative
            representative_item = user_train_items[-1]

            # Get recommendations
            recommendations = content_recommender.get_recommendations(
                title_id=representative_item,
                top_k=config.TOP_K,
                exclude_items=user_train_items,
                identifier_type='title'
            )

            # Calculate metrics
            precision = precision_at_k(recommendations, user_test_items, k=config.TOP_K)
            recall = recall_at_k(recommendations, user_test_items, k=config.TOP_K)
            f_score = f_score_at_k(precision, recall)
            ndcg = ndcg_at_k(recommendations, user_test_items, k=config.TOP_K)
            diversity = calculate_diversity(recommendations, item_features_dict)
            novelty = calculate_novelty(recommendations, item_popularity)
            coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
            ctr = estimated_ctr(recommendations, user_test_items)

            # Store results
            results.append({
                'Model Type': 'Content-Based',
                'Algorithm': 'Cosine Similarity',
                'User ID': user_id,
                'Precision@K': precision,
                'Recall@K': recall,
                'F-Score@K': f_score,
                'NDCG@K': ndcg,
                'Diversity': diversity,
                'Novelty': novelty,
                'Coverage': coverage,
                'Estimated CTR': ctr
            })

    # Evaluate Collaborative Filtering Recommender
    if 'collaborative' in model_types:
        logging.info("Evaluating Collaborative Filtering Models on the test set...")

        for collaborative_recommender in collaborative_recommenders:
            if collaborative_recommender.user_based:
                # User-Based Collaborative Filtering Evaluation
                logging.info(f"Evaluating User-Based Collaborative Filtering Model: {collaborative_recommender.algorithm_name}")

                for user_id in tqdm(sample_users, desc=f"Evaluating {collaborative_recommender.algorithm_name}"):
                    # Get the items the user interacted with in the test set
                    user_test_items = test_df_filtered[test_df_filtered['BE_ID'] == user_id]['TITLE_ID'].tolist()

                    if not user_test_items:
                        continue  # Skip users with no test interactions

                    # Get recommendations
                    recommendations = collaborative_recommender.get_recommendations(
                        identifier=user_id,
                        top_k=config.TOP_K,
                        exclude_items=None,
                        identifier_type='user'
                    )

                    # Calculate metrics
                    precision = precision_at_k(recommendations, user_test_items, k=config.TOP_K)
                    recall = recall_at_k(recommendations, user_test_items, k=config.TOP_K)
                    f_score = f_score_at_k(precision, recall)
                    ndcg = ndcg_at_k(recommendations, user_test_items, k=config.TOP_K)
                    diversity = calculate_diversity(recommendations, item_features_dict)
                    novelty = calculate_novelty(recommendations, item_popularity)
                    coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
                    ctr = estimated_ctr(recommendations, user_test_items)

                    # Store results
                    results.append({
                        'Model Type': 'Collaborative Filtering',
                        'Algorithm': collaborative_recommender.algorithm_name,
                        'User ID': user_id,
                        'Precision@K': precision,
                        'Recall@K': recall,
                        'F-Score@K': f_score,
                        'NDCG@K': ndcg,
                        'Diversity': diversity,
                        'Novelty': novelty,
                        'Coverage': coverage,
                        'Estimated CTR': ctr
                    })
            else:
                # Item-Based Collaborative Filtering Evaluation
                logging.info(f"Evaluating Item-Based Collaborative Filtering Model: {collaborative_recommender.algorithm_name}")

                for user_id in tqdm(sample_users, desc=f"Evaluating {collaborative_recommender.algorithm_name}"):
                    # Get the items the user interacted with in the test set
                    user_test_items = test_df_filtered[test_df_filtered['BE_ID'] == user_id]['TITLE_ID'].tolist()

                    if not user_test_items:
                        continue  # Skip users with no test interactions

                    # Get the user's training items
                    user_train_items = train_df_sampled[train_df_sampled['BE_ID'] == user_id]['TITLE_ID'].tolist()
                    if not user_train_items:
                        continue  # Skip users with no training interactions

                    # Assume the last item in training is the representative
                    representative_item = user_train_items[-1]

                    # Get recommendations
                    recommendations = collaborative_recommender.get_recommendations(
                        identifier=representative_item,
                        top_k=config.TOP_K,
                        exclude_items=user_train_items,
                        identifier_type='title'
                    )

                    # Calculate metrics
                    precision = precision_at_k(recommendations, user_test_items, k=config.TOP_K)
                    recall = recall_at_k(recommendations, user_test_items, k=config.TOP_K)
                    f_score = f_score_at_k(precision, recall)
                    ndcg = ndcg_at_k(recommendations, user_test_items, k=config.TOP_K)
                    diversity = calculate_diversity(recommendations, item_features_dict)
                    novelty = calculate_novelty(recommendations, item_popularity)
                    coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
                    ctr = estimated_ctr(recommendations, user_test_items)

                    # Store results
                    results.append({
                        'Model Type': 'Collaborative Filtering',
                        'Algorithm': collaborative_recommender.algorithm_name,
                        'User ID': user_id,
                        'Precision@K': precision,
                        'Recall@K': recall,
                        'F-Score@K': f_score,
                        'NDCG@K': ndcg,
                        'Diversity': diversity,
                        'Novelty': novelty,
                        'Coverage': coverage,
                        'Estimated CTR': ctr
                    })

    # Evaluate Hybrid Recommender
    if 'hybrid' in model_types:
        logging.info("Evaluating Hybrid Recommender Models on the test set...")

        for hybrid_recommender in hybrid_recommenders:
            for user_id in tqdm(sample_users, desc="Evaluating Hybrid Recommender"):
                # Get the items the user interacted with in the test set
                user_test_items = test_df_filtered[test_df_filtered['BE_ID'] == user_id]['TITLE_ID'].tolist()

                if not user_test_items:
                    continue  # Skip users with no test interactions

                # Get the user's training items
                user_train_items = train_df_sampled[train_df_sampled['BE_ID'] == user_id]['TITLE_ID'].tolist()
                if not user_train_items:
                    continue  # Skip users with no training interactions

                # Assume the last item in training is the representative
                representative_item = user_train_items[-1]

                # Get recommendations
                recommendations = hybrid_recommender.get_recommendations(
                    identifier=user_id,
                    top_k=config.TOP_K,
                    exclude_items=user_train_items,
                    identifier_type='user'
                )

                # Calculate metrics
                precision = precision_at_k(recommendations, user_test_items, k=config.TOP_K)
                recall = recall_at_k(recommendations, user_test_items, k=config.TOP_K)
                f_score = f_score_at_k(precision, recall)
                ndcg = ndcg_at_k(recommendations, user_test_items, k=config.TOP_K)
                diversity = calculate_diversity(recommendations, item_features_dict)
                novelty = calculate_novelty(recommendations, item_popularity)
                coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
                ctr = estimated_ctr(recommendations, user_test_items)

                # Store results
                results.append({
                    'Model Type': 'Hybrid',
                    'Algorithm': 'Content + Collaborative',
                    'User ID': user_id,
                    'Precision@K': precision,
                    'Recall@K': recall,
                    'F-Score@K': f_score,
                    'NDCG@K': ndcg,
                    'Diversity': diversity,
                    'Novelty': novelty,
                    'Coverage': coverage,
                    'Estimated CTR': ctr
                })

    # Create DataFrame to display results
    if results:
        results_df = pd.DataFrame(results)
        # Compute average metrics across all users (only numeric columns)
        numeric_cols = ['Precision@K', 'Recall@K', 'F-Score@K', 'NDCG@K', 'Diversity', 'Novelty', 'Coverage', 'Estimated CTR']
        average_metrics = results_df.groupby(['Model Type', 'Algorithm'])[numeric_cols].mean().reset_index()
        logging.info('\nAverage Recommendations Evaluation Results:')
        logging.info('\n' + average_metrics.to_string(index=False))

    else:
        logging.info("No evaluation results to display.")

if __name__ == '__main__':
    main()
