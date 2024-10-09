# app/main.py

import argparse
import logging
import os
import pandas as pd
from surprise import Dataset, Reader

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
        infer_datetime_format=True,
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

    # Use the Reader class to parse the DataFrame
    min_rating = interactions_df['rating'].min()
    max_rating = interactions_df['rating'].max()
    reader = Reader(rating_scale=(min_rating, max_rating))

    # Build full trainset for collaborative filtering
    full_train_data = Dataset.load_from_df(
        train_df_full[['BE_ID', 'TITLE_ID', 'rating']],
        reader
    )

    # Initialize and train content-based recommender
    logging.info('Initializing content-based recommender...')
    content_recommender = ContentBasedRecommender(titles_df, count_matrix, algorithm='cosine')
    content_recommender.train()

    # Initialize and train collaborative filtering recommenders
    collaborative_recommenders = []
    for algo in algorithms:
        logging.info(f'Initializing collaborative filtering recommender with algorithm: {algo}')
        collaborative_recommender = CollaborativeFilteringRecommender(
            data=full_train_data,
            algorithm=algo,
            algo_params={}  # Empty as hyperparameter tuning is removed
        )
        collaborative_recommender.train()
        collaborative_recommenders.append(collaborative_recommender)

    # Initialize and train hybrid recommenders
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
        logging.info("Evaluating Content-Based Model...")

        # Define test titles for evaluation
        test_titles = ['tm107473', 'tm50355', 'ts89259']

        # Create relevant items dictionary (replace with actual relevant TITLE_IDs)
        relevant_titles_dict = {
            'tm107473': ['tm1074731', 'tm1074732', 'tm1074733'],  # Replace with actual relevant TITLE_IDs
            'tm50355': ['tm503551', 'tm503552', 'tm503553'],
            'ts89259': ['ts892591', 'ts892592', 'ts892593']
        }

        avg_precision, avg_recall, avg_f_score = content_recommender.evaluate(
            test_titles=test_titles,
            relevant_titles_dict=relevant_titles_dict,
            top_k=config.TOP_K
        )
        logging.info(f"Content-Based Model Evaluation - Precision@K: {avg_precision:.4f}, Recall@K: {avg_recall:.4f}, F-Score@K: {avg_f_score:.4f}")

        for title_id in test_titles:
            recommendations = content_recommender.get_recommendations(title_id, top_k=config.TOP_K, exclude_items=None)
            recommended_titles = [title_id_to_title.get(item_id, item_id) for item_id in recommendations]
            logging.info(f"\nContent-Based Recommendations for Title ID '{title_id}':")
            for title in recommended_titles:
                logging.info(title)

            # Evaluate individual recommendations if relevant items are provided
            relevant_items = relevant_titles_dict.get(title_id, [])
            if relevant_items:
                precision = precision_at_k(recommendations, relevant_items, k=config.TOP_K)
                recall = recall_at_k(recommendations, relevant_items, k=config.TOP_K)
                f_score = f_score_at_k(precision, recall)
                ndcg = ndcg_at_k(recommendations, relevant_items, k=config.TOP_K)
                diversity = calculate_diversity(recommendations, item_features_dict)
                novelty = calculate_novelty(recommendations, item_popularity)
                coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
                ctr = estimated_ctr(recommendations, relevant_items)

                logging.info(f"\nContent-Based Recommendations Evaluation for Title ID '{title_id}':")
                logging.info(f"Precision@{config.TOP_K}: {precision:.4f}")
                logging.info(f"Recall@{config.TOP_K}: {recall:.4f}")
                logging.info(f"F-Score@{config.TOP_K}: {f_score:.4f}")
                logging.info(f"NDCG@{config.TOP_K}: {ndcg:.4f}")
                logging.info(f"Diversity: {diversity:.4f}")
                logging.info(f"Novelty: {novelty:.4f}")
                logging.info(f"Coverage: {coverage:.4f}")
                logging.info(f"Estimated CTR: {ctr:.4f}")

                # Store results
                result_entry = {
                    'Model Type': 'Content-Based',
                    'Algorithm': 'Cosine Similarity',
                    'Title ID': title_id,
                    'Precision@K': precision,
                    'Recall@K': recall,
                    'F-Score@K': f_score,
                    'NDCG@K': ndcg,
                    'Diversity': diversity,
                    'Novelty': novelty,
                    'Coverage': coverage,
                    'Estimated CTR': ctr
                }

                results.append(result_entry)

    # Evaluate Collaborative Filtering Recommender
    if 'collaborative' in model_types:
        logging.info("Evaluating Collaborative Filtering Model...")
        # Removed user-based evaluations as per your request

    # Evaluate Hybrid Recommender
    if 'hybrid' in model_types:
        logging.info("Evaluating Hybrid Recommender Model...")

        # Define test titles for evaluation
        test_titles = ['tm107473', 'tm50355', 'ts89259']

        # Create relevant items dictionary for test titles (replace with actual relevant TITLE_IDs)
        relevant_titles_dict = {
            'tm107473': ['tm1074731', 'tm1074732', 'tm1074733'],  # Replace with actual relevant TITLE_IDs
            'tm50355': ['tm503551', 'tm503552', 'tm503553'],
            'ts89259': ['ts892591', 'ts892592', 'ts892593']
        }

        for hybrid_recommender in hybrid_recommenders:
            avg_precision, avg_recall, avg_f_score = hybrid_recommender.evaluate(
                identifiers=test_titles,
                relevant_items_dict=relevant_titles_dict,
                top_k=config.TOP_K,
                identifier_type='title'
            )
            logging.info(f"Hybrid Recommender Model Evaluation - Precision@K: {avg_precision:.4f}, Recall@K: {avg_recall:.4f}, F-Score@K: {avg_f_score:.4f}")

            for title_id in test_titles:
                recommendations = hybrid_recommender.get_recommendations(
                    identifier=title_id,
                    top_k=config.TOP_K,
                    exclude_items=None,
                    identifier_type='title'
                )
                recommended_titles = [title_id_to_title.get(item_id, item_id) for item_id in recommendations]
                logging.info(f"\nHybrid Recommendations for Title ID '{title_id}':")
                for title in recommended_titles:
                    logging.info(title)

                # Evaluate individual recommendations if relevant items are provided
                relevant_items = relevant_titles_dict.get(title_id, [])
                if relevant_items:
                    precision = precision_at_k(recommendations, relevant_items, k=config.TOP_K)
                    recall = recall_at_k(recommendations, relevant_items, k=config.TOP_K)
                    f_score = f_score_at_k(precision, recall)
                    ndcg = ndcg_at_k(recommendations, relevant_items, k=config.TOP_K)
                    diversity = calculate_diversity(recommendations, item_features_dict)
                    novelty = calculate_novelty(recommendations, item_popularity)
                    coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
                    ctr = estimated_ctr(recommendations, relevant_items)

                    logging.info(f"\nHybrid Recommendations Evaluation for Title ID '{title_id}':")
                    logging.info(f"Precision@{config.TOP_K}: {precision:.4f}")
                    logging.info(f"Recall@{config.TOP_K}: {recall:.4f}")
                    logging.info(f"F-Score@{config.TOP_K}: {f_score:.4f}")
                    logging.info(f"NDCG@{config.TOP_K}: {ndcg:.4f}")
                    logging.info(f"Diversity: {diversity:.4f}")
                    logging.info(f"Novelty: {novelty:.4f}")
                    logging.info(f"Coverage: {coverage:.4f}")
                    logging.info(f"Estimated CTR: {ctr:.4f}")

                    # Store results
                    result_entry = {
                        'Model Type': 'Hybrid',
                        'Algorithm': 'Content + Collaborative',
                        'Title ID': title_id,
                        'Precision@K': precision,
                        'Recall@K': recall,
                        'F-Score@K': f_score,
                        'NDCG@K': ndcg,
                        'Diversity': diversity,
                        'Novelty': novelty,
                        'Coverage': coverage,
                        'Estimated CTR': ctr
                    }

                    results.append(result_entry)

    # Create a DataFrame to display results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('NDCG@K', ascending=False)

        logging.info('\nModel Ranking Based on NDCG@K:')
        logging.info('\n' + results_df.to_string(index=False))
    else:
        logging.info("No evaluation results to display.")

if __name__ == '__main__':
    main()
