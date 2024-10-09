# main.py

import argparse
import logging
import os
import time
import pandas as pd
from surprise import Dataset, Reader
from src import config
from src.data_processing import load_data, clean_titles_data, clean_interactions_data, save_processed_data, load_processed_data
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
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

def setup_logging(log_file='recommender.log'):
    """
    Sets up logging to output to console and file.
    """
    log_dir = os.path.join(config.BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """
    Parses command-line arguments for hyperparameters and model options.
    """

    parser = argparse.ArgumentParser(description='Recommendation System')

    parser.add_argument('--algorithms', type=str, default='SVD', help='Collaborative filtering algorithms to use, separated by commas (e.g., SVD,KNNBasic)')
    parser.add_argument('--model_type', type=str, default='all', help='Type of recommender: content, collaborative, hybrid, all')
    parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations to generate')
    parser.add_argument('--max_features', type=int, default=None, help='Max features for count vectorizer (default: use all features)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weighting factor for hybrid recommender')
    parser.add_argument('--n_factors', type=int, default=50, help='Number of factors for SVD')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs for SVD')
    parser.add_argument('--sample_percentage', type=float, default=20.0, help='Percentage of data to use for training')
    parser.add_argument('--fast_tuning', action='store_true', help='Enable fast hyperparameter tuning with less data (5%)')
    parser.add_argument('--cross_validate', action='store_true', help='Enable cross-validation for collaborative filtering')
    parser.add_argument('--test_train', type=bool, help='check test or train',default=True)
    parser.add_argument('--', action='store_true', help='Enable cross-validation for collaborative filtering')


    args = parser.parse_args()
    return args

def tune_hyperparameters(algo_name, data, max_evals=20):
    """
    Tunes hyperparameters for the collaborative filtering algorithm using Hyperopt.

    Parameters:
    - algo_name: The name of the algorithm to tune.
    - data: The Dataset object from Surprise.
    - max_evals: The maximum number of evaluations during tuning.

    Returns:
    - best_params: Dictionary of best hyperparameters found.
    """
    from surprise.model_selection import cross_validate

    def objective(params):
        if algo_name == 'SVD':
            algo = CollaborativeFilteringRecommender.get_algorithm_static(algo_name, params)
        elif algo_name in ['KNNBasic', 'KNNWithMeans', 'KNNBaseline']:
            algo = CollaborativeFilteringRecommender.get_algorithm_static(algo_name, params)
        else:
            return {'loss': float('inf'), 'status': STATUS_OK}

        cv_results = cross_validate(algo, data, measures=['Precision@K'], cv=3, verbose=False)
        mean_precision = cv_results['test_precision'].mean()
        return {'loss': -mean_precision, 'status': STATUS_OK}

    if algo_name == 'SVD':
        space = {
            'n_factors': hp.choice('n_factors', [20, 50, 100, 150]),
            'n_epochs': hp.choice('n_epochs', [10, 20, 30]),
            'lr_all': hp.uniform('lr_all', 0.002, 0.01),
            'reg_all': hp.uniform('reg_all', 0.02, 0.1),
        }
    elif algo_name in ['KNNBasic', 'KNNWithMeans', 'KNNBaseline']:
        space = {
            'k': hp.choice('k', [10, 20, 40, 60]),
            'min_k': hp.choice('min_k', [1, 2, 3]),
            'sim_options': {
                'name': hp.choice('name', ['cosine', 'msd']),
                'user_based': hp.choice('user_based', [True, False]),
            }
        }
    else:
        return {}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_params = space_eval(space, best)

    logging.info(f"Best hyperparameters for {algo_name}: {best_params}")
    return best_params

def main():
    # Setup logging
    setup_logging()

    # Parse arguments
    args = parse_arguments()
    algorithms = args.algorithms.split(',')

    # Update config with arguments
    config.TOP_K = args.top_k

    # Determine sample_percentage for hyperparameter tuning
    if args.fast_tuning:
        tuning_sample_percentage = 5.0
    else:
        tuning_sample_percentage = 100.0  # Use full data for tuning if fast_tuning is not enabled

    # Load or preprocess data
    # Check if preprocessed data exists
    if os.path.exists(config.PROCESSED_TITLES_PATH) and os.path.exists(config.PROCESSED_INTERACTIONS_PATH):
        logging.info('Loading preprocessed data...')
        titles_df, interactions_df = load_processed_data()
    else:
        logging.info('Loading raw data...')
        # Load data
        titles_df, interactions_df = load_data()

        logging.info('Preprocessing data...')
        # Data preprocessing
        titles_df = clean_titles_data(titles_df)
        interactions_df = clean_interactions_data(interactions_df)

        # Save processed data
        save_processed_data(titles_df, interactions_df)

    logging.info('Performing feature engineering...')
    # Feature engineering
    titles_df = process_text_features(titles_df)
    count_matrix, _ = create_count_matrix(titles_df, max_features=args.max_features)

    # Map TITLE_ID to ORIGINAL_TITLE
    title_id_to_title = dict(zip(titles_df['TITLE_ID'], titles_df['ORIGINAL_TITLE']))

    # Map interaction types to ratings
    # Adjusted INTERACTION_WEIGHTS to have ratings in the range 2-5
    interactions_df['rating'] = interactions_df['INTERACTION_TYPE'].map(config.INTERACTION_WEIGHTS)
    interactions_df['rating'] = interactions_df['rating'].fillna(0)

    # Ensure COLLECTOR_TSTAMP is datetime
    interactions_df['COLLECTOR_TSTAMP'] = pd.to_datetime(
        interactions_df['COLLECTOR_TSTAMP'],
        infer_datetime_format=True,
        utc=True,
        errors='coerce'
    )

    # Check for NaT values
    num_nat = interactions_df['COLLECTOR_TSTAMP'].isna().sum()
    if num_nat > 0:
        logging.warning(f"Found {num_nat} invalid timestamps in COLLECTOR_TSTAMP.")
        # Optionally, drop rows with NaT in COLLECTOR_TSTAMP
        interactions_df = interactions_df.dropna(subset=['COLLECTOR_TSTAMP']).reset_index(drop=True)

    # Log the number of data points after dropping NaT values
    logging.info(f"Number of data points after dropping invalid timestamps: {len(interactions_df)}")

    # Sort interactions by timestamp
    interactions_df = interactions_df.sort_values('COLLECTOR_TSTAMP').reset_index(drop=True)

    # Split into training and test sets chronologically
    logging.info('Splitting data into train and test sets chronologically...')
    split_index = int(len(interactions_df) * 0.8)  # 80% for training
    train_df_full = interactions_df.iloc[:split_index].reset_index(drop=True)
    test_df = interactions_df.iloc[split_index:].reset_index(drop=True)

    # Apply sample percentage for training
    if args.sample_percentage < 100.0:
        logging.info(f"Sampling {args.sample_percentage}% of training data...")
        train_df_full = train_df_full.sample(frac=args.sample_percentage / 100.0,
                                             random_state=config.RANDOM_STATE).reset_index(drop=True)

    # Log number of data points after sampling
    logging.info(f"Number of training data points after sampling: {len(train_df_full)}")
    logging.info(f"Number of test data points: {len(test_df)}")

    # Use the Reader class to parse the DataFrame
    # Adjusted rating scale to match the ratings in interactions_df['rating']
    min_rating = interactions_df['rating'].min()
    max_rating = interactions_df['rating'].max()
    reader = Reader(rating_scale=(min_rating, max_rating))

    # Build full trainset for final model training
    full_train_data = Dataset.load_from_df(
        train_df_full[['BE_ID', 'TITLE_ID', 'rating']],
        reader
    )

    # Build testset
    testset = list(test_df[['BE_ID', 'TITLE_ID', 'rating']].itertuples(index=False, name=None))

    # Initialize content-based recommender
    logging.info('Initializing content-based recommender...')
    content_recommender = ContentBasedRecommender(titles_df, count_matrix)

    # Prepare item features for diversity calculation
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

    # Process models based on model_type
    if 'content' in model_types:
        logging.info("Processing Content-Based Model...")
        # Choose a user and relevant items
        test_users = set(test_df['BE_ID'].unique())
        train_users = set(train_df_full['BE_ID'].unique())
        common_users = test_users & train_users

        if not common_users:
            logging.warning("No common users between train and test set.")
        else:
            user_interaction_counts = test_df[test_df['BE_ID'].isin(common_users)]['BE_ID'].value_counts()
            user_id = user_interaction_counts.idxmax()

            relevant_items = test_df[test_df['BE_ID'] == user_id]['TITLE_ID'].unique()
            user_train_items = train_df_full[train_df_full['BE_ID'] == user_id]['TITLE_ID'].unique()

            # Content-based recommendations
            title_id = relevant_items[0] if len(relevant_items) > 0 else titles_df['TITLE_ID'].iloc[0]
            recommendations = content_recommender.get_recommendations(title_id)['TITLE_ID'].tolist()

            # Map recommended item IDs to titles
            recommended_titles = [title_id_to_title.get(item_id, item_id) for item_id in recommendations]
            logging.info(f"\nContent Recommendations for User ID '{user_id}':")
            for title in recommended_titles:
                logging.info(title)

            # Log the number of recommendations and relevant items
            logging.info(f"Number of recommended items for content: {len(recommendations)}")

            # Handle empty recommendations or relevant items
            if not recommendations:
                logging.warning(f"No recommendations generated for content.")
            elif len(relevant_items) == 0:
                logging.warning(f"No relevant items found for user {user_id}.")
            else:
                # Evaluate recommendations
                precision = precision_at_k(recommendations, relevant_items, k=config.TOP_K)
                recall = recall_at_k(recommendations, relevant_items, k=config.TOP_K)
                f_score = f_score_at_k(precision, recall)
                ndcg = ndcg_at_k(recommendations, relevant_items, k=config.TOP_K)
                diversity = calculate_diversity(recommendations, item_features_dict)
                novelty = calculate_novelty(recommendations, item_popularity)
                coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
                ctr = estimated_ctr(recommendations, relevant_items)

                logging.info(f"\nContent Recommendations Evaluation:")
                logging.info(f"Precision@{config.TOP_K}: {precision:.4f}")
                logging.info(f"Recall@{config.TOP_K}: {recall:.4f}")
                logging.info(f"F-Score@{config.TOP_K}: {f_score:.4f}")
                logging.info(f"NDCG@{config.TOP_K}: {ndcg:.4f}")
                logging.info(f"Diversity: {diversity:.4f}")
                logging.info(f"Novelty: {novelty:.4f}")
                logging.info(f"Coverage: {coverage:.4f}")
                logging.info(f"Estimated CTR: {ctr:.4f}")

                # Store results
                results.append({
                    'Model Type': 'Content',
                    'Algorithm': 'N/A',
                    'Precision@K': precision,
                    'Recall@K': recall,
                    'F-Score@K': f_score,
                    'NDCG@K': ndcg,
                    'Diversity': diversity,
                    'Novelty': novelty,
                    'Coverage': coverage,
                    'Estimated CTR': ctr
                })

    if any(mt in ['collaborative', 'hybrid'] for mt in model_types):
        for algorithm in algorithms:
            algorithm = algorithm.strip()
            logging.info(f"Using algorithm: {algorithm}")

            # Hyperparameter tuning
            if args.fast_tuning:
                logging.info("Hyperparameter tuning enabled with fast_tuning.")
                # Sample a small percentage of data for tuning
                tuning_sample_size = int(len(train_df_full) * (tuning_sample_percentage / 100.0))
                tuning_df = train_df_full.sample(n=tuning_sample_size, random_state=config.RANDOM_STATE).reset_index(drop=True)
                tuning_data = Dataset.load_from_df(
                    tuning_df[['BE_ID', 'TITLE_ID', 'rating']],
                    reader
                )
                # Tune hyperparameters
                best_params = tune_hyperparameters(algorithm, tuning_data, max_evals=20)
            else:
                best_params = {}

            # Set algorithm hyperparameters
            if algorithm in ['KNNBasic', 'KNNWithMeans', 'KNNBaseline']:
                default_params = {
                    'k': 20,
                    'min_k': 1,
                    'sim_options': {
                        'name': 'cosine',  # Use cosine similarity
                        'user_based': False,  # Use item-item similarity
                    }
                }
                algo_params = {**default_params, **best_params}
            else:
                default_params = {
                    'n_factors': args.n_factors,
                    'n_epochs': args.n_epochs,
                    'random_state': config.RANDOM_STATE
                }
                algo_params = {**default_params, **best_params}

            # Initialize collaborative filtering recommender
            collaborative_recommender = CollaborativeFilteringRecommender(
                data=full_train_data,
                algorithm=algorithm,
                algo_params=algo_params
            )

            # Cross-validation (optional)
            if args.cross_validate:
                cv_precision, cv_recall = collaborative_recommender.cross_validate(cv=5)
                cv_f_score = f_score_at_k(cv_precision, cv_recall)
                logging.info(f"Cross-Validation Results - Precision@K: {cv_precision:.4f}, Recall@K: {cv_recall:.4f}, F-Score@K: {cv_f_score:.4f}")

            # Train on full training set
            collaborative_recommender.train()

            # Get list of users in both train and test sets
            test_users = set(test_df['BE_ID'].unique())
            train_users = set(train_df_full['BE_ID'].unique())
            common_users = test_users & train_users

            if not common_users:
                logging.warning("No common users between train and test set.")
                continue

            # Choose a user with most interactions in test set among common users
            user_interaction_counts = test_df[test_df['BE_ID'].isin(common_users)]['BE_ID'].value_counts()
            user_id = user_interaction_counts.idxmax()

            # Get relevant items from test set
            relevant_items = test_df[test_df['BE_ID'] == user_id]['TITLE_ID'].unique()

            # Get user's training interactions to exclude from recommendations
            user_train_items = train_df_full[train_df_full['BE_ID'] == user_id]['TITLE_ID'].unique()

            if 'collaborative' in model_types:
                # Collaborative filtering recommendations
                recommendations = collaborative_recommender.get_recommendations(user_id, exclude_items=user_train_items)

                # Map recommended item IDs to titles
                recommended_titles = [title_id_to_title.get(item_id, item_id) for item_id in recommendations]
                logging.info(f"\nCollaborative Recommendations for User ID '{user_id}':")
                for title in recommended_titles:
                    logging.info(title)

                # Log the number of recommendations and relevant items
                logging.info(f"Number of recommended items for collaborative: {len(recommendations)}")
                logging.info(f"Number of relevant items for user {user_id}: {len(relevant_items)}")

                # Handle empty recommendations or relevant items
                if not recommendations:
                    logging.warning(f"No recommendations generated for collaborative.")
                elif len(relevant_items) == 0:
                    logging.warning(f"No relevant items found for user {user_id}.")
                else:
                    # Evaluate recommendations
                    precision = precision_at_k(recommendations, relevant_items, k=config.TOP_K)
                    recall = recall_at_k(recommendations, relevant_items, k=config.TOP_K)
                    f_score = f_score_at_k(precision, recall)
                    ndcg = ndcg_at_k(recommendations, relevant_items, k=config.TOP_K)
                    diversity = calculate_diversity(recommendations, item_features_dict)
                    novelty = calculate_novelty(recommendations, item_popularity)
                    coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
                    ctr = estimated_ctr(recommendations, relevant_items)

                    logging.info(f"\nCollaborative Recommendations Evaluation:")
                    logging.info(f"Algorithm: {algorithm}")
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
                        'Model Type': 'Collaborative',
                        'Algorithm': algorithm,
                        'Precision@K': precision,
                        'Recall@K': recall,
                        'F-Score@K': f_score,
                        'NDCG@K': ndcg,
                        'Diversity': diversity,
                        'Novelty': novelty,
                        'Coverage': coverage,
                        'Estimated CTR': ctr
                    }

                    if args.cross_validate:
                        result_entry['Cross-Validation Precision@K'] = cv_precision
                        result_entry['Cross-Validation Recall@K'] = cv_recall
                        result_entry['Cross-Validation F-Score@K'] = cv_f_score

                    results.append(result_entry)

            if 'hybrid' in model_types:
                # Hybrid recommendations
                title_id = relevant_items[0] if len(relevant_items) > 0 else titles_df['TITLE_ID'].iloc[0]
                hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender, alpha=args.alpha)
                recommendations = hybrid_recommender.get_recommendations(user_id, title_id=title_id, exclude_items=user_train_items)

                # Map recommended item IDs to titles
                recommended_titles = [title_id_to_title.get(item_id, item_id) for item_id in recommendations]
                logging.info(f"\nHybrid Recommendations for User ID '{user_id}':")
                for title in recommended_titles:
                    logging.info(title)

                # Log the number of recommendations and relevant items
                logging.info(f"Number of recommended items for hybrid: {len(recommendations)}")
                logging.info(f"Number of relevant items for user {user_id}: {len(relevant_items)}")

                # Handle empty recommendations or relevant items
                if not recommendations:
                    logging.warning(f"No recommendations generated for hybrid.")
                elif len(relevant_items) == 0:
                    logging.warning(f"No relevant items found for user {user_id}.")
                else:
                    # Evaluate recommendations
                    precision = precision_at_k(recommendations, relevant_items, k=config.TOP_K)
                    recall = recall_at_k(recommendations, relevant_items, k=config.TOP_K)
                    f_score = f_score_at_k(precision, recall)
                    ndcg = ndcg_at_k(recommendations, relevant_items, k=config.TOP_K)
                    diversity = calculate_diversity(recommendations, item_features_dict)
                    novelty = calculate_novelty(recommendations, item_popularity)
                    coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
                    ctr = estimated_ctr(recommendations, relevant_items)

                    logging.info(f"\nHybrid Recommendations Evaluation:")
                    logging.info(f"Algorithm: {algorithm}")
                    logging.info(f"Alpha: {args.alpha}")
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
                        'Algorithm': algorithm,
                        'Alpha': args.alpha,
                        'Precision@K': precision,
                        'Recall@K': recall,
                        'F-Score@K': f_score,
                        'NDCG@K': ndcg,
                        'Diversity': diversity,
                        'Novelty': novelty,
                        'Coverage': coverage,
                        'Estimated CTR': ctr
                    }

                    if args.cross_validate:
                        result_entry['Cross-Validation Precision@K'] = cv_precision
                        result_entry['Cross-Validation Recall@K'] = cv_recall
                        result_entry['Cross-Validation F-Score@K'] = cv_f_score

                    results.append(result_entry)

    # Create a DataFrame to display results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('NDCG@K', ascending=False)

    logging.info('\nModel Ranking Based on NDCG@K:')
    logging.info('\n' + results_df.to_string(index=False))

if __name__ == '__main__':
    main()
