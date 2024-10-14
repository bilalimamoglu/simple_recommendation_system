# app/main.py

import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from app.logger import setup_logging
from app.parser import parse_arguments
from app.utils import set_seed, save_model, load_model
from src.workflow import (
    load_and_preprocess_data,
    perform_feature_engineering,
    initialize_and_train_recommenders,
    evaluate_recommenders,
    save_trained_models
)


def main():
    # Setup logging and set seed
    setup_logging()
    set_seed()

    # Parse command-line arguments
    args = parse_arguments()

    # Load and preprocess data
    titles_df, interactions_df = load_and_preprocess_data(args)

    # Perform feature engineering
    tfidf_matrix_svd, weighted_tfidf_matrix_svd = perform_feature_engineering(titles_df, args)

    # Initialize and train recommenders
    recommenders, sampled_users = initialize_and_train_recommenders(
        titles_df, interactions_df, tfidf_matrix_svd, weighted_tfidf_matrix_svd, args
    )

    # Evaluate recommenders
    evaluation_results = evaluate_recommenders(
        recommenders, interactions_df, titles_df, sampled_users, args
    )

    # Save evaluation results
    evaluation_results.to_csv(
        os.path.join(config.OUTPUT_DIR, 'recommendation_evaluation_results.csv'),
        index=False
    )

    # Save trained models
    save_trained_models(recommenders)

    logging.info("All processes completed successfully.")


if __name__ == '__main__':
    main()
