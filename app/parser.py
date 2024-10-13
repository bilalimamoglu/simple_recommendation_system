# app/parser.py

import argparse


def parse_arguments():
    """
    Parses command-line arguments for hyperparameters and model options.
    """
    parser = argparse.ArgumentParser(description='Recommendation System')

    parser.add_argument('--algorithms', type=str, default='SVD',
                        help='Collaborative filtering algorithms to use, separated by commas (e.g., SVD,KNNBasic)')
    parser.add_argument('--model_type', type=str, default='all',
                        help='Type of recommender: content, collaborative, hybrid, all')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of recommendations to generate')
    parser.add_argument('--max_features', type=int, default=5000,
                        help='Max features for TF-IDF vectorizer (default: 5000)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weighting factor for hybrid recommender')
    parser.add_argument('--sample_percentage', type=float, default=5.0,
                        help='Percentage of training data to use (default: 5.0)')

    args = parser.parse_args()
    return args
