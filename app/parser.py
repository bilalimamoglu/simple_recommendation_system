# app/parser.py

import argparse


def parse_arguments():
    """
    Parses command-line arguments for hyperparameters and model options.
    """
    parser = argparse.ArgumentParser(description='Recommendation System')

    parser.add_argument('--algorithms', type=str, default='SVD',
                        help='Collaborative filtering algorithms to use, separated by commas (e.g., SVD,KNNBaseline)')
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

    # Add weights for weighted content-based recommender
    parser.add_argument('--title_weight', type=float, default=0.3, help='Weight for ORIGINAL_TITLE feature')
    parser.add_argument('--genre_weight', type=float, default=0.3, help='Weight for GENRE_TMDB feature')
    parser.add_argument('--director_weight', type=float, default=0.2, help='Weight for DIRECTOR feature')
    parser.add_argument('--actor_weight', type=float, default=0.1, help='Weight for ACTOR feature')
    parser.add_argument('--producer_weight', type=float, default=0.1, help='Weight for PRODUCER feature')

    args = parser.parse_args()
    return args
