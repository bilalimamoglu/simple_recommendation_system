# app/parser.py

import argparse


def parse_arguments():
    """
    Parses command-line arguments for hyperparameters and model options.
    """
    parser = argparse.ArgumentParser(description='Recommendation System Workflow')
    parser.add_argument('--max_features', type=int, default=10000, help='Maximum number of features for TF-IDF')
    parser.add_argument('--title_weight', type=float, default=1.0, help='Weight for ORIGINAL_TITLE feature')
    parser.add_argument('--genre_weight', type=float, default=1.0, help='Weight for GENRE_TMDB feature')
    parser.add_argument('--director_weight', type=float, default=1.0, help='Weight for DIRECTOR feature')
    parser.add_argument('--actor_weight', type=float, default=1.0, help='Weight for ACTOR feature')
    parser.add_argument('--producer_weight', type=float, default=1.0, help='Weight for PRODUCER feature')
    parser.add_argument('--sample_percentage', type=float, default=100.0, help='Percentage of training data to sample')
    parser.add_argument('--algorithms', type=str, default='SVD,SVDpp,NMF,KNNBaseline,KNNBasic,KNNWithMeans',
                        help='Comma-separated list of collaborative filtering algorithms to use')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weighting factor for hybrid recommender')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top recommendations to evaluate')

    args = parser.parse_args()
    return args
