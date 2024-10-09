import argparse


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

    args = parser.parse_args()
    return args