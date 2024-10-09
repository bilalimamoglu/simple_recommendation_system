# src/models/collaborative_filtering.py

import logging
import pandas as pd
from surprise import Dataset
from surprise import SVD, KNNBasic, KNNWithMeans, KNNBaseline, SVDpp, NMF
from src import config
from src.models.evaluation import precision_at_k, recall_at_k, f_score_at_k, ndcg_at_k, calculate_diversity, calculate_novelty, estimated_ctr, calculate_coverage

class CollaborativeFilteringRecommender:
    def __init__(self, data, algorithm='SVD', algo_params=None):
        """
        Initializes the collaborative filtering model using the Surprise library.

        Parameters:
        - data: The Dataset object from Surprise.
        - algorithm: Name of the algorithm to use.
        - algo_params: Dictionary of algorithm-specific hyperparameters.
        """
        self.data = data
        self.algorithm_name = algorithm
        self.algo_params = algo_params if algo_params is not None else {}
        self.algo = self.get_algorithm(algorithm)
        self.trained = False
        self.trainset = None

    @staticmethod
    def get_algorithm_static(algorithm_name, algo_params):
        """
        Static method to get an algorithm instance.
        """
        algorithms = {
            'SVD': SVD,
            'SVDpp': SVDpp,
            'NMF': NMF,
            'KNNBasic': KNNBasic,
            'KNNWithMeans': KNNWithMeans,
            'KNNBaseline': KNNBaseline,
        }
        if algorithm_name not in algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' is not supported.")
        return algorithms[algorithm_name](**algo_params)

    def get_algorithm(self, algorithm_name):
        """
        Returns a Surprise algorithm object based on the algorithm name and hyperparameters.
        """
        return self.get_algorithm_static(algorithm_name, self.algo_params)

    def train(self):
        """
        Trains the collaborative filtering model.
        """
        logging.info(f"Training Collaborative Filtering model using {self.algorithm_name}...")
        self.trainset = self.data.build_full_trainset()
        self.algo.fit(self.trainset)
        self.trained = True
        logging.info("Collaborative Filtering model training complete.")

    def get_recommendations(self, user_id, top_k=config.TOP_K, exclude_items=None):
        """
        Generates top K recommendations for a given user.

        Parameters:
        - user_id: The user ID for whom to generate recommendations.
        - top_k: The number of recommendations to generate.
        - exclude_items: A list of item IDs to exclude from recommendations (e.g., items the user has already interacted with).

        Returns:
        - A list of recommended TITLE_IDs.
        """
        if not self.trained:
            self.train()

        # Check if user_id is in the training set
        if user_id not in self.trainset._raw2inner_id_users:
            logging.warning(f"User {user_id} is not in the training set.")
            return []

        # Get all items in the training set as inner IDs
        all_items_inner_ids = set(self.trainset.all_items())

        # Exclude items the user has already interacted with
        if exclude_items is not None and len(exclude_items) > 0:
            exclude_inner_ids = {self.trainset.to_inner_iid(item) for item in exclude_items if item in self.trainset._raw2inner_id_items}
            all_items_inner_ids = all_items_inner_ids - exclude_inner_ids

        # Generate predictions for all items not interacted with
        testset = [(user_id, self.trainset.to_raw_iid(item_inner_id), 0) for item_inner_id in all_items_inner_ids]
        predictions = self.algo.test(testset)

        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get the top K item IDs
        top_k_items = [pred.iid for pred in predictions[:top_k]]

        return top_k_items

    def evaluate(self, user_ids, relevant_items_dict, top_k=config.TOP_K):
        """
        Evaluates the collaborative filtering recommender using precision, recall, and F-score.

        Parameters:
        - user_ids: List of user IDs to evaluate on.
        - relevant_items_dict: Dictionary where keys are user_ids and values are lists of relevant items.
        - top_k: Number of recommendations to evaluate.

        Returns:
        - Tuple of average precision, recall, and F-score.
        """
        precisions = []
        recalls = []

        for user_id in user_ids:
            relevant_items = relevant_items_dict.get(user_id, [])
            recommendations = self.get_recommendations(user_id, top_k=top_k, exclude_items=None)

            precision = precision_at_k(recommendations, relevant_items, k=top_k)
            recall = recall_at_k(recommendations, relevant_items, k=top_k)
            precisions.append(precision)
            recalls.append(recall)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f_score = f_score_at_k(avg_precision, avg_recall)

        return avg_precision, avg_recall, avg_f_score

    def get_predicted_ratings(self, user_id):
        """
        Returns predicted ratings for all items for a given user.

        Parameters:
        - user_id: The user ID for whom to predict ratings.

        Returns:
        - A pandas Series with item IDs as index and predicted ratings as values.
        """
        if not self.trained:
            self.train()

        # Check if user_id is in the training set
        if user_id not in self.trainset._raw2inner_id_users:
            logging.warning(f"User {user_id} is not in the training set.")
            return pd.Series()

        # Get all items in the training set as inner IDs
        all_items_inner_ids = set(self.trainset.all_items())

        # Generate predictions for all items
        testset = [(user_id, self.trainset.to_raw_iid(item_inner_id), 0) for item_inner_id in all_items_inner_ids]
        predictions = self.algo.test(testset)

        # Create a Series with item IDs as index and predicted ratings as values
        predicted_ratings = pd.Series({pred.iid: pred.est for pred in predictions})

        return predicted_ratings
