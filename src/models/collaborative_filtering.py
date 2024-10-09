# src/models/collaborative_filtering.py

import logging
import pandas as pd
from surprise import accuracy, Dataset
from surprise.model_selection import cross_validate
from surprise import SVD, KNNBasic, KNNWithMeans, KNNBaseline, SVDpp, NMF
from src import config

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
        Static method to get an algorithm instance for hyperparameter tuning.
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

    def cross_validate(self, cv=5):
        """
        Performs cross-validation and returns the average RMSE and MAE.
        """
        logging.info(f"Performing cross-validation with {cv}-folds...")
        results = cross_validate(self.algo, self.data, measures=['RMSE', 'MAE'], cv=cv, verbose=False)
        avg_rmse = results['test_rmse'].mean()
        avg_mae = results['test_mae'].mean()
        logging.info(f"Cross-Validation Results - RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}")
        return avg_rmse, avg_mae

    def train(self):
        """
        Trains the collaborative filtering model.
        """
        self.trainset = self.data.build_full_trainset()
        self.algo.fit(self.trainset)
        self.trained = True

    def evaluate(self, testset):
        """
        Evaluates the model on the test set and returns the RMSE and MAE.
        """
        if not self.trained:
            self.train()

        predictions = self.algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return rmse, mae

    def get_recommendations(self, user_id, top_k=config.TOP_K, exclude_items=None):
        """
        Generates top K recommendations for a given user.

        Parameters:
        - user_id: The user ID for whom to generate recommendations.
        - top_k: The number of recommendations to generate.
        - exclude_items: A list of item IDs to exclude from recommendations (e.g., items the user has already interacted with).
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
