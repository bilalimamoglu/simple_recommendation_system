# src/models/collaborative_filtering.py

import logging
from surprise import KNNBaseline, SVD, SVDpp, NMF, Dataset
from src import config
import numpy as np

class CollaborativeFilteringRecommender:
    def __init__(self, data, algorithm='SVD', algo_params=None, user_based=True):
        """
        Initializes the collaborative filtering model using the Surprise library.

        Parameters:
        - data: The Dataset object from Surprise.
        - algorithm: Name of the algorithm to use.
        - algo_params: Dictionary of algorithm-specific hyperparameters.
        - user_based: Boolean flag indicating whether to perform user-based (True) or item-based (False) CF.
        """
        self.data = data
        self.algorithm_name = algorithm
        self.algo_params = algo_params if algo_params is not None else {}
        self.user_based = user_based
        self.algo = self.get_algorithm(algorithm)
        self.trained = False
        self.trainset = None

    def get_algorithm(self, algorithm_name):
        """
        Returns a Surprise algorithm object based on the algorithm name and hyperparameters.

        Parameters:
        - algorithm_name: Name of the algorithm.

        Returns:
        - An instance of the specified algorithm.
        """
        algorithms = {
            'KNNBaseline': KNNBaseline,
            'SVD': SVD,
            'SVDpp': SVDpp,
            'NMF': NMF,
        }
        if algorithm_name not in algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' is not supported.")

        # For KNN-based algorithms, set similarity options
        if algorithm_name.startswith('KNN'):
            default_sim_options = {'name': 'cosine', 'user_based': self.user_based}
            sim_options = self.algo_params.get('sim_options', default_sim_options)
            return algorithms[algorithm_name](sim_options=sim_options)
        else:
            # For matrix factorization algorithms like SVD, SVDpp, NMF
            return algorithms[algorithm_name](**self.algo_params)

    def train(self):
        """
        Trains the collaborative filtering model.
        """
        logging.info(f"Training Collaborative Filtering model using {self.algorithm_name} ({'User-Based' if self.user_based else 'Item-Based'})...")
        self.trainset = self.data.build_full_trainset()
        self.algo.fit(self.trainset)
        self.trained = True
        logging.info("Collaborative Filtering model training complete.")

    def get_recommendations(self, identifier, top_k=config.TOP_K, exclude_items=None, identifier_type='user'):
        """
        Generates top K recommendations based on the identifier type.

        Parameters:
        - identifier: TITLE_ID (for item-based) or USER_ID (for user-based) for whom to generate recommendations.
        - top_k: The number of recommendations to generate.
        - exclude_items: A list of item IDs to exclude from recommendations.
        - identifier_type: Type of the identifier ('title' or 'user').

        Returns:
        - A list of recommended TITLE_IDs.
        """
        if not self.trained:
            self.train()

        if identifier_type == 'user' and self.user_based:
            # User-Based Collaborative Filtering
            user_id = identifier
            # Check if user_id exists in the training set
            if user_id not in self.trainset._raw2inner_id_users:
                logging.warning(f"User ID {user_id} is not in the training set.")
                return []

            # Get all items in the training set as inner IDs
            all_items_inner_ids = set(self.trainset.all_items())

            # Exclude items the user has already interacted with
            user_inner_id = self.trainset.to_inner_uid(user_id)
            interacted_items_inner_ids = set([j for (j, _) in self.trainset.ur[user_inner_id]])
            items_to_predict = all_items_inner_ids - interacted_items_inner_ids

            # Generate predictions for all items not yet interacted with
            testset = [(user_id, self.trainset.to_raw_iid(item_inner_id), 0) for item_inner_id in items_to_predict]
            predictions = self.algo.test(testset)

            # Sort predictions by estimated rating
            predictions.sort(key=lambda x: x.est, reverse=True)

            # Get the top K item IDs
            top_k_items = [pred.iid for pred in predictions[:top_k]]

            return top_k_items

        else:
            logging.error(f"Identifier type mismatch or unsupported combination: identifier_type={identifier_type}, user_based={self.user_based}")
            return []
