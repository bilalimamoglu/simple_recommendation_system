# src/models/collaborative_filtering.py

import logging
from surprise import KNNBaseline, SVD, SVDpp, NMF, Dataset
from surprise import accuracy
from src.models.recommender import Recommender
from src import config


class CollaborativeFilteringRecommender(Recommender):
    def __init__(self, data, algorithm='SVD', algo_params=None, user_based=True):
        """
        Initializes the collaborative filtering model using the Surprise library.

        Parameters:
        - data: The Dataset object from Surprise.
        - algorithm: Name of the algorithm to use (e.g., 'SVD', 'KNNBaseline').
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
        logging.info(
            f"Training Collaborative Filtering model using {self.algorithm_name} ({'User-Based' if self.user_based else 'Item-Based'})...")
        self.trainset = self.data.build_full_trainset()
        self.algo.fit(self.trainset)
        self.trained = True
        logging.info("Collaborative Filtering model training complete.")

    def get_recommendations(self, identifier, top_k=10, exclude_items=None, identifier_type='user'):
        """
        Generates top K recommendations based on the identifier type.

        Parameters:
        - identifier: USER_ID (for user-based) or TITLE_ID (for item-based) for whom to generate recommendations.
        - top_k: Number of recommendations to generate.
        - exclude_items: List of item IDs to exclude from recommendations.
        - identifier_type: Type of the identifier ('user' or 'title').

        Returns:
        - List of recommended TITLE_IDs.
        """
        if not self.trained:
            self.train()

        if self.user_based and identifier_type == 'user':
            # User-Based Collaborative Filtering
            user_id = identifier
            if user_id not in self.trainset._raw2inner_id_users:
                logging.warning(f"User ID {user_id} not in the training set.")
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

        elif not self.user_based and identifier_type == 'title':
            # Item-Based Collaborative Filtering
            title_id = identifier
            if title_id not in self.trainset._raw2inner_id_items:
                logging.warning(f"Title ID {title_id} not in the training set.")
                return []

            # Get inner ID for the given title_id
            inner_id = self.trainset.to_inner_iid(title_id)

            # Get top K neighbors
            try:
                neighbors = self.algo.get_neighbors(inner_id, k=top_k + (len(exclude_items) if exclude_items else 0))
            except AttributeError:
                logging.error(f"The algorithm '{self.algorithm_name}' does not support 'get_neighbors' method.")
                return []

            # Convert inner IDs back to raw IDs
            recommended_items = [self.trainset.to_raw_iid(inner_id) for inner_id in neighbors]

            # Exclude specified items
            if exclude_items:
                recommended_items = [item for item in recommended_items if item not in exclude_items]

            # Return top_k items
            return recommended_items[:top_k]

        else:
            logging.error(f"Unsupported identifier_type '{identifier_type}' for algorithm '{self.algorithm_name}'.")
            return []
