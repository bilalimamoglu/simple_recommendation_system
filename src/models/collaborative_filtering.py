# src/models/collaborative_filtering.py

import logging
import pandas as pd
from surprise import KNNBaseline, SVD, SVDpp, NMF, Dataset
from src import config
from src.models.evaluation import (
    precision_at_k, recall_at_k, f_score_at_k, ndcg_at_k,
    calculate_diversity, calculate_novelty, estimated_ctr,
    calculate_coverage
)
import numpy as np

class CollaborativeFilteringRecommender:
    def __init__(self, data, algorithm='KNNBaseline', algo_params=None, user_based=True):
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

    @staticmethod
    def get_algorithm_static(algorithm_name, algo_params, user_based):
        """
        Static method to get an algorithm instance based on the algorithm name and type.

        Parameters:
        - algorithm_name: Name of the algorithm.
        - algo_params: Hyperparameters for the algorithm.
        - user_based: Boolean flag indicating CF type.

        Returns:
        - An instance of the specified algorithm.
        """
        algorithms = {
            'KNNBaseline': KNNBaseline,
            'SVD': SVD,
            'SVDpp': SVDpp,
            'NMF': NMF,
            # Add other algorithms if needed
        }
        if algorithm_name not in algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' is not supported.")

        # For KNN-based algorithms, set similarity options
        if algorithm_name.startswith('KNN'):
            default_sim_options = {'name': 'cosine', 'user_based': user_based}
            sim_options = algo_params.get('sim_options', default_sim_options)
            return algorithms[algorithm_name](**algo_params)
        else:
            # For matrix factorization algorithms like SVD, SVDpp, NMF
            return algorithms[algorithm_name](**algo_params)

    def get_algorithm(self, algorithm_name):
        """
        Returns a Surprise algorithm object based on the algorithm name and hyperparameters.

        Parameters:
        - algorithm_name: Name of the algorithm.

        Returns:
        - An instance of the specified algorithm.
        """
        return self.get_algorithm_static(algorithm_name, self.algo_params, self.user_based)

    def train(self):
        """
        Trains the collaborative filtering model.
        """
        logging.info(f"Training Collaborative Filtering model using {self.algorithm_name} ({'User-Based' if self.user_based else 'Item-Based'})...")
        self.trainset = self.data.build_full_trainset()
        self.algo.fit(self.trainset)
        self.trained = True
        logging.info("Collaborative Filtering model training complete.")

    def get_recommendations(self, identifier, top_k=config.TOP_K, exclude_items=None, identifier_type='title'):
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

        if identifier_type == 'title' and not self.user_based:
            # Item-Based Collaborative Filtering
            title_id = identifier
            # Check if title_id exists in the training set
            if title_id not in self.trainset._raw2inner_id_items:
                logging.warning(f"Title ID {title_id} is not in the training set.")
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

        elif identifier_type == 'user' and self.user_based:
            # User-Based Collaborative Filtering
            user_id = identifier
            # Check if user_id exists in the training set
            if user_id not in self.trainset._raw2inner_id_users:
                logging.warning(f"User ID {user_id} is not in the training set.")
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

        else:
            logging.error(f"Identifier type mismatch or unsupported combination: identifier_type={identifier_type}, user_based={self.user_based}")
            return []

    def evaluate(self, identifiers, test_df_filtered, top_k=10, identifier_type='user', item_features_dict=None, item_popularity=None, titles_df=None):
        """
        Evaluates the collaborative filtering recommender using multiple metrics.

        Parameters:
        - identifiers: List of USER_IDs to evaluate.
        - test_df_filtered: The filtered test DataFrame containing interactions for common users.
        - top_k: Number of recommendations to evaluate.
        - identifier_type: Type of the identifiers ('user').
        - item_features_dict: Dictionary mapping ITEM_ID to feature vectors.
        - item_popularity: Dictionary mapping ITEM_ID to popularity counts.
        - titles_df: DataFrame containing all titles.

        Returns:
        - Dictionary containing average metrics.
        """
        precisions = []
        recalls = []
        f_scores = []
        ndcgs = []
        diversities = []
        novelties = []
        coverages = []
        ctrs = []

        for user_id in identifiers:
            # Get the items the user interacted with in the test set
            user_test_items = test_df_filtered[test_df_filtered['BE_ID'] == user_id]['TITLE_ID'].tolist()

            if not user_test_items:
                continue  # Skip users with no test interactions

            # Get recommendations
            recommendations = self.get_recommendations(
                identifier=user_id,
                top_k=top_k,
                exclude_items=None,
                identifier_type=identifier_type
            )

            # Calculate metrics
            precision = precision_at_k(recommendations, user_test_items, k=top_k)
            recall = recall_at_k(recommendations, user_test_items, k=top_k)
            f_score = f_score_at_k(precision, recall)
            ndcg = ndcg_at_k(recommendations, user_test_items, k=top_k)
            diversity = calculate_diversity(recommendations, item_features_dict)
            novelty = calculate_novelty(recommendations, item_popularity)
            coverage = calculate_coverage(recommendations, titles_df['TITLE_ID'].unique())
            ctr = estimated_ctr(recommendations, user_test_items)

            # Append metrics
            precisions.append(precision)
            recalls.append(recall)
            f_scores.append(f_score)
            ndcgs.append(ndcg)
            diversities.append(diversity)
            novelties.append(novelty)
            coverages.append(coverage)
            ctrs.append(ctr)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f_score = sum(f_scores) / len(f_scores) if f_scores else 0.0
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        avg_diversity = sum(diversities) / len(diversities) if diversities else 0.0
        avg_novelty = sum(novelties) / len(novelties) if novelties else 0.0
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
        avg_ctr = sum(ctrs) / len(ctrs) if ctrs else 0.0

        return {
            'Precision@K': avg_precision,
            'Recall@K': avg_recall,
            'F-Score@K': avg_f_score,
            'NDCG@K': avg_ndcg,
            'Diversity': avg_diversity,
            'Novelty': avg_novelty,
            'Coverage': avg_coverage,
            'Estimated CTR': avg_ctr
        }
