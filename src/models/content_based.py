# src/models/content_based.py

import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from src import config
from src.models.evaluation import precision_at_k, recall_at_k, f_score_at_k
from scipy.sparse import csr_matrix
import numpy as np

class ContentBasedRecommender:
    def __init__(self, titles_df, count_matrix):
        """
        Initializes the content-based recommender system.

        Parameters:
        - titles_df: DataFrame containing title information.
        - count_matrix: Sparse matrix (e.g., CSR) representing item features.
        """
        self.titles_df = titles_df.reset_index(drop=True)
        self.count_matrix = count_matrix.tocsr()  # Ensure CSR format for efficient row slicing
        self.title_ids = self.titles_df['TITLE_ID'].tolist()
        self.user_profiles = None  # Sparse matrix to store user profiles
        self.user_id_mapping = {}   # Maps user_id to row index in user_profiles
        self.item_features = None   # Normalized item features matrix

    def train(self):
        """
        Trains the content-based recommender by normalizing item profiles.
        """
        logging.info("Training Content-Based Recommender...")
        # Normalize the count_matrix along the rows (items) to create item profiles
        self.item_features = normalize(self.count_matrix, axis=1, norm='l2', copy=False)
        logging.info("Content-Based Recommender training complete.")

    def build_user_profiles(self, interactions_df):
        """
        Build user profiles by aggregating normalized item profiles of items the user has interacted with.

        Parameters:
        - interactions_df: DataFrame containing user interactions.
        """
        logging.info("Building user profiles...")

        # Extract unique users and map them to unique indices
        unique_users = interactions_df['BE_ID'].unique()
        num_users = len(unique_users)
        num_items = self.item_features.shape[0]
        num_features = self.item_features.shape[1]

        logging.info(f"Number of unique users: {num_users}")

        # Create a mapping from user_id to index
        self.user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}

        # Initialize lists to construct the sparse user-item interaction matrix
        data = []
        row_indices = []
        col_indices = []

        # Group interactions by user and iterate to build the interaction data
        user_groups = interactions_df.groupby('BE_ID')['TITLE_ID'].apply(list)

        for user_id, item_ids in user_groups.items():
            user_idx = self.user_id_mapping[user_id]
            for item_id in item_ids:
                if item_id in self.title_ids:
                    item_idx = self.title_ids.index(item_id)
                    row_indices.append(user_idx)
                    col_indices.append(item_idx)
                    data.append(1)  # Binary interaction

        # Create a sparse user-item interaction matrix
        user_item_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_users, num_items), dtype=np.float32)

        # Multiply user-item matrix with item features to get user profiles
        # This operation sums the normalized item vectors for each user
        user_profiles = user_item_matrix.dot(self.item_features)

        # Normalize user profiles to unit vectors
        user_profiles = normalize(user_profiles, axis=1, norm='l2', copy=False)

        self.user_profiles = user_profiles.tocsr()  # Ensure CSR format
        logging.info("User profiles built successfully.")

    def get_recommendations(self, user_id, top_k=config.TOP_K, exclude_items=None):
        """
        Recommends top K items to the given user_id based on content similarity.

        Parameters:
        - user_id: The user ID for whom to generate recommendations.
        - top_k: The number of recommendations to generate.
        - exclude_items: A list of item IDs to exclude from recommendations.

        Returns:
        - A list of recommended TITLE_IDs.
        """
        if self.user_profiles is None:
            logging.error("User profiles have not been built. Call build_user_profiles first.")
            return []

        if user_id not in self.user_id_mapping:
            # Changed from logging.warning to logging.debug to reduce log verbosity
            logging.debug(f"User ID {user_id} not found in user profiles.")
            return []

        user_idx = self.user_id_mapping[user_id]
        user_profile = self.user_profiles.getrow(user_idx)

        if user_profile.nnz == 0:
            # Changed from logging.warning to logging.debug to reduce log verbosity
            logging.debug(f"User ID {user_id} has an empty profile.")
            return []

        # Compute cosine similarity between user profile and all item profiles
        similarities = cosine_similarity(user_profile, self.item_features).flatten()

        # Create a Series with TITLE_ID as index
        similarity_scores = pd.Series(similarities, index=self.title_ids)

        # Exclude items if necessary
        if exclude_items and len(exclude_items) > 0:
            similarity_scores = similarity_scores.drop(exclude_items, errors='ignore')

        # Get top K items
        top_k_items = similarity_scores.nlargest(top_k).index.tolist()

        return top_k_items

    def evaluate(self, user_ids, relevant_items_dict, top_k=config.TOP_K):
        """
        Evaluates the content-based recommender using precision, recall, and F-score.

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
