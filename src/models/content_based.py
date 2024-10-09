# src/models/content_based.py

import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from src import config
from src.models.evaluation import precision_at_k, recall_at_k, f_score_at_k
from scipy.sparse import csr_matrix


class ContentBasedRecommender:
    def __init__(self, titles_df, feature_matrix, algorithm='cosine'):
        """
        Initializes the content-based recommender system.

        Parameters:
        - titles_df: DataFrame containing title information.
        - feature_matrix: Sparse matrix representing item features.
        - algorithm: Similarity algorithm to use (default: 'cosine').
        """
        self.titles_df = titles_df.reset_index(drop=True)
        self.feature_matrix = feature_matrix.tocsr()  # Ensure CSR format
        self.title_ids = self.titles_df['TITLE_ID'].tolist()
        self.similarity_algorithm = algorithm
        self.item_similarity = None  # To be computed during training

    def train(self):
        """
        Trains the content-based recommender by computing the item similarity matrix.
        """
        logging.info("Training Content-Based Recommender...")
        # Normalize the feature matrix
        normalized_features = normalize(self.feature_matrix, axis=1, norm='l2', copy=False)
        # Compute cosine similarity
        self.item_similarity = cosine_similarity(normalized_features, normalized_features)
        logging.info("Content-Based Recommender training complete.")

    def get_recommendations(self, title_id, top_k=config.TOP_K, exclude_items=None):
        """
        Recommends top K similar items to the given title_id.

        Parameters:
        - title_id: The TITLE_ID for which to find similar items.
        - top_k: The number of recommendations to generate.
        - exclude_items: A list of TITLE_IDs to exclude from recommendations.

        Returns:
        - A list of recommended TITLE_IDs.
        """
        if self.item_similarity is None:
            logging.error("Model has not been trained. Call train() before getting recommendations.")
            return []

        if title_id not in self.title_ids:
            logging.error(f"Title ID {title_id} not found in the dataset.")
            return []

        # Get the index of the given title_id
        idx = self.title_ids.index(title_id)
        # Get similarity scores for all items with the given item
        sim_scores = self.item_similarity[idx]
        # Create a Series with TITLE_ID as index
        similarity_scores = pd.Series(sim_scores, index=self.title_ids)
        # Exclude specified items
        if exclude_items and len(exclude_items) > 0:
            similarity_scores = similarity_scores.drop(exclude_items, errors='ignore')
        # Get top K items
        top_k_items = similarity_scores.nlargest(top_k).index.tolist()
        return top_k_items

    def evaluate(self, test_titles, relevant_titles_dict, top_k=config.TOP_K):
        """
        Evaluates the content-based recommender using precision, recall, and F-score.

        Parameters:
        - test_titles: List of TITLE_IDs to evaluate on.
        - relevant_titles_dict: Dictionary where keys are TITLE_IDs and values are lists of relevant TITLE_IDs.
        - top_k: Number of recommendations to evaluate.

        Returns:
        - Tuple of average precision, recall, and F-score.
        """
        precisions = []
        recalls = []

        for title_id in test_titles:
            relevant_items = relevant_titles_dict.get(title_id, [])
            recommendations = self.get_recommendations(title_id, top_k=top_k, exclude_items=None)

            precision = precision_at_k(recommendations, relevant_items, k=top_k)
            recall = recall_at_k(recommendations, relevant_items, k=top_k)
            precisions.append(precision)
            recalls.append(recall)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f_score = f_score_at_k(avg_precision, avg_recall)

        return avg_precision, avg_recall, avg_f_score
