# src/models/content_based.py

import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.models.recommender import Recommender


class ContentBasedRecommender(Recommender):
    def __init__(self, titles_df: pd.DataFrame, tfidf_matrix_svd, idx_to_title_id):
        """
        Initializes the standard content-based recommender.

        Parameters:
        - titles_df: DataFrame containing titles data.
        - tfidf_matrix_svd: Reduced TF-IDF matrix after applying SVD.
        - idx_to_title_id: Mapping from indices to TITLE_IDs.
        """
        self.titles_df = titles_df
        self.tfidf_matrix_svd = tfidf_matrix_svd
        self.idx_to_title_id = idx_to_title_id
        self.similarity_matrix = None
        self.trained = False

    def train(self):
        """
        Computes the similarity matrix using cosine similarity.
        """
        logging.info("Training Standard Content-Based Recommender...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix_svd, self.tfidf_matrix_svd)
        self.trained = True
        logging.info("Standard Content-Based Recommender training complete.")

    def get_recommendations(self, title_id, top_k=10, exclude_items=None, identifier_type='title'):
        """
        Generates top K recommendations for a given title.

        Parameters:
        - title_id: TITLE_ID for which to generate recommendations.
        - top_k: Number of recommendations to generate.
        - exclude_items: List of TITLE_IDs to exclude from recommendations.
        - identifier_type: Type of the identifier ('title').

        Returns:
        - List of recommended TITLE_IDs.
        """
        if not self.trained:
            self.train()

        if identifier_type != 'title':
            logging.error("ContentBasedRecommender only supports identifier_type='title'.")
            return []

        if title_id not in self.titles_df['TITLE_ID'].values:
            logging.warning(f"Title ID {title_id} not found in the dataset.")
            return []

        idx = self.titles_df[self.titles_df['TITLE_ID'] == title_id].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:]  # Exclude the title itself

        # Get indices of the most similar titles
        recommended_indices = [i[0] for i in sim_scores]

        # Map indices back to TITLE_IDs
        recommended_titles = [self.idx_to_title_id.get(i) for i in recommended_indices]

        # Exclude specified items
        if exclude_items:
            recommended_titles = [title for title in recommended_titles if title not in exclude_items]

        # Return top_k items
        return recommended_titles[:top_k]


class WeightedContentBasedRecommender(Recommender):
    def __init__(self, titles_df: pd.DataFrame, tfidf_matrix_svd, idx_to_title_id):
        """
        Initializes the weighted content-based recommender.

        Parameters:
        - titles_df: DataFrame containing titles data.
        - tfidf_matrix_svd: Reduced weighted TF-IDF matrix after applying SVD.
        - idx_to_title_id: Mapping from indices to TITLE_IDs.
        """
        self.titles_df = titles_df
        self.tfidf_matrix_svd = tfidf_matrix_svd
        self.idx_to_title_id = idx_to_title_id
        self.similarity_matrix = None
        self.trained = False

    def train(self):
        """
        Computes the similarity matrix using cosine similarity.
        """
        logging.info("Training Weighted Content-Based Recommender...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix_svd, self.tfidf_matrix_svd)
        self.trained = True
        logging.info("Weighted Content-Based Recommender training complete.")

    def get_recommendations(self, title_id, top_k=10, exclude_items=None, identifier_type='title'):
        """
        Generates top K recommendations for a given title using weighted features.

        Parameters:
        - title_id: TITLE_ID for which to generate recommendations.
        - top_k: Number of recommendations to generate.
        - exclude_items: List of TITLE_IDs to exclude from recommendations.
        - identifier_type: Type of the identifier ('title').

        Returns:
        - List of recommended TITLE_IDs.
        """
        if not self.trained:
            self.train()

        if identifier_type != 'title':
            logging.error("WeightedContentBasedRecommender only supports identifier_type='title'.")
            return []

        if title_id not in self.titles_df['TITLE_ID'].values:
            logging.warning(f"Title ID {title_id} not found in the dataset.")
            return []

        idx = self.titles_df[self.titles_df['TITLE_ID'] == title_id].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:]  # Exclude the title itself

        # Get indices of the most similar titles
        recommended_indices = [i[0] for i in sim_scores]

        # Map indices back to TITLE_IDs
        recommended_titles = [self.idx_to_title_id.get(i) for i in recommended_indices]

        # Exclude specified items
        if exclude_items:
            recommended_titles = [title for title in recommended_titles if title not in exclude_items]

        # Return top_k items
        return recommended_titles[:top_k]
