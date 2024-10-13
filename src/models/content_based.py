# src/models/content_based.py

import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBasedRecommender:
    def __init__(self, titles_df: pd.DataFrame, tfidf_matrix_svd, idx_to_title_id):
        """
        Initializes the content-based recommender.

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
        logging.info("Training Content-Based Recommender...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix_svd, self.tfidf_matrix_svd)
        self.trained = True
        logging.info("Content-Based Recommender training complete.")

    def get_recommendations(self, title_id, top_k=10, exclude_items=None):
        """
        Generates top K recommendations for a given title.

        Parameters:
        - title_id: TITLE_ID for which to generate recommendations.
        - top_k: Number of recommendations to generate.
        - exclude_items: List of TITLE_IDs to exclude from recommendations.

        Returns:
        - List of recommended TITLE_IDs.
        """
        if not self.trained:
            self.train()

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
