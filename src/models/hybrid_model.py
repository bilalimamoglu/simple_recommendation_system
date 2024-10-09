# src/models/hybrid_model.py

import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from src import config

class HybridRecommender:
    def __init__(self, content_recommender, collaborative_recommender, alpha=0.5):
        """
        Initializes the hybrid recommender system.

        Parameters:
        - content_recommender: An instance of ContentBasedRecommender.
        - collaborative_recommender: An instance of CollaborativeFilteringRecommender.
        - alpha: Weighting factor for combining scores (0 <= alpha <= 1).
        """
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.alpha = alpha

    def get_recommendations(self, user_id, title_id=None, top_k=config.TOP_K, exclude_items=None):
        """
        Generates top K hybrid recommendations for a given user and item.

        Parameters:
        - user_id: The user ID for whom to generate recommendations.
        - title_id: The item ID to base content-based recommendations on.
        - top_k: The number of recommendations to generate.
        - exclude_items: A list of item IDs to exclude from recommendations.
        """
        # Get content-based scores
        if title_id is not None:
            content_scores = self.content_recommender.get_similarity_scores(title_id)
        else:
            content_scores = pd.Series(0, index=self.content_recommender.titles_df['TITLE_ID'])

        # Get collaborative filtering scores
        collaborative_scores = self.collaborative_recommender.get_predicted_ratings(user_id)

        # Align the indices
        all_item_ids = set(content_scores.index) & set(collaborative_scores.index)
        all_item_ids = list(all_item_ids)  # Convert set to list

        content_scores = content_scores.loc[all_item_ids]
        collaborative_scores = collaborative_scores.loc[all_item_ids]

        # Normalize the scores
        scaler = MinMaxScaler()

        content_scores_values = content_scores.values.reshape(-1, 1)
        collaborative_scores_values = collaborative_scores.values.reshape(-1, 1)

        content_scores_norm = scaler.fit_transform(content_scores_values).flatten()
        collaborative_scores_norm = scaler.fit_transform(collaborative_scores_values).flatten()

        content_scores = pd.Series(content_scores_norm, index=all_item_ids)
        collaborative_scores = pd.Series(collaborative_scores_norm, index=all_item_ids)

        # Combine the scores
        hybrid_scores = self.alpha * content_scores + (1 - self.alpha) * collaborative_scores

        # Exclude items if necessary
        if exclude_items is not None and len(exclude_items) > 0:
            hybrid_scores = hybrid_scores.drop(exclude_items, errors='ignore')

        # Sort the combined scores
        hybrid_scores = hybrid_scores.sort_values(ascending=False)

        # Get the top K recommendations
        top_k_items = hybrid_scores.head(top_k).index.tolist()

        return top_k_items
