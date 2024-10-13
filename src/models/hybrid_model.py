# src/models/hybrid_model.py

import logging
from src.models.recommender import Recommender


class HybridRecommender(Recommender):
    def __init__(self, content_recommender, collaborative_recommender, alpha=0.5):
        """
        Initializes the hybrid recommender by combining content-based and collaborative filtering recommenders.

        Parameters:
        - content_recommender: An instance of ContentBasedRecommender.
        - collaborative_recommender: An instance of CollaborativeFilteringRecommender.
        - alpha: Weighting factor to balance content-based and collaborative filtering recommendations.
        """
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.alpha = alpha

    def train(self):
        """
        Trains both the content-based and collaborative filtering recommenders.
        """
        logging.info("Training Hybrid Recommender...")
        self.content_recommender.train()
        self.collaborative_recommender.train()
        logging.info("Hybrid Recommender training complete.")

    def get_recommendations(self, identifier, top_k=10, exclude_items=None, identifier_type='user'):
        """
        Generates top K recommendations by combining content-based and collaborative filtering scores.

        Parameters:
        - identifier: USER_ID for whom to generate recommendations.
        - top_k: Number of recommendations to generate.
        - exclude_items: List of item IDs to exclude from recommendations.
        - identifier_type: Type of the identifier ('user').

        Returns:
        - List of recommended TITLE_IDs.
        """
        if identifier_type != 'user':
            logging.error("Hybrid Recommender only supports identifier_type='user'.")
            return []

        # Get collaborative filtering recommendations
        cf_recs = self.collaborative_recommender.get_recommendations(
            identifier=identifier,
            top_k=top_k * 2,  # Get more items to allow overlap
            exclude_items=exclude_items,
            identifier_type=identifier_type
        )

        # Get the user's training items
        user_train_items = exclude_items if exclude_items else []

        # Assume the last item in training is the representative
        if user_train_items:
            representative_item = user_train_items[-1]
            # Get content-based recommendations
            cb_recs = self.content_recommender.get_recommendations(
                title_id=representative_item,
                top_k=top_k * 2,
                exclude_items=user_train_items
            )
        else:
            cb_recs = []

        # Combine recommendations
        combined_recs = list(set(cf_recs + cb_recs))

        # Exclude already interacted items
        if exclude_items:
            combined_recs = [item for item in combined_recs if item not in exclude_items]

        # Return top K items
        return combined_recs[:top_k]
