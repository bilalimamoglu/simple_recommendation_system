# src/models/hybrid_model.py

import logging
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import CollaborativeFilteringRecommender
from src import config
from src.models.evaluation import precision_at_k, recall_at_k, f_score_at_k, ndcg_at_k, calculate_diversity, calculate_novelty, estimated_ctr, calculate_coverage

class HybridRecommender:
    def __init__(self, content_recommender: ContentBasedRecommender, collaborative_recommender: CollaborativeFilteringRecommender, alpha=0.5):
        """
        Initializes the hybrid recommender by combining content-based and collaborative filtering recommenders.

        Parameters:
        - content_recommender: An instance of ContentBasedRecommender.
        - collaborative_recommender: An instance of CollaborativeFilteringRecommender.
        - alpha: Weighting factor to balance content-based and collaborative filtering scores.
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

    def get_recommendations(self, identifier, top_k=config.TOP_K, exclude_items=None, identifier_type='title'):
        """
        Generates top K recommendations based on the identifier type.

        Parameters:
        - identifier: TITLE_ID or USER_ID for whom to generate recommendations.
        - top_k: The number of recommendations to generate.
        - exclude_items: A list of item IDs to exclude from recommendations.
        - identifier_type: Type of the identifier ('title' or 'user').

        Returns:
        - A list of recommended TITLE_IDs.
        """
        if identifier_type == 'title':
            # Get content-based recommendations
            content_scores = self.content_recommender.get_recommendations(identifier, top_k=top_k, exclude_items=exclude_items)
            # Optionally, you can adjust or combine scores here if needed
            return content_scores
        elif identifier_type == 'user':
            # Get collaborative filtering recommendations
            collaborative_scores = self.collaborative_recommender.get_recommendations(identifier, top_k=top_k, exclude_items=exclude_items)
            # Combine with content-based scores if desired
            # For simplicity, we'll return only collaborative scores here
            return collaborative_scores
        else:
            logging.error(f"Unknown identifier_type: {identifier_type}. Must be 'title' or 'user'.")
            return []

    def evaluate(self, identifiers, relevant_items_dict, top_k=config.TOP_K, identifier_type='title'):
        """
        Evaluates the hybrid recommender using precision, recall, and F-score.

        Parameters:
        - identifiers: List of TITLE_IDs or USER_IDs to evaluate on.
        - relevant_items_dict: Dictionary where keys are identifiers and values are lists of relevant items.
        - top_k: Number of recommendations to evaluate.
        - identifier_type: Type of the identifiers ('title' or 'user').

        Returns:
        - Tuple of average precision, recall, and F-score.
        """
        precisions = []
        recalls = []

        for identifier in identifiers:
            relevant_items = relevant_items_dict.get(identifier, [])
            recommendations = self.get_recommendations(identifier, top_k=top_k, exclude_items=None, identifier_type=identifier_type)

            precision = precision_at_k(recommendations, relevant_items, k=top_k)
            recall = recall_at_k(recommendations, relevant_items, k=top_k)
            precisions.append(precision)
            recalls.append(recall)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f_score = f_score_at_k(avg_precision, avg_recall)

        return avg_precision, avg_recall, avg_f_score
