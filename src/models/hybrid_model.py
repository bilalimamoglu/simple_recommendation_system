# src/models/hybrid_model.py

import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from src import config
from src.models.evaluation import precision_at_k, recall_at_k, f_score_at_k

class HybridRecommender:
    def __init__(self, content_recommender, collaborative_recommender, alpha=0.5):
        """
        Initializes the hybrid recommender system.
        """
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.alpha = alpha

    def get_recommendations(self, user_id, top_k=config.TOP_K, exclude_items=None):
        """
        Generates top K hybrid recommendations for a given user.
        """
        # Get content-based scores
        content_scores = self.content_recommender.get_recommendations(user_id, top_k=None, exclude_items=exclude_items)
        content_scores_series = pd.Series([1.0]*len(content_scores), index=content_scores)

        # Get collaborative filtering scores
        collaborative_scores = self.collaborative_recommender.get_predicted_ratings(user_id)

        # Combine the indices
        all_item_ids = set(content_scores_series.index) | set(collaborative_scores.index)

        # Align scores
        content_scores_series = content_scores_series.reindex(all_item_ids).fillna(0)
        collaborative_scores = collaborative_scores.reindex(all_item_ids).fillna(0)

        # Normalize the scores
        scaler = MinMaxScaler()
        content_scores_norm = scaler.fit_transform(content_scores_series.values.reshape(-1, 1)).flatten()
        collaborative_scores_norm = scaler.fit_transform(collaborative_scores.values.reshape(-1, 1)).flatten()

        # Combine the scores
        hybrid_scores = self.alpha * content_scores_norm + (1 - self.alpha) * collaborative_scores_norm
        hybrid_scores_series = pd.Series(hybrid_scores, index=all_item_ids)

        # Exclude items if necessary
        if exclude_items is not None and len(exclude_items) > 0:
            hybrid_scores_series = hybrid_scores_series.drop(exclude_items, errors='ignore')

        # Get the top K recommendations
        top_k_items = hybrid_scores_series.nlargest(top_k).index.tolist()

        return top_k_items

    def train(self):
        """
        Trains the hybrid recommender by training the underlying content-based and collaborative models.
        """
        logging.info("Training Hybrid Recommender...")
        self.content_recommender.train()
        self.collaborative_recommender.train()
        logging.info("Hybrid Recommender training complete.")


    def evaluate(self, user_ids, title_id, relevant_items_dict, top_k=config.TOP_K):
        """
        Evaluates the hybrid recommender using precision, recall, and F-score.

        Parameters:
        - user_ids: List of users to evaluate on.
        - title_id: The item ID to base content-based recommendations on.
        - relevant_items_dict: Dictionary where keys are user_ids and values are lists of relevant items.
        - top_k: Number of recommendations to evaluate.

        Returns:
        - Average precision, recall, and F-score.
        """
        precisions = []
        recalls = []

        for user_id in user_ids:
            relevant_items = relevant_items_dict.get(user_id, [])
            recommendations = self.get_recommendations(user_id, title_id, top_k=top_k)

            precision = precision_at_k(recommendations, relevant_items, k=top_k)
            recall = recall_at_k(recommendations, relevant_items, k=top_k)
            precisions.append(precision)
            recalls.append(recall)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f_score = f_score_at_k(avg_precision, avg_recall)

        return avg_precision, avg_recall, avg_f_score
