# src/models/hybrid_model.py

import logging
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import CollaborativeFilteringRecommender
from src import config
from src.models.evaluation import (
    precision_at_k, recall_at_k, f_score_at_k, ndcg_at_k,
    calculate_diversity, calculate_novelty, estimated_ctr,
    calculate_coverage
)

class HybridRecommender:
    def __init__(self, content_recommender: ContentBasedRecommender,
                 collaborative_recommender: CollaborativeFilteringRecommender, alpha=0.5):
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

    def get_recommendations(self, identifier, top_k=config.TOP_K, exclude_items=None, identifier_type='user'):
        """
        Generates top K recommendations based on the identifier type.

        Parameters:
        - identifier: USER_ID for whom to generate recommendations.
        - top_k: The number of recommendations to generate.
        - exclude_items: A list of item IDs to exclude from recommendations.
        - identifier_type: Type of the identifier ('user').

        Returns:
        - A list of recommended TITLE_IDs.
        """
        if identifier_type == 'user':
            # Get collaborative-based recommendations
            collaborative_recs = self.collaborative_recommender.get_recommendations(
                identifier=identifier,
                top_k=top_k,
                exclude_items=exclude_items,
                identifier_type=identifier_type
            )

            # Optionally, integrate content-based recommendations
            # For simplicity, we'll blend the scores if available or concatenate
            # Here, we simply return collaborative recommendations
            return collaborative_recs
        else:
            logging.error("Hybrid Recommender is configured for user-based recommendations only.")
            return []

    def evaluate(self, identifiers, test_df_filtered, top_k=config.TOP_K, identifier_type='user', item_features_dict=None, item_popularity=None, titles_df=None):
        """
        Evaluates the hybrid recommender using multiple metrics.

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
