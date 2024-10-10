# src/models/content_based.py

import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.models.evaluation import (
    precision_at_k, recall_at_k, f_score_at_k, ndcg_at_k,
    calculate_diversity, calculate_novelty, estimated_ctr,
    calculate_coverage
)
import numpy as np

class ContentBasedRecommender:
    def __init__(self, titles_df: pd.DataFrame, count_matrix, algorithm='cosine'):
        """
        Initializes the content-based recommender.
        """
        self.titles_df = titles_df
        self.count_matrix = count_matrix
        self.algorithm = algorithm
        self.similarity_matrix = None
        self.trained = False

    def train(self):
        """
        Computes the similarity matrix using the specified algorithm.
        """
        logging.info("Training Content-Based Recommender...")
        if self.algorithm == 'cosine':
            self.similarity_matrix = cosine_similarity(self.count_matrix)
        else:
            raise ValueError(f"Algorithm '{self.algorithm}' is not supported.")
        self.trained = True
        logging.info("Content-Based Recommender training complete.")

    def get_recommendations(self, title_id, top_k=10, exclude_items=None, identifier_type='title'):
        """
        Generates top K recommendations for a given title.
        """
        if not self.trained:
            self.train()

        if title_id not in self.titles_df['TITLE_ID'].values:
            logging.warning(f"Title ID {title_id} not found in the dataset.")
            return []

        idx = self.titles_df[self.titles_df['TITLE_ID'] == title_id].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]  # Exclude the title itself

        recommended_indices = [i[0] for i in sim_scores]
        recommended_titles = self.titles_df.iloc[recommended_indices]['TITLE_ID'].tolist()

        if exclude_items:
            recommended_titles = [title for title in recommended_titles if title not in exclude_items]

        return recommended_titles[:top_k]

    def evaluate(self, identifiers, test_df_filtered, top_k=10, identifier_type='user', item_features_dict=None, item_popularity=None, titles_df=None):
        """
        Evaluates the content-based recommender using multiple metrics.

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

            # Get the user's training items
            user_train_items = test_df_filtered[test_df_filtered['BE_ID'] == user_id]['TITLE_ID'].tolist()
            # Note: Since we're evaluating on test interactions, use representative from training
            # Assuming the last interacted item in training is the representative

            # To get representative item, find last item in training interactions
            user_training_interactions = train_df_sampled[train_df_sampled['BE_ID'] == user_id]['TITLE_ID'].tolist()
            if not user_training_interactions:
                continue  # Skip users with no training interactions

            representative_item = user_training_interactions[-1]

            # Get recommendations
            recommendations = self.get_recommendations(
                title_id=representative_item,
                top_k=top_k,
                exclude_items=user_training_interactions,
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
