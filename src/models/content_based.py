# src/models/content_based.py

import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity
from src import config
from surprise import Prediction

class ContentBasedRecommender:
    def __init__(self, titles_df, count_matrix):
        self.titles_df = titles_df.reset_index(drop=True)
        self.count_matrix = count_matrix
        self.cosine_sim_matrix = cosine_similarity(self.count_matrix, self.count_matrix)
        self.title_indices = pd.Series(self.titles_df.index, index=self.titles_df['TITLE_ID']).drop_duplicates()

    def get_recommendations(self, title_id, top_k=config.TOP_K):
        """
        Recommends top K similar titles to the given title_id.
        """
        idx = self.title_indices.get(title_id)
        if idx is None:
            return pd.DataFrame()  # Return empty DataFrame if title_id not found

        sim_scores = list(enumerate(self.cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: top_k + 1]
        title_indices = [i[0] for i in sim_scores]
        recommendations = self.titles_df.iloc[title_indices][['TITLE_ID', 'ORIGINAL_TITLE']]
        return recommendations

    def get_similarity_scores(self, title_id):
        """
        Returns similarity scores for all items based on the given title_id.

        Parameters:
        - title_id: The item ID to base similarity scores on.

        Returns:
        - A pandas Series with item IDs as index and similarity scores as values.
        """
        idx = self.title_indices.get(title_id)
        if idx is None:
            logging.warning(f"Title ID {title_id} not found in the dataset.")
            return pd.Series()

        # Get pairwise similarity scores
        sim_scores = list(enumerate(self.cosine_sim_matrix[idx]))

        # Convert to a Series
        sim_scores_series = pd.Series({self.titles_df['TITLE_ID'].iloc[i]: score for i, score in sim_scores})

        return sim_scores_series

    def predict_ratings(self, testset):
        """
        Predicts ratings for user-item pairs in the testset.

        Parameters:
        - testset: A list of (user_id, item_id, actual_rating) tuples.

        Returns:
        - A list of surprise.Prediction objects.
        """
        predictions = []
        for user_id, item_id, actual_rating in testset:
            # For content-based, we can use average similarity scores as predicted ratings
            sim_scores = self.get_similarity_scores(item_id)
            if sim_scores.empty:
                est_rating = 0  # or some default value
            else:
                est_rating = sim_scores.mean()

            # Create a Prediction object
            pred = Prediction(uid=user_id, iid=item_id, r_ui=actual_rating, est=est_rating, details=None)
            predictions.append(pred)

        return predictions
