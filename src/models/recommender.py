# src/models/recommender.py

from abc import ABC, abstractmethod


class Recommender(ABC):
    """
    Abstract base class for all recommenders.
    """

    @abstractmethod
    def train(self):
        """
        Trains the recommender model.
        """
        pass

    @abstractmethod
    def get_recommendations(self, identifier, top_k=10, exclude_items=None, identifier_type='user'):
        """
        Generates top K recommendations.

        Parameters:
        - identifier: USER_ID or TITLE_ID for whom to generate recommendations.
        - top_k: Number of recommendations to generate.
        - exclude_items: List of items to exclude from recommendations.
        - identifier_type: Type of the identifier ('user' or 'title').

        Returns:
        - List of recommended item IDs.
        """
        pass
