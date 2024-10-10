# src/models/evaluation.py

import numpy as np

def precision_at_k(recommended_items, relevant_items, k):
    """
    Calculates Precision@K.
    """
    recommended_items = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_items)
    precision = len(recommended_set & relevant_set) / k
    return precision

def recall_at_k(recommended_items, relevant_items, k):
    """
    Calculates Recall@K.
    """
    recommended_items = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_items)
    recall = len(recommended_set & relevant_set) / len(relevant_set) if relevant_set else 0.0
    return recall

def f_score_at_k(precision, recall):
    """
    Calculates F-Score@K.
    """
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def ndcg_at_k(recommended_items, relevant_items, k):
    """
    Calculates NDCG@K.
    """
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), k))])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

def calculate_diversity(recommendations, item_features_dict):
    """
    Calculates the diversity of the recommended items.
    """
    n = len(recommendations)
    diversity = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(item_features_dict[recommendations[i]], item_features_dict[recommendations[j]])
            diversity += 1 - sim
    diversity = (2 * diversity) / (n * (n - 1)) if n > 1 else 0.0
    return diversity

def calculate_novelty(recommendations, popularity_dict):
    """
    Calculates the novelty of the recommended items based on their popularity.
    """
    n = len(recommendations)
    novelty = 0.0
    total_popularity = sum(popularity_dict.values())
    for item in recommendations:
        popularity = popularity_dict.get(item, 1)
        probability = popularity / total_popularity if total_popularity > 0 else 0.0
        novelty += -np.log2(probability) if probability > 0 else 0.0
    novelty = novelty / n if n > 0 else 0.0
    return novelty

def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Return zero similarity if either vector is zero
    return np.dot(vec1, vec2) / (norm1 * norm2)

def estimated_ctr(recommended_items, user_interactions):
    """
    Estimates the CTR as the proportion of recommended items the user has interacted with.
    """
    recommended_set = set(recommended_items)
    interactions_set = set(user_interactions)
    clicked_items = recommended_set & interactions_set
    ctr = len(clicked_items) / len(recommended_items) if recommended_items else 0.0
    return ctr

def calculate_coverage(recommendations, all_items):
    """
    Calculates the coverage of the recommendations.

    Parameters:
    - recommendations: A list of recommended item IDs.
    - all_items: A list or array of all possible item IDs.

    Returns:
    - Coverage as a float.
    """
    unique_recommended_items = set(recommendations)
    total_unique_items = len(set(all_items))
    coverage = len(unique_recommended_items) / total_unique_items
    return coverage
