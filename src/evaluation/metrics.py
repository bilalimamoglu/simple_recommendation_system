# src/evaluation/metrics.py

import numpy as np
import pandas as pd

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Calculates Precision@K for a single user.

    Parameters:
    - recommended_items: List of recommended item IDs.
    - relevant_items: List of relevant (test) item IDs.
    - k: Number of top recommendations to consider.

    Returns:
    - Precision@K value.
    """
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_k)
    true_positives = recommended_set.intersection(relevant_set)
    precision = len(true_positives) / k
    return precision

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Calculates Recall@K for a single user.

    Parameters:
    - recommended_items: List of recommended item IDs.
    - relevant_items: List of relevant (test) item IDs.
    - k: Number of top recommendations to consider.

    Returns:
    - Recall@K value.
    """
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_k)
    true_positives = recommended_set.intersection(relevant_set)
    recall = len(true_positives) / len(relevant_set) if relevant_set else 0.0
    return recall

def f_score_at_k(precision, recall, beta=1.0):
    """
    Calculates the F1-Score (or F-beta Score) for a single user.

    Parameters:
    - precision: Precision@K value.
    - recall: Recall@K value.
    - beta: Weight of recall in the combined score.

    Returns:
    - F-beta Score value.
    """
    if precision + recall == 0:
        return 0.0
    f_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    return f_score


def ndcg_at_k(recommended_items, relevant_items, k=10):
    """
    Calculates nDCG@K for a single user.

    Parameters:
    - recommended_items: List of recommended item IDs.
    - relevant_items: List of relevant (test) item IDs.
    - k: Number of top recommendations to consider.

    Returns:
    - nDCG@K value.
    """
    recommended_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)  # Positions are 1-indexed

    # Ideal DCG
    ideal_relevant = sorted(relevant_items, key=lambda x: 1, reverse=True)
    ideal_k = ideal_relevant[:k]
    idcg = 0.0
    for i in range(min(len(ideal_k), k)):
        idcg += 1 / np.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg


def mean_average_precision(recommended_items, relevant_items, k=10):
    """
    Calculates Mean Average Precision@K for a single user.

    Parameters:
    - recommended_items: List of recommended item IDs.
    - relevant_items: List of relevant (test) item IDs.
    - k: Number of top recommendations to consider.

    Returns:
    - Average Precision@K value.
    """
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    score = 0.0
    num_hits = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    average_precision = score / len(relevant_set) if relevant_set else 0.0
    return average_precision

def area_under_roc_curve(recommended_items, relevant_items, all_items):
    """
    Calculates ROC AUC for a single user.

    Parameters:
    - recommended_items: List of recommended item IDs.
    - relevant_items: List of relevant (test) item IDs.
    - all_items: Set or list of all available item IDs.

    Returns:
    - ROC AUC value.
    """
    # Create binary labels
    labels = []
    scores = []
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_items)
    for item in all_items:
        labels.append(1 if item in relevant_set else 0)
        if item in recommended_set:
            scores.append(1)  # Recommended items have higher scores
        else:
            scores.append(0)
    # Handle cases where there are no positive or no negative labels
    if all(label == 0 for label in labels) or all(label == 1 for label in labels):
        return 0.5  # Undefined AUC, return 0.5 (random)
    # Compute AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores)
    return auc
