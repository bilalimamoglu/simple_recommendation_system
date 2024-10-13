# src/models/evaluation.py

import numpy as np
from sklearn.metrics import roc_auc_score

def precision_at_k(recommended_items, relevant_items, k):
    """
    Calculates Precision@K.
    """
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_k)
    intersection = recommended_set.intersection(relevant_set)
    precision = len(intersection) / k
    return precision

def recall_at_k(recommended_items, relevant_items, k):
    """
    Calculates Recall@K.
    """
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_k)
    intersection = recommended_set.intersection(relevant_set)
    recall = len(intersection) / len(relevant_set) if len(relevant_set) > 0 else 0.0
    return recall

def f_score_at_k(precision, recall):
    """
    Calculates F1-Score@K.
    """
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def ndcg_at_k(recommended_items, relevant_items, k):
    """
    Calculates Normalized Discounted Cumulative Gain (nDCG) at K.
    """
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), k))])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

def mean_average_precision(recommended_items, relevant_items):
    """
    Calculates Mean Average Precision (MAP).
    """
    ap_sum = 0.0
    num_hits = 0
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            num_hits += 1
            ap_sum += num_hits / (i + 1)
    map_score = ap_sum / len(relevant_items) if relevant_items else 0.0
    return map_score

def area_under_roc_curve(recommended_items, relevant_items, all_items):
    """
    Calculates Area Under the ROC Curve (AUC).
    """
    y_true = [1 if item in relevant_items else 0 for item in all_items]
    y_scores = [1 if item in recommended_items else 0 for item in all_items]
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.0
    return auc

def coverage_at_k(recommended_items, all_items):
    """
    Calculates Coverage at K.

    Parameters:
    - recommended_items: List of recommended item IDs.
    - all_items: List or array of all possible item IDs.

    Returns:
    - Coverage as a float.
    """
    unique_recommended_items = set(recommended_items)
    total_unique_items = len(set(all_items))
    coverage = len(unique_recommended_items) / total_unique_items
    return coverage
