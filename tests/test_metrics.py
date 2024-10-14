# tests/test_metrics.py

import unittest
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f_score_at_k,
    ndcg_at_k,
    mean_average_precision,
    area_under_roc_curve
)

class TestMetrics(unittest.TestCase):
    def test_precision_at_k(self):
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = ['item2', 'item4', 'item6']
        precision = precision_at_k(recommended, relevant, k=5)
        expected = 2 / 5  # item2 and item4 are relevant
        self.assertAlmostEqual(precision, expected)

    def test_recall_at_k(self):
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = ['item2', 'item4', 'item6']
        recall = recall_at_k(recommended, relevant, k=5)
        expected = 2 / 3  # 2 out of 3 relevant items are recommended
        self.assertAlmostEqual(recall, expected)

    def test_f_score_at_k(self):
        precision = 0.4
        recall = 0.6666666667
        f1 = f_score_at_k(precision, recall)
        expected = (2 * 0.4 * 0.6666666667) / (0.4 + 0.6666666667)
        self.assertAlmostEqual(f1, expected)

    def test_ndcg_at_k(self):
        recommended = ['item3', 'item1', 'item2', 'item4', 'item5']
        relevant = ['item2', 'item4']
        ndcg = ndcg_at_k(recommended, relevant, k=3)
        # DCG@3 = 1 / log2(4) for item2
        # IDCG@3 = 1 + 1 / log2(3) for item2 and item4
        expected = 0.5 / 1.63093
        self.assertAlmostEqual(ndcg, expected, places=4)

    def test_mean_average_precision(self):
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = ['item2', 'item4']
        map_k = mean_average_precision(recommended, relevant, k=5)
        # Precision at rank 2: 1/2
        # Precision at rank 4: 2/4
        # MAP@5 = (1/2 + 2/4) / 2 = 0.5 + 0.5 / 2 = 0.5
        expected = 0.5
        self.assertAlmostEqual(map_k, expected)

    def test_auc(self):
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = ['item2', 'item4']
        all_items = set(['item1', 'item2', 'item3', 'item4', 'item5'])
        auc = area_under_roc_curve(recommended, relevant, all_items)
        # Labels: [0,1,0,1,0]
        # Scores: [1,1,1,1,1]
        # Since all recommended items have the same score, AUC should be 0.5
        expected = 0.5
        self.assertAlmostEqual(auc, expected)

if __name__ == '__main__':
    unittest.main()
