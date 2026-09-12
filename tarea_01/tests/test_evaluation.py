"""Small counterexamples for evaluation semantics; run with unittest discovery."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tarea_01.evaluation import evaluate_model, paired_bootstrap, prepare_evaluation


def interactions(rows):
    return pd.DataFrame(rows, columns=["user_id", "recipe_id", "rating"])


class Scores:
    def __init__(self, scores=None, default=1.0):
        self.scores = scores or {}
        self.default = default
        self.trainset = SimpleNamespace(rating_scale=(0, 5))
        self.calls = []

    def predict(self, uid, iid, r_ui=None, clip=True):
        # A true label must never be available to the model at evaluation time.
        assert r_ui is None
        self.calls.append((uid, iid, clip))
        return SimpleNamespace(
            est=self.scores.get((uid, iid), self.default),
            details={"was_impossible": iid == 3, "actual_k": 2},
        )


class BatchScores(Scores):
    def predict(self, uid, iid, r_ui=None, clip=True):
        raise AssertionError("The scalar API must not be used when predict_many exists.")

    def predict_many(self, uid, raw_items):
        self.calls.append((uid, tuple(raw_items)))
        return [
            SimpleNamespace(
                est=self.scores.get((uid, iid), self.default),
                details={"was_impossible": False},
            )
            for iid in raw_items
        ]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.train = interactions([(1, 1, 5), (2, 2, 4), (2, 3, 3), (3, 4, 5), (3, 2, 5)])
        self.validation = interactions([(1, 4, 5), (1, 2, 2)])
        self.metadata = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "tags": ["['x']", "['a', 'b']", "['b', 'c']", "['d']"],
            }
        )

    def test_candidates_keep_nonrelevant_and_exclude_seen(self):
        context = prepare_evaluation(self.train, self.validation, n_negatives=100)
        uid, rows = next(context.iter_candidates())
        self.assertEqual(uid, 1)
        self.assertEqual(rows, [(2, 2, True), (3, 0.0, False), (4, 5, True)])

    def test_sample_is_stable_under_row_permutation(self):
        train = interactions([(2, iid, 4) for iid in range(30)] + [(1, 99, 3)])
        val = interactions([(1, 3, 2), (1, 8, 5), (4, 7, 4)])
        a = prepare_evaluation(train, val, n_negatives=5)
        b = prepare_evaluation(
            train.sample(frac=1, random_state=3),
            val.iloc[::-1],
            n_negatives=5,
        )
        self.assertEqual(list(a.iter_candidates()), list(b.iter_candidates()))
        self.assertEqual(a.protocol["train_sha256"], b.protocol["train_sha256"])
        self.assertEqual(a.protocol["validation_sha256"], b.protocol["validation_sha256"])

    def test_ties_use_item_id_never_positives_first(self):
        first = evaluate_model(Scores(), self.train, self.validation, k=1)
        self.assertEqual(first.top_k["recipe_id"].tolist(), [2])
        self.assertEqual(first.metrics["Recall@1"], 0)
        self.assertEqual(first.metrics["RankingTieAt1Fraction"], 1)
        swapped = self.validation.copy()
        swapped["rating"] = [2, 5]
        second = evaluate_model(Scores(), self.train, swapped, k=1)
        self.assertEqual(first.top_k["recipe_id"].tolist(), second.top_k["recipe_id"].tolist())

    def test_exact_metrics_rating_is_separate_and_pairs_scored_once(self):
        model = Scores({(1, 2): 4, (1, 3): 3, (1, 4): 5})
        result = evaluate_model(model, self.train, self.validation, k=2)
        self.assertAlmostEqual(result.metrics["RMSE"], np.sqrt(2))
        self.assertEqual(result.metrics["MAE"], 1)
        self.assertEqual(result.metrics["Recall@2"], 1)
        self.assertEqual(result.metrics["Precision@2"], 0.5)
        self.assertEqual(result.metrics["NDCG@2"], 1)
        self.assertEqual(result.metrics["MAP@2"], 1)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(len(set(model.calls)), 3)
        self.assertEqual(result.metrics["RatingObservations"], 2)

    def test_batch_scoring_preserves_metrics_and_hides_labels(self):
        scores = {(1, 2): 4, (1, 3): 3, (1, 4): 5}
        scalar = evaluate_model(Scores(scores), self.train, self.validation, k=2)
        model = BatchScores(scores)
        batch = evaluate_model(model, self.train, self.validation, k=2)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0], (1, (2, 3, 4)))
        self.assertEqual(batch.protocol["scoring_api"], "predict_many")
        for metric in ("RMSE", "MAE", "Recall@2", "NDCG@2", "MAP@2"):
            self.assertAlmostEqual(batch.metrics[metric], scalar.metrics[metric])
        pd.testing.assert_frame_equal(
            batch.top_k[["user_id", "recipe_id", "score", "rank"]],
            scalar.top_k[["user_id", "recipe_id", "score", "rank"]],
        )

    def test_accuracy_uses_relevant_users_beyond_accuracy_uses_everyone(self):
        val = interactions([(1, 4, 5), (4, 2, 2)])
        result = evaluate_model(Scores({(1, 4): 5}), self.train, val, k=1)
        self.assertEqual(result.metrics["UsersEvaluated"], 2)
        self.assertEqual(result.metrics["UsersWithRelevant"], 1)
        self.assertEqual(result.metrics["Recall@1"], 1)
        val["rating"] = 2
        result = evaluate_model(Scores(), self.train, val, k=1)
        for name in ("Precision@1", "Recall@1", "NDCG@1", "MAP@1"):
            self.assertEqual(result.metrics[name], 0)
        self.assertEqual(result.metrics["UsersWithRelevant"], 0)

    def test_ild_tags_and_both_novelty_scales_are_distinct(self):
        result = evaluate_model(Scores(), self.train, self.validation, self.metadata, k=2)
        self.assertEqual(result.top_k["recipe_id"].tolist(), [2, 3])
        # Item 2 has users {2,3}; item 3 has {2}: cosine=1/sqrt(2).
        self.assertAlmostEqual(result.metrics["InteractionILD"], 1 - 1 / np.sqrt(2))
        self.assertEqual(result.metrics["TagDiversity"], 3)
        expected_list = -np.log2(2 / 5) - np.log2(1 / 5)
        self.assertAlmostEqual(result.metrics["NoveltyBitsPerList"], expected_list)
        self.assertAlmostEqual(result.metrics["NoveltyBitsPerItem"], expected_list / 2)
        self.assertEqual(result.metrics["CatalogCoverage"], 0.5)
        self.assertEqual(result.metrics["TagMetadataCoverage"], 1)

    def test_raw_scores_preserve_order_clipped_rating_is_reused(self):
        model = Scores({(1, 2): 6, (1, 3): -3, (1, 4): 7})
        raw = evaluate_model(model, self.train, self.validation, k=1)
        self.assertEqual(raw.top_k["recipe_id"].tolist(), [4])
        self.assertEqual(raw.metrics["MAE"], 1.5)
        self.assertEqual(raw.metrics["RankingOutsideScaleFraction"], 1)
        clipped = evaluate_model(model, self.train, self.validation, k=1, ranking_clip=True)
        self.assertEqual(clipped.top_k["recipe_id"].tolist(), [2])
        implicit = evaluate_model(model, self.train, self.validation, k=1, rating_metrics=False)
        self.assertTrue(np.isnan(implicit.metrics["RMSE"]))
        self.assertTrue(np.isnan(implicit.metrics["MAE"]))

    def test_fallback_and_neighbor_diagnostics(self):
        result = evaluate_model(Scores(), self.train, self.validation)
        self.assertAlmostEqual(result.metrics["RankingFallbackFraction"], 1 / 3)
        self.assertEqual(result.metrics["RatingFallbackFraction"], 0)
        self.assertEqual(result.metrics["ActualKObservedPredictions"], 3)
        self.assertEqual(result.metrics["ActualKMean"], 2)

    def test_unknown_metadata_and_single_item_ild_are_not_perfect_diversity(self):
        result = evaluate_model(Scores(), self.train, self.validation, k=1)
        self.assertTrue(np.isnan(result.metrics["TagDiversity"]))
        self.assertTrue(np.isnan(result.metrics["InteractionILD"]))
        self.assertEqual(result.metrics["UsersWithILDPairs"], 0)
        self.assertEqual(result.metrics["TagMetadataCoverage"], 0)

    def test_duplicate_pairs_and_split_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_evaluation(self.train, pd.concat([self.validation, self.validation]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare_evaluation(self.train, interactions([(1, 1, 5)]))

    def test_full_catalog_matches_saturated_sample(self):
        full = evaluate_model(Scores(), self.train, self.validation, n_negatives=None)
        sampled = evaluate_model(Scores(), self.train, self.validation, n_negatives=100)
        pd.testing.assert_frame_equal(full.top_k, sampled.top_k)
        self.assertEqual(full.protocol["candidate_sha256"], sampled.protocol["candidate_sha256"])

    def test_bootstrap_identical_models_is_zero_and_rejects_different_scores_protocol(self):
        first = evaluate_model(Scores(), self.train, self.validation)
        interval = paired_bootstrap(first, first, n_resamples=25)
        self.assertEqual(interval["ci95_low"], 0)
        self.assertEqual(interval["ci95_high"], 0)
        clipped = evaluate_model(Scores(), self.train, self.validation, ranking_clip=True)
        with self.assertRaisesRegex(ValueError, "protocol"):
            paired_bootstrap(first, clipped)


if __name__ == "__main__":
    unittest.main()
