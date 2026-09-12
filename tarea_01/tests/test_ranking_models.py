"""Focused tests for the implicit and content-based adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import surprise

from tarea_01.ranking_models import (
    ContentBased,
    ImplicitWrapper,
    bert_features,
    ingredient_tfidf,
)


def build_trainset():
    frame = pd.DataFrame(
        [
            (1, 1, 5),
            (1, 2, 3),
            (2, 2, 4),
            (2, 3, 5),
        ],
        columns=["user_id", "recipe_id", "rating"],
    )
    data = surprise.Dataset.load_from_df(frame, surprise.Reader(rating_scale=(0, 5)))
    return data.build_full_trainset()


class DummyImplicit:
    def fit(self, matrix, show_progress=False):
        self.matrix = matrix.copy()
        self.user_factors = np.arange(matrix.shape[0] * 2, dtype=float).reshape(-1, 2) + 1
        self.item_factors = np.arange(matrix.shape[1] * 2, dtype=float).reshape(-1, 2) + 1


class RankingModelTests(unittest.TestCase):
    def test_implicit_wrapper_binarizes_and_batches_native_scores(self):
        trainset = build_trainset()
        native = DummyImplicit()
        model = ImplicitWrapper(native, binarize_threshold=4).fit(trainset)
        self.assertEqual(native.matrix.nnz, 3)
        self.assertEqual(model.positive_interactions, 3)
        predictions = model.predict_many(1, [1, 2, 999])
        expected = np.dot(
            native.user_factors[trainset.to_inner_uid(1)],
            native.item_factors[trainset.to_inner_iid(1)],
        )
        self.assertEqual(predictions[0].est, expected)
        self.assertEqual(predictions[-1].est, 0)

    def test_content_profiles_use_only_relevant_training_items(self):
        trainset = build_trainset()
        model = ContentBased(
            np.array([1, 2, 3]),
            np.eye(3, dtype=np.float32),
            relevance_threshold=4,
            weighting="rating_excess",
        ).fit(trainset)
        scores = [prediction.est for prediction in model.predict_many(1, [1, 2, 3])]
        self.assertAlmostEqual(scores[0], 1)
        self.assertAlmostEqual(scores[1], 0)
        self.assertAlmostEqual(scores[2], 0)
        self.assertEqual(model.positive_interactions, 3)
        self.assertEqual(model.zero_profile_users, 0)

    def test_ingredient_tfidf_parses_lists_and_reports_zero_rows(self):
        metadata = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "ingredients": ["['rice', 'salt']", "['rice']", "[]"],
            }
        )
        item_ids, matrix, diagnostics = ingredient_tfidf(metadata, min_df=2)
        np.testing.assert_array_equal(item_ids, [1, 2, 3])
        self.assertEqual(matrix.shape, (3, 1))
        self.assertEqual(diagnostics["content_zero_vector_items"], 1)

    def test_bert_loader_accepts_unique_suffixed_dataset_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(
                root / "food_bert_embeddings (2).npy",
                np.array([[3, 4], [0, 2]], dtype=np.float32),
            )
            np.save(root / "food_item_id_to_idx (2).npy", {20: 0, 10: 1})
            item_ids, matrix, diagnostics = bert_features(pd.DataFrame({"id": [10, 20]}), root)
        np.testing.assert_array_equal(item_ids, [10, 20])
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1, 1])
        self.assertEqual(diagnostics["content_dimensions"], 2)
        self.assertEqual(diagnostics["embedding_file"], "food_bert_embeddings (2).npy")


if __name__ == "__main__":
    unittest.main()
