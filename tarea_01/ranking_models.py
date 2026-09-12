"""Ranking models shared by the consolidated Tarea 1 experiment.

The adapters keep the evaluation API uniform without changing the instructor's
shared utilities. Implicit models receive binary positive feedback and the
content model builds each profile exclusively from the training interactions.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from surprise import AlgoBase
from threadpoolctl import threadpool_limits


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(term).strip() for term in value if str(term).strip()]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = [part.strip() for part in str(value).split(",")]
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]
    return [str(term).strip().lower() for term in parsed if str(term).strip()]


class ImplicitWrapper(AlgoBase):
    """Adapt an implicit estimator to Surprise IDs and preserve raw scores."""

    def __init__(self, implicit_model, binarize_threshold: float = 4.0):
        super().__init__()
        self.implicit_model = implicit_model
        self.binarize_threshold = float(binarize_threshold)

    def fit(self, trainset):
        super().fit(trainset)
        positives = [
            (u, i) for u, i, rating in trainset.all_ratings() if rating >= self.binarize_threshold
        ]
        if not positives:
            raise ValueError("Implicit ranking requires at least one positive interaction.")
        rows, cols = zip(*positives, strict=True)
        matrix = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(trainset.n_users, trainset.n_items),
        )
        self.positive_interactions = len(positives)
        with threadpool_limits(limits=1, user_api="blas"):
            self.implicit_model.fit(matrix, show_progress=False)
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return 0.0
        return float(
            np.dot(self.implicit_model.user_factors[u], self.implicit_model.item_factors[i])
        )

    def predict_many(self, raw_uid, raw_items):
        """Score one user's candidates in one factor-matrix operation."""
        if raw_uid not in self.trainset._raw2inner_id_users:
            scores = np.zeros(len(raw_items), dtype=float)
        else:
            inner_user = self.trainset.to_inner_uid(raw_uid)
            inner_items = np.array(
                [
                    self.trainset.to_inner_iid(item)
                    if item in self.trainset._raw2inner_id_items
                    else -1
                    for item in raw_items
                ],
                dtype=int,
            )
            scores = np.zeros(len(raw_items), dtype=float)
            known = inner_items >= 0
            if known.any():
                scores[known] = (
                    self.implicit_model.item_factors[inner_items[known]]
                    @ self.implicit_model.user_factors[inner_user]
                )
        return [
            SimpleNamespace(est=float(score), details={"was_impossible": False}) for score in scores
        ]


def ingredient_tfidf(metadata: pd.DataFrame, min_df: int = 1):
    """Build normalized ingredient TF-IDF vectors in metadata row order."""
    if "id" not in metadata or "ingredients" not in metadata:
        raise ValueError("Metadata must contain id and ingredients columns.")
    documents = metadata["ingredients"].map(_parse_list).tolist()
    vectorizer = TfidfVectorizer(
        analyzer=lambda terms: terms,
        lowercase=False,
        min_df=int(min_df),
        dtype=np.float32,
    )
    matrix = normalize(vectorizer.fit_transform(documents), norm="l2", copy=False)
    diagnostics = {
        "content_source": "ingredient_tfidf",
        "content_min_df": int(min_df),
        "content_dimensions": int(matrix.shape[1]),
        "content_items": int(matrix.shape[0]),
        "content_zero_vector_items": int((matrix.getnnz(axis=1) == 0).sum()),
    }
    return metadata["id"].to_numpy(), matrix, diagnostics


def _resolve_dataset_file(dataset_dir: Path, canonical_name: str) -> Path:
    exact = dataset_dir / canonical_name
    if exact.exists():
        return exact
    stem, suffix = exact.stem, exact.suffix
    candidates = sorted(dataset_dir.glob(f"{stem}*{suffix}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected {canonical_name!r} or one suffixed equivalent in {dataset_dir}; "
            f"found {[path.name for path in candidates]}."
        )
    return candidates[0]


def bert_features(metadata: pd.DataFrame, dataset_dir: str | Path):
    """Load only metadata-matched BERT rows from the memory-mapped embedding file."""
    dataset_dir = Path(dataset_dir)
    embeddings_path = _resolve_dataset_file(dataset_dir, "food_bert_embeddings.npy")
    mapping_path = _resolve_dataset_file(dataset_dir, "food_item_id_to_idx.npy")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    mapping = np.load(mapping_path, allow_pickle=True).item()
    if not isinstance(mapping, dict):
        raise ValueError(f"Expected a dictionary in {mapping_path}.")
    item_ids = metadata["id"].to_numpy()
    missing = [item for item in item_ids if item not in mapping]
    if missing:
        raise ValueError(f"BERT mapping is missing {len(missing)} metadata items.")
    indices = np.fromiter((mapping[item] for item in item_ids), dtype=np.int64)
    matrix = np.asarray(embeddings[indices], dtype=np.float32)
    matrix = normalize(matrix, norm="l2", copy=False)
    diagnostics = {
        "content_source": "bert",
        "content_min_df": None,
        "content_dimensions": int(matrix.shape[1]),
        "content_items": int(matrix.shape[0]),
        "content_zero_vector_items": int((np.linalg.norm(matrix, axis=1) == 0).sum()),
        "embedding_file": embeddings_path.name,
        "mapping_file": mapping_path.name,
    }
    return item_ids, matrix, diagnostics


class ContentBased(AlgoBase):
    """Cosine ranking between item vectors and positive-history user profiles."""

    def __init__(
        self,
        item_ids,
        item_features,
        *,
        relevance_threshold: float = 4.0,
        weighting: str = "rating_excess",
    ):
        super().__init__()
        if weighting not in {"binary", "rating_excess"}:
            raise ValueError("weighting must be binary or rating_excess.")
        self.raw_item_ids = np.asarray(item_ids)
        self.source_features = item_features
        self.relevance_threshold = float(relevance_threshold)
        self.weighting = weighting

    def fit(self, trainset):
        super().fit(trainset)
        source_index = {item: index for index, item in enumerate(self.raw_item_ids)}
        missing = [
            trainset.to_raw_iid(inner_item)
            for inner_item in trainset.all_items()
            if trainset.to_raw_iid(inner_item) not in source_index
        ]
        if missing:
            raise ValueError(f"Content features are missing {len(missing)} training items.")
        ordered = np.fromiter(
            (source_index[trainset.to_raw_iid(inner_item)] for inner_item in trainset.all_items()),
            dtype=np.int64,
        )
        if issparse(self.source_features):
            self.item_features = normalize(
                self.source_features[ordered].tocsr(), norm="l2", copy=True
            )
        else:
            self.item_features = normalize(
                np.asarray(self.source_features[ordered], dtype=np.float32),
                norm="l2",
                copy=True,
            )
        positives = [
            (u, i, rating)
            for u, i, rating in trainset.all_ratings()
            if rating >= self.relevance_threshold
        ]
        rows = [u for u, _, _ in positives]
        cols = [i for _, i, _ in positives]
        values = [
            1.0 if self.weighting == "binary" else rating - self.relevance_threshold + 1.0
            for _, _, rating in positives
        ]
        preferences = csr_matrix(
            (np.asarray(values, dtype=np.float32), (rows, cols)),
            shape=(trainset.n_users, trainset.n_items),
        )
        profiles = preferences @ self.item_features
        self.user_profiles = normalize(profiles, norm="l2", copy=False)
        if issparse(self.user_profiles):
            profile_squared_norms = np.asarray(
                self.user_profiles.multiply(self.user_profiles).sum(axis=1)
            ).ravel()
        else:
            profile_squared_norms = np.einsum("ij,ij->i", self.user_profiles, self.user_profiles)
        self.positive_interactions = len(positives)
        self.zero_profile_users = int((profile_squared_norms == 0).sum())
        self._compatibility_cache = {}
        return self

    def _score_inner_items(self, inner_user: int, inner_items: np.ndarray) -> np.ndarray:
        features = self.item_features[inner_items]
        profile = self.user_profiles[inner_user]
        product = features @ profile.T
        if issparse(product):
            return product.toarray().ravel()
        return np.asarray(product).ravel()

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return 0.0
        if u not in self._compatibility_cache:
            inner_items = np.arange(self.trainset.n_items, dtype=int)
            self._compatibility_cache[u] = self._score_inner_items(u, inner_items)
        return float(self._compatibility_cache[u][i])

    def predict_many(self, raw_uid, raw_items):
        scores = np.zeros(len(raw_items), dtype=float)
        if raw_uid in self.trainset._raw2inner_id_users:
            inner_user = self.trainset.to_inner_uid(raw_uid)
            inner_items = np.array(
                [
                    self.trainset.to_inner_iid(item)
                    if item in self.trainset._raw2inner_id_items
                    else -1
                    for item in raw_items
                ],
                dtype=int,
            )
            known = inner_items >= 0
            if known.any():
                scores[known] = self._score_inner_items(inner_user, inner_items[known])
        return [
            SimpleNamespace(est=float(score), details={"was_impossible": False}) for score in scores
        ]
