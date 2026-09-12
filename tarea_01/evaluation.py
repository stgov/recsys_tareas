"""Reproducible rating and Top-K evaluation shared by the assignment notebooks.

Only observed validation ratings enter RMSE/MAE. Ranking retains *all* validation
items and adds unseen training-catalog candidates whose relevance is unknown and
treated as zero. Metrics are conditional on that candidate protocol. Ties use
ascending item IDs, never validation labels or positive-first insertion order.

This module leaves the team's ``utils.py`` intact. The accuracy formulas agree
with that utility for identical predictions without ties; naming, diagnostics,
raw ranking scores, and auditable per-user outputs are explicit here.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

PROTOCOL_VERSION = "validation-unseen-v2-raw-score-item-id-ties"
INTERACTION_COLUMNS = ["user_id", "recipe_id", "rating"]


def _ordered(values: Any) -> list[Any]:
    values = list(set(values))
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _parse_tags(value: Any) -> frozenset[str] | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(str(tag).strip() for tag in value if str(tag).strip())
    return None


def _mean(values: Any, default: float = float("nan")) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else default


@dataclass
class EvaluationContext:
    """Model-independent inputs; create once and reuse across all configurations."""

    validation: pd.DataFrame
    catalog: list[Any]
    users: list[Any]
    validation_by_user: dict[Any, list[tuple[Any, float]]]
    seen: dict[Any, set[Any]]
    item_order: dict[Any, int]
    item_counts: dict[Any, int]
    train_user_counts: dict[Any, int]
    train_size: int
    train_vectors: csr_matrix
    train_item_index: dict[Any, int]
    item_tags: dict[Any, frozenset[str] | None]
    k: int
    threshold: float
    n_negatives: int | None
    seed: int
    protocol: dict[str, Any]

    def iter_candidates(self) -> Iterator[tuple[Any, list[tuple[Any, float, bool]]]]:
        """Yield one user's candidates, bounded by one catalog even in full mode."""
        rng = np.random.default_rng(self.seed)
        for uid in self.users:
            observed = self.validation_by_user[uid]
            excluded = self.seen.get(uid, set()).union(iid for iid, _ in observed)
            unseen = [iid for iid in self.catalog if iid not in excluded]
            if self.n_negatives is not None and len(unseen) > self.n_negatives:
                indices = rng.choice(len(unseen), self.n_negatives, replace=False)
                unseen = [unseen[int(index)] for index in indices]
            rows = [(iid, rating, True) for iid, rating in observed]
            rows.extend((iid, 0.0, False) for iid in unseen)
            rows.sort(key=lambda row: self.item_order[row[0]])
            yield uid, rows


def prepare_evaluation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
    *,
    k: int = 10,
    threshold: float = 4.0,
    n_negatives: int | None = 100,
    seed: int = 42,
) -> EvaluationContext:
    """Validate the split and prepare a deterministic, model-independent protocol.

    Duplicate validation pairs and overlap with training are rejected because
    their ranking denominators would be ambiguous. Empty validation is rejected.
    A training item with repeated observations still has a binary interaction
    vector for ILD, while novelty uses its actual observation count.
    """
    if k < 1 or (n_negatives is not None and n_negatives < 0):
        raise ValueError("k must be positive and n_negatives nonnegative or None.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive; synthetic negatives have rating 0.")
    for name, frame in (("train", train_df), ("validation", validation_df)):
        missing = set(INTERACTION_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame.empty or frame[INTERACTION_COLUMNS].isna().any().any():
            raise ValueError(f"{name} must be nonempty with non-null IDs and ratings.")
        if not np.isfinite(frame["rating"].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains non-finite ratings.")
    if validation_df.duplicated(["user_id", "recipe_id"]).any():
        raise ValueError("Duplicate validation user-item pairs are not supported.")
    train_pairs = set(train_df[["user_id", "recipe_id"]].itertuples(index=False, name=None))
    val_pairs = set(validation_df[["user_id", "recipe_id"]].itertuples(index=False, name=None))
    if train_pairs.intersection(val_pairs):
        raise ValueError("Train/validation overlap: ranking must exclude seen user-item pairs.")

    catalog = _ordered(train_df["recipe_id"])
    users = _ordered(validation_df["user_id"])
    all_items = _ordered([*catalog, *validation_df["recipe_id"].tolist()])
    item_order = {iid: index for index, iid in enumerate(all_items)}
    user_order = {uid: index for index, uid in enumerate(users)}
    validation = validation_df[INTERACTION_COLUMNS].copy()
    validation = (
        validation.assign(
            _user=validation["user_id"].map(user_order),
            _item=validation["recipe_id"].map(item_order),
        )
        .sort_values(["_user", "_item"])
        .drop(columns=["_user", "_item"])
        .reset_index(drop=True)
    )
    by_user = {
        uid: list(group[["recipe_id", "rating"]].itertuples(index=False, name=None))
        for uid, group in validation.groupby("user_id", sort=False)
    }
    seen = train_df.groupby("user_id", sort=False)["recipe_id"].agg(set).to_dict()
    counts = train_df["recipe_id"].value_counts().to_dict()
    train_users = _ordered(train_df["user_id"])
    train_user_index = {uid: index for index, uid in enumerate(train_users)}
    train_item_index = {iid: index for index, iid in enumerate(catalog)}
    pairs = train_df.drop_duplicates(["user_id", "recipe_id"])
    matrix = csr_matrix(
        (
            np.ones(len(pairs), dtype=np.float64),
            (
                pairs["recipe_id"].map(train_item_index),
                pairs["user_id"].map(train_user_index),
            ),
        ),
        shape=(len(catalog), len(train_users)),
    )
    norms = np.sqrt(np.asarray(matrix.sum(axis=1)).ravel())
    vectors = diags(1.0 / norms).dot(matrix).tocsr()
    item_tags: dict[Any, frozenset[str] | None] = {}
    if metadata_df is not None:
        id_column = "recipe_id" if "recipe_id" in metadata_df.columns else "id"
        if id_column not in metadata_df or "tags" not in metadata_df:
            raise ValueError("metadata must have id/recipe_id and tags columns.")
        if metadata_df[id_column].duplicated().any():
            raise ValueError("metadata contains duplicate item IDs.")
        item_tags = {
            iid: _parse_tags(tags)
            for iid, tags in metadata_df[[id_column, "tags"]].itertuples(index=False, name=None)
        }

    # Hashes also reveal accidental comparisons across different data snapshots.
    def frame_hash(frame: pd.DataFrame) -> str:
        hashes = pd.util.hash_pandas_object(frame[INTERACTION_COLUMNS], index=False)
        return hashlib.sha256(np.sort(hashes.to_numpy()).tobytes()).hexdigest()

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "ranking_candidate_mode": (
            "full_catalog" if n_negatives is None else f"negative_sample_{n_negatives}"
        ),
        "k": k,
        "relevance_threshold": threshold,
        "n_negatives": n_negatives,
        "seed": seed,
        "sampler": "numpy.default_rng; sorted users and catalog; without replacement",
        "tie_break": "score descending, raw item ID ascending",
        "validation_items": "all observed validation items, including ratings below threshold",
        "unobserved_relevance": "unknown, treated as zero for offline ranking",
        "accuracy_population": "macro mean over users with at least one relevant validation item",
        "beyond_accuracy_population": "all evaluated users",
        "precision_denominator": k,
        "map_denominator": "min(k, number of relevant validation items)",
        "novelty_probability": "training item observation count / training observation count",
        "interaction_ild": "mean pairwise 1-cosine of binary training item-user vectors",
        "tag_diversity": "mean distinct raw metadata tags in each recommendation list",
        "unknown_metadata": "excluded from tag union, coverage reported",
        "unknown_train_items": "excluded from ILD pairs and novelty, coverage reported",
        "train_sha256": frame_hash(train_df),
        "validation_sha256": frame_hash(validation_df),
        "train_interactions": len(train_df),
        "validation_interactions": len(validation_df),
        "catalog_size": len(catalog),
        "users_evaluated": len(users),
    }
    return EvaluationContext(
        validation,
        catalog,
        users,
        by_user,
        seen,
        item_order,
        counts,
        train_df.groupby("user_id").size().to_dict(),
        len(train_df),
        vectors,
        train_item_index,
        item_tags,
        k,
        threshold,
        n_negatives,
        seed,
        protocol,
    )


def build_ranking_testset(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    n_negatives: int | None = 100,
    seed: int = 42,
) -> list[tuple[Any, Any, float]]:
    """Compatibility helper; use evaluate_model for memory-bounded full ranking."""
    context = prepare_evaluation(train_df, validation_df, n_negatives=n_negatives, seed=seed)
    return [
        (uid, iid, rating)
        for uid, candidates in context.iter_candidates()
        for iid, rating, _ in candidates
    ]


@dataclass
class EvaluationResult:
    metrics: dict[str, float | int]
    per_user: pd.DataFrame
    top_k: pd.DataFrame
    rating_predictions: pd.DataFrame
    protocol: dict[str, Any]

    def export(self, output_dir: str | Path, prefix: str) -> None:
        """Persist auditable outputs; no model refitting or prediction is performed."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([self.metrics]).to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
        self.per_user.to_csv(output_dir / f"{prefix}_per_user.csv", index=False)
        self.top_k.to_csv(output_dir / f"{prefix}_top_k.csv", index=False)
        self.rating_predictions.to_csv(output_dir / f"{prefix}_rating_predictions.csv", index=False)
        (output_dir / f"{prefix}_protocol.json").write_text(
            json.dumps(self.protocol, indent=2, ensure_ascii=False, default=_json_scalar) + "\n",
            encoding="utf-8",
        )


def evaluate_model(
    model: Any,
    train_df: pd.DataFrame | None = None,
    validation_df: pd.DataFrame | None = None,
    metadata_df: pd.DataFrame | None = None,
    *,
    context: EvaluationContext | None = None,
    k: int = 10,
    threshold: float = 4.0,
    n_negatives: int | None = 100,
    seed: int = 42,
    ranking_clip: bool = False,
    rating_metrics: bool = True,
) -> EvaluationResult:
    """Score each pair once, using a batch scorer when the model provides one.

    Raw scores rank candidates; rating estimates are clipped to the model's
    training scale, matching Surprise's rating prediction convention. Set
    ``rating_metrics=False`` for uncalibrated implicit scorers (RMSE/MAE become
    NaN). A supplied context owns k, threshold, sample size, and seed.

    Only one user's candidates and all Top-K outputs are retained, so full
    catalog mode does not materialize the user-by-catalog prediction matrix.
    The scorer receives identifiers only: validation labels are never passed.
    """
    if context is None:
        if train_df is None or validation_df is None:
            raise ValueError("Supply context or both train_df and validation_df.")
        context = prepare_evaluation(
            train_df,
            validation_df,
            metadata_df,
            k=k,
            threshold=threshold,
            n_negatives=n_negatives,
            seed=seed,
        )
    k = context.k
    lower, upper = getattr(getattr(model, "trainset", None), "rating_scale", (0.0, 5.0))
    per_user_rows, top_rows, rating_rows = [], [], []
    total_candidates = total_fallbacks = total_clipped = 0
    all_actual_k: list[int] = []
    used_batch_scoring = False
    candidate_hasher = hashlib.sha256()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for uid, candidates in context.iter_candidates():
        rows = []
        raw_items = [iid for iid, _, _ in candidates]
        predict_many = getattr(model, "predict_many", None)
        if callable(predict_many):
            # Only raw identifiers are passed; validation labels remain hidden.
            predictions = list(predict_many(uid, raw_items))
            used_batch_scoring = True
        else:
            predictions = [model.predict(uid, iid, clip=False) for iid in raw_items]
        if len(predictions) != len(candidates):
            raise ValueError(
                f"The scorer returned {len(predictions)} predictions for "
                f"{len(candidates)} candidates belonging to user={uid!r}."
            )
        for (iid, true_rating, is_validation), prediction in zip(
            candidates, predictions, strict=True
        ):
            raw_score = float(prediction.est)
            if not np.isfinite(raw_score):
                raise ValueError(f"Non-finite score for user={uid!r}, item={iid!r}.")
            rating_est = float(np.clip(raw_score, lower, upper))
            details = prediction.details or {}
            actual_k = details.get("actual_k")
            row = {
                "user_id": uid,
                "recipe_id": iid,
                "true_rating": true_rating,
                "is_validation": is_validation,
                "is_relevant": bool(is_validation and true_rating >= context.threshold),
                "score": rating_est if ranking_clip else raw_score,
                "raw_score": raw_score,
                "rating_est": rating_est,
                "was_impossible": bool(details.get("was_impossible", False)),
                "actual_k": actual_k,
                "outside_rating_scale": raw_score != rating_est,
            }
            rows.append(row)
            if is_validation:
                rating_rows.append(row.copy())
            if actual_k is not None:
                all_actual_k.append(int(actual_k))
            candidate_hasher.update(
                json.dumps(
                    [uid, iid, true_rating, is_validation],
                    default=_json_scalar,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        rows.sort(key=lambda row: (-row["score"], context.item_order[row["recipe_id"]]))
        selected = rows[:k]
        relevant = sum(row["is_relevant"] for row in rows)
        hit_flags = np.array([row["is_relevant"] for row in selected], dtype=float)
        hits = int(hit_flags.sum())
        cumulative_precision = np.cumsum(hit_flags) / np.arange(1, len(selected) + 1)
        ideal_hits = min(k, relevant)
        idcg = float(discounts[:ideal_hits].sum())
        tie_at_k = len(rows) > k and rows[k - 1]["score"] == rows[k]["score"]
        score_counts = pd.Series([row["score"] for row in rows]).value_counts()
        tied_candidates = int(score_counts[score_counts > 1].sum())
        fallback_count = sum(row["was_impossible"] for row in rows)
        user_actual_k = [row["actual_k"] for row in rows if row["actual_k"] is not None]
        total_candidates += len(rows)
        total_fallbacks += fallback_count
        total_clipped += sum(row["outside_rating_scale"] for row in rows)
        known_indices = []
        novelty = []
        tags: set[str] = set()
        tags_known = 0
        for rank, row in enumerate(selected, 1):
            iid = row["recipe_id"]
            count = context.item_counts.get(iid, 0)
            info = -np.log2(count / context.train_size) if count else float("nan")
            if count:
                known_indices.append(context.train_item_index[iid])
                novelty.append(float(info))
            item_tags = context.item_tags.get(iid)
            if item_tags is not None:
                tags.update(item_tags)
                tags_known += 1
            top_rows.append(
                {
                    **row,
                    "rank": rank,
                    "train_count": count,
                    "self_information_bits": info,
                    "metadata_tags_known": item_tags is not None,
                }
            )
        ild = float("nan")
        if len(known_indices) > 1:
            summed = np.asarray(context.train_vectors[known_indices].sum(axis=0)).ravel()
            n_known = len(known_indices)
            mean_similarity = (float(summed @ summed) - n_known) / (n_known * (n_known - 1))
            ild = float(np.clip(1.0 - mean_similarity, 0.0, 1.0))
        per_user_rows.append(
            {
                "user_id": uid,
                "Candidates": len(rows),
                "RelevantItems": relevant,
                "HasRelevant": relevant > 0,
                "RecommendedItems": len(selected),
                "Hits": hits,
                f"Precision@{k}": hits / k if relevant else 0.0,
                f"Recall@{k}": hits / relevant if relevant else 0.0,
                f"NDCG@{k}": float(hit_flags @ discounts[: len(selected)]) / idcg if idcg else 0.0,
                f"MAP@{k}": float(hit_flags @ cumulative_precision) / ideal_hits
                if ideal_hits
                else 0.0,
                "InteractionILD": ild,
                "TagDiversity": len(tags) if context.item_tags else float("nan"),
                "TagMetadataCoverage": tags_known / len(selected),
                "NoveltyBitsPerItem": _mean(novelty),
                "NoveltyBitsPerList": (
                    float(sum(novelty)) if len(novelty) == len(selected) else float("nan")
                ),
                "NoveltyKnownItemFraction": len(novelty) / len(selected),
                "RankingFallbackFraction": fallback_count / len(rows),
                "RankingTiedCandidateFraction": tied_candidates / len(rows),
                f"RankingTieAt{k}": bool(tie_at_k),
                "ActualKMean": _mean(user_actual_k),
                "ActualKMin": min(user_actual_k) if user_actual_k else float("nan"),
                "TrainInteractions": context.train_user_counts.get(uid, 0),
            }
        )
    per_user = pd.DataFrame(per_user_rows)
    top_k_frame = pd.DataFrame(top_rows)
    ratings = pd.DataFrame(rating_rows)
    errors = ratings["rating_est"] - ratings["true_rating"]
    eligible = per_user.loc[per_user["HasRelevant"]]
    recommended_known = set(top_k_frame["recipe_id"]).intersection(context.catalog)
    metrics: dict[str, float | int] = {
        "RMSE": float(np.sqrt(np.mean(errors**2))) if rating_metrics else float("nan"),
        "MAE": float(np.mean(np.abs(errors))) if rating_metrics else float("nan"),
        **{
            f"{name}@{k}": _mean(eligible[f"{name}@{k}"], default=0.0)
            for name in ("Precision", "Recall", "NDCG", "MAP")
        },
        **{
            name: _mean(per_user[name])
            for name in (
                "InteractionILD",
                "TagDiversity",
                "TagMetadataCoverage",
                "NoveltyBitsPerList",
                "NoveltyKnownItemFraction",
                "RankingTiedCandidateFraction",
            )
        },
        "NoveltyBitsPerItem": _mean(top_k_frame["self_information_bits"]),
        "CatalogCoverage": len(recommended_known) / len(context.catalog),
        "UsersEvaluated": len(per_user),
        "UsersWithRelevant": len(eligible),
        "UsersWithILDPairs": int(per_user["InteractionILD"].notna().sum()),
        "RatingObservations": len(ratings),
        "RankingCandidates": total_candidates,
        "CandidatesPerUserMean": float(per_user["Candidates"].mean()),
        "RecommendedItemsPerUserMean": float(per_user["RecommendedItems"].mean()),
        "RatingFallbackFraction": float(ratings["was_impossible"].mean()),
        "RankingFallbackFraction": total_fallbacks / total_candidates,
        f"RankingTieAt{k}Fraction": float(per_user[f"RankingTieAt{k}"].mean()),
        "RatingOutsideScaleFraction": float(ratings["outside_rating_scale"].mean()),
        "RankingOutsideScaleFraction": total_clipped / total_candidates,
        "ActualKObservedPredictions": len(all_actual_k),
        "ActualKMean": _mean(all_actual_k),
        "ActualKMedian": float(np.median(all_actual_k)) if all_actual_k else float("nan"),
        "ActualKMin": min(all_actual_k) if all_actual_k else float("nan"),
        "ActualKMax": max(all_actual_k) if all_actual_k else float("nan"),
    }
    protocol = {
        **context.protocol,
        "candidate_sha256": candidate_hasher.hexdigest(),
        "ranking_score": "clipped to rating scale" if ranking_clip else "raw, clip=False",
        "rating_score": (
            "raw score clipped to trainset.rating_scale"
            if rating_metrics
            else "not calibrated; RMSE/MAE omitted"
        ),
        "rating_scale": [lower, upper],
        "scoring_api": "predict_many" if used_batch_scoring else "predict",
    }
    return EvaluationResult(metrics, per_user, top_k_frame, ratings, protocol)


def paired_bootstrap(
    first: EvaluationResult | pd.DataFrame,
    second: EvaluationResult | pd.DataFrame,
    metric: str = "NDCG@10",
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Paired user bootstrap CI for first-minus-second on relevant users only.

    This is uncertainty across the fixed validation users, not across training
    seeds, hyperparameter search, or candidate samplings. No significance claim
    is made for configurations selected on this same validation set.
    """
    left = first.per_user if isinstance(first, EvaluationResult) else first
    right = second.per_user if isinstance(second, EvaluationResult) else second
    if isinstance(first, EvaluationResult) and isinstance(second, EvaluationResult):
        for key in ("candidate_sha256", "ranking_score", "relevance_threshold", "k"):
            if first.protocol[key] != second.protocol[key]:
                raise ValueError(f"Incompatible evaluation protocols: {key}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    if set(left["user_id"]) != set(right["user_id"]):
        raise ValueError("Paired bootstrap requires identical user populations.")
    paired = left[["user_id", "HasRelevant", metric]].merge(
        right[["user_id", "HasRelevant", metric]],
        on="user_id",
        validate="one_to_one",
        suffixes=("_first", "_second"),
    )
    if not paired["HasRelevant_first"].equals(paired["HasRelevant_second"]):
        raise ValueError("Relevance eligibility differs between evaluations.")
    paired = paired.loc[paired["HasRelevant_first"]]
    differences = (paired[f"{metric}_first"] - paired[f"{metric}_second"]).to_numpy()
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        raise ValueError("No eligible users for paired bootstrap.")
    rng = np.random.default_rng(seed)
    means = [
        float(rng.choice(differences, size=len(differences), replace=True).mean())
        for _ in range(n_resamples)
    ]
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "metric": metric,
        "difference_first_minus_second": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "users": len(differences),
        "n_resamples": n_resamples,
        "seed": seed,
    }
