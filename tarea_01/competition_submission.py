"""Generate and validate the two Tarea 1 competition submissions.

The official rating track uses a 1--5 scale, so rows with rating zero in the
development data are treated as out-of-scale observations for this bonus only.
The report experiments remain unchanged and continue to describe their own
protocol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import surprise
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

TASK_DIR = Path(__file__).resolve().parent
DATASET_DIR = TASK_DIR / "dataset"
OUTPUT_DIR = TASK_DIR / "bonus_submission"
RATING_FILENAME = "competicion_rating_completado.csv"
RANKING_FILENAME = "competicion_ranking.json"
SEED = 42
TOP_K = 10


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rating_models(train: pd.DataFrame):
    valid = train.loc[train.rating.between(1, 5), ["user_id", "recipe_id", "rating"]]
    trainset = surprise.Dataset.load_from_df(
        valid,
        surprise.Reader(rating_scale=(1, 5)),
    ).build_full_trainset()
    baseline = surprise.BaselineOnly(
        bsl_options={"method": "als", "reg_u": 5, "reg_i": 10, "n_epochs": 20},
        verbose=False,
    ).fit(trainset)
    svd = surprise.SVD(
        n_factors=20,
        n_epochs=30,
        lr_all=0.003,
        reg_all=0.1,
        biased=True,
        random_state=SEED,
    ).fit(trainset)
    return baseline, svd


def _rating_predictions(models, pairs: pd.DataFrame) -> np.ndarray:
    baseline, svd = models
    testset = [
        (row.user_id, row.recipe_id, float(getattr(row, "rating", 0.0)))
        for row in pairs.itertuples(index=False)
    ]
    baseline_scores = np.fromiter(
        (prediction.est for prediction in baseline.test(testset)),
        dtype=float,
        count=len(testset),
    )
    svd_scores = np.fromiter(
        (prediction.est for prediction in svd.test(testset)),
        dtype=float,
        count=len(testset),
    )
    return np.clip(0.5 * baseline_scores + 0.5 * svd_scores, 1.0, 5.0)


def _error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "MAE": float(np.mean(np.abs(error))),
        "observations": len(actual),
    }


def _validate_rating_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    rating_template: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    valid = validation.loc[validation.rating.between(1, 5)].copy()
    predicted = _rating_predictions(_rating_models(train), valid)
    overall = _error_metrics(valid.rating.to_numpy(dtype=float), predicted)

    target_users = set(rating_template.user_id)
    proxy = (
        valid.loc[valid.user_id.isin(target_users)]
        .assign(_date=pd.to_datetime(valid.loc[valid.user_id.isin(target_users), "date"]))
        .sort_values(["user_id", "_date"])
        .groupby("user_id", sort=False)
        .tail(1)
    )
    proxy_positions = (
        pd.Series(np.arange(len(valid)), index=valid.index).loc[proxy.index].to_numpy()
    )
    return {
        "validation_ratings_1_to_5": overall,
        "last_validation_rating_per_target_user": _error_metrics(
            proxy.rating.to_numpy(dtype=float),
            predicted[proxy_positions],
        ),
    }


def generate_rating_submission(
    interactions: pd.DataFrame,
    template: pd.DataFrame,
) -> pd.DataFrame:
    submission = template.copy()
    submission["rating"] = _rating_predictions(_rating_models(interactions), template)
    return submission


def _ranking_inputs(interactions: pd.DataFrame, target_users: list[int]) -> dict[str, Any]:
    users = np.asarray(sorted(interactions.user_id.unique()), dtype=np.int64)
    items = np.asarray(sorted(interactions.recipe_id.unique()), dtype=np.int64)
    user_index = pd.Series(np.arange(len(users)), index=users)
    item_index = pd.Series(np.arange(len(items)), index=items)
    interaction_users = user_index.loc[interactions.user_id].to_numpy(dtype=np.int64)
    interaction_items = item_index.loc[interactions.recipe_id].to_numpy(dtype=np.int64)
    target_inner_users = user_index.loc[target_users].to_numpy(dtype=np.int64)
    seen = csr_matrix(
        (
            np.ones(len(interactions), dtype=np.int8),
            (interaction_users, interaction_items),
        ),
        shape=(len(users), len(items)),
    )
    positives = interactions.rating.ge(4).to_numpy()
    positive_matrix = csr_matrix(
        (
            np.ones(int(positives.sum()), dtype=np.float32),
            (interaction_users[positives], interaction_items[positives]),
        ),
        shape=seen.shape,
    )
    return {
        "users": users,
        "items": items,
        "target_inner_users": target_inner_users,
        "interaction_items": interaction_items,
        "seen": seen,
        "positive_matrix": positive_matrix,
    }


def _ranking_scores(interactions: pd.DataFrame, prepared: dict[str, Any]) -> np.ndarray:
    with threadpool_limits(limits=1, user_api="blas"):
        model = AlternatingLeastSquares(
            factors=20,
            regularization=0.01,
            alpha=10.0,
            iterations=30,
            random_state=SEED,
            num_threads=1,
            use_gpu=False,
        )
        model.fit(prepared["positive_matrix"], show_progress=False)
        collaborative = model.user_factors[prepared["target_inner_users"]] @ model.item_factors.T

    collaborative = collaborative.astype(np.float32, copy=False)
    collaborative = (collaborative - collaborative.mean(axis=1, keepdims=True)) / (
        collaborative.std(axis=1, keepdims=True) + 1e-7
    )

    dates = pd.to_datetime(interactions.date)
    ages = (dates.max() - dates).dt.days.to_numpy()
    recency_weights = np.exp2(-ages / 545.0)
    popularity = np.bincount(
        prepared["interaction_items"],
        weights=recency_weights,
        minlength=len(prepared["items"]),
    ).astype(np.float32)
    popularity = np.log1p(popularity)
    popularity = (popularity - popularity.mean()) / (popularity.std() + 1e-7)
    return 0.5 * collaborative + 0.5 * popularity[None, :]


def _top_k_unseen(scores: np.ndarray, prepared: dict[str, Any]) -> np.ndarray:
    scores = scores.copy()
    target_seen = prepared["seen"][prepared["target_inner_users"]].tocoo()
    scores[target_seen.row, target_seen.col] = -np.inf
    if np.any(np.isfinite(scores).sum(axis=1) < TOP_K):
        raise ValueError("At least one target user has fewer than ten unseen catalog items.")

    partition = np.argpartition(scores, -TOP_K, axis=1)[:, -TOP_K:]
    recommendations = np.empty((len(scores), TOP_K), dtype=np.int64)
    items = prepared["items"]
    for row_number, candidates in enumerate(partition):
        cutoff = scores[row_number, candidates].min()
        eligible = np.flatnonzero(scores[row_number] >= cutoff)
        order = np.lexsort((items[eligible], -scores[row_number, eligible]))
        recommendations[row_number] = items[eligible[order[:TOP_K]]]
    return recommendations


def generate_ranking_submission(
    interactions: pd.DataFrame,
    target_keys: list[str],
) -> dict[str, list[int]]:
    target_users = [int(user_id) for user_id in target_keys]
    prepared = _ranking_inputs(interactions, target_users)
    recommendations = _top_k_unseen(_ranking_scores(interactions, prepared), prepared)
    return {
        raw_user: [int(item) for item in row]
        for raw_user, row in zip(target_keys, recommendations, strict=True)
    }


def _ranking_metrics(
    recommendations: dict[str, list[int]],
    validation: pd.DataFrame,
) -> dict[str, float]:
    relevant = validation.groupby("user_id").recipe_id.agg(set).to_dict()
    discounts = 1.0 / np.log2(np.arange(2, TOP_K + 2))
    ndcg, recall = [], []
    for raw_user, row in recommendations.items():
        user_relevant = relevant.get(int(raw_user), set())
        if not user_relevant:
            continue
        hits = np.fromiter((item in user_relevant for item in row), dtype=float, count=TOP_K)
        ideal = discounts[: min(TOP_K, len(user_relevant))].sum()
        ndcg.append(float(np.dot(hits, discounts) / ideal))
        recall.append(float(hits.sum() / min(TOP_K, len(user_relevant))))
    return {
        "NDCG@10": float(np.mean(ndcg)),
        "Recall@10": float(np.mean(recall)),
        "users_with_validation_ground_truth": len(ndcg),
    }


def _validate_rating_submission(template: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["user_id", "recipe_id", "rating"]:
        raise ValueError("Rating submission must contain user_id, recipe_id, rating in that order.")
    if not submission[["user_id", "recipe_id"]].equals(template[["user_id", "recipe_id"]]):
        raise ValueError("Rating submission keys or row order differ from the official template.")
    ratings = submission.rating.to_numpy(dtype=float)
    if not np.isfinite(ratings).all() or not np.logical_and(ratings >= 1, ratings <= 5).all():
        raise ValueError("Rating predictions must be finite and inside [1, 5].")
    if submission[["user_id", "recipe_id"]].duplicated().any():
        raise ValueError("Rating submission contains duplicate user-item pairs.")


def _validate_ranking_submission(
    template: dict[str, list[int]],
    submission: dict[str, list[int]],
    interactions: pd.DataFrame,
) -> None:
    if list(submission) != list(template):
        raise ValueError("Ranking submission keys or key order differ from the official template.")
    catalog = set(int(item) for item in interactions.recipe_id.unique())
    seen = interactions.groupby("user_id").recipe_id.agg(set).to_dict()
    for raw_user, items in submission.items():
        if len(items) != TOP_K or len(set(items)) != TOP_K:
            raise ValueError(f"User {raw_user} does not have ten unique recommendations.")
        if not set(items).issubset(catalog):
            raise ValueError(f"User {raw_user} has recommendations outside the known catalog.")
        if set(items).intersection(seen.get(int(raw_user), set())):
            raise ValueError(f"User {raw_user} has already-seen recommendations.")


def main(dataset_dir: Path = DATASET_DIR, output_dir: Path = OUTPUT_DIR) -> Path:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.csv"
    validation_path = dataset_dir / "validation.csv"
    rating_template_path = dataset_dir / "rating_template_fill.csv"
    ranking_template_path = dataset_dir / "ranking_template_fill.json"
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    interactions = pd.concat([train, validation], ignore_index=True)
    rating_template = pd.read_csv(rating_template_path)
    ranking_template = json.loads(ranking_template_path.read_text(encoding="utf-8"))

    print("Validating rating ensemble on train -> validation...", flush=True)
    rating_validation = _validate_rating_model(train, validation, rating_template)
    print("Validating ranking ensemble on the full training catalog...", flush=True)
    validation_ranking = generate_ranking_submission(train, list(ranking_template))
    ranking_validation = _ranking_metrics(validation_ranking, validation)

    print("Refitting both tracks on train + validation...", flush=True)
    rating_submission = generate_rating_submission(interactions, rating_template)
    ranking_submission = generate_ranking_submission(interactions, list(ranking_template))
    _validate_rating_submission(rating_template, rating_submission)
    _validate_ranking_submission(ranking_template, ranking_submission, interactions)

    rating_output = output_dir / RATING_FILENAME
    ranking_output = output_dir / RANKING_FILENAME
    rating_submission.to_csv(rating_output, index=False)
    _write_json(ranking_output, ranking_submission)
    manifest = {
        "submitted": False,
        "official_scoring_page": "https://contest-recsys.vercel.app/scoring",
        "templates": {
            "rating_rows": len(rating_template),
            "ranking_users": len(ranking_template),
        },
        "rating": {
            "file": RATING_FILENAME,
            "model": "0.5 * BaselineOnly + 0.5 * FunkSVD",
            "fit_split": "train.csv + validation.csv; only ratings in [1, 5]",
            "clip": [1, 5],
            "development_evidence": rating_validation,
        },
        "ranking": {
            "file": RANKING_FILENAME,
            "model": "0.5 * ALS implicit + 0.5 * time-decayed popularity",
            "fit_split": "train.csv + validation.csv",
            "positive_threshold": 4,
            "candidates": "full known catalog excluding train + validation history",
            "top_k": TOP_K,
            "tie_break": "score descending, recipe_id ascending",
            "development_evidence": ranking_validation,
        },
        "source_sha256": {
            path.name: _sha256(path)
            for path in (train_path, validation_path, rating_template_path, ranking_template_path)
        },
        "output_sha256": {
            rating_output.name: _sha256(rating_output),
            ranking_output.name: _sha256(ranking_output),
        },
    }
    _write_json(output_dir / "submission_manifest.json", manifest)
    print(f"Validated competition files written to {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
