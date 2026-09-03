"""Run ItemKNN and FunkSVD experiments for Tarea 1.

This script is meant to be the reproducible backbone for our part of the
collaborative work. It trains on ``train.csv`` and reports every metric on
``validation.csv``, as required by the assignment.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import surprise
from surprise import accuracy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tarea_01.utils import calculate_metrics  # noqa: E402


def parse_optional_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "full", "all"}:
        return None
    return int(value)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_data(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(dataset_dir / "train.csv")
    validation_df = pd.read_csv(dataset_dir / "validation.csv")
    return train_df, validation_df


def build_surprise_sets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[surprise.Trainset, list[tuple[Any, Any, float]]]:
    reader = surprise.Reader(rating_scale=(0, 5))
    train_data = surprise.Dataset.load_from_df(
        train_df[["user_id", "recipe_id", "rating"]],
        reader=reader,
    )
    validation_data = surprise.Dataset.load_from_df(
        validation_df[["user_id", "recipe_id", "rating"]],
        reader=reader,
    )
    trainset = train_data.build_full_trainset()
    testset = validation_data.construct_testset(validation_data.raw_ratings)
    return trainset, testset


def build_ranking_testset(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    n_negatives: int | None,
    seed: int,
) -> list[tuple[int, int, float]]:
    """Build validation positives plus sampled negatives for Top-K evaluation.

    The shared ``calculate_metrics`` ranks only the items present in the
    provided testset. To avoid evaluating Top-K over validation positives only,
    we add unseen training-catalog items as artificial negatives with rating 0.
    """
    ranking_testset = [
        (int(row.user_id), int(row.recipe_id), float(row.rating))
        for row in validation_df[["user_id", "recipe_id", "rating"]].itertuples(index=False)
    ]

    rng = np.random.default_rng(seed)
    catalog_items = np.array(sorted(map(int, train_df["recipe_id"].unique())), dtype=int)
    train_user_items = train_df.groupby("user_id")["recipe_id"].apply(
        lambda values: set(map(int, values))
    )
    validation_user_items = validation_df.groupby("user_id")["recipe_id"].apply(
        lambda values: set(map(int, values))
    )

    if n_negatives is None:
        print("Building full ranking candidate set; this can be slow and memory intensive.")

    for user_id, validation_items in validation_user_items.items():
        seen_items = train_user_items.get(user_id, set()).union(validation_items)
        candidate_negatives = [
            item_id for item_id in catalog_items if int(item_id) not in seen_items
        ]
        if n_negatives is None:
            sampled_negatives = candidate_negatives
        else:
            sample_size = min(n_negatives, len(candidate_negatives))
            sampled_negatives = (
                rng.choice(
                    np.array(candidate_negatives, dtype=int),
                    size=sample_size,
                    replace=False,
                ).tolist()
                if sample_size > 0
                else []
            )

        ranking_testset.extend((int(user_id), int(item_id), 0.0) for item_id in sampled_negatives)

    return ranking_testset


def prediction_metrics(
    model: surprise.AlgoBase,
    testset: list[tuple[Any, Any, float]],
) -> dict[str, float]:
    predictions = model.test(testset)
    return {
        "RMSE": round(float(accuracy.rmse(predictions, verbose=False)), 4),
        "MAE": round(float(accuracy.mae(predictions, verbose=False)), 4),
    }


def train_item_knn(
    trainset: surprise.Trainset,
    k: int,
    min_k: int,
    sim_name: str,
) -> surprise.KNNBasic:
    model = surprise.KNNBasic(
        k=k,
        min_k=min_k,
        sim_options={"name": sim_name, "user_based": False},
        verbose=False,
    )
    model.fit(trainset)
    return model


def train_funksvd(
    trainset: surprise.Trainset,
    n_factors: int,
    n_epochs: int,
    lr_all: float,
    reg_all: float,
    biased: bool,
    random_state: int,
) -> surprise.SVD:
    model = surprise.SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        biased=biased,
        random_state=random_state,
    )
    model.fit(trainset)
    return model


def maybe_limit_ranking_users(validation_df: pd.DataFrame, max_users: int | None) -> pd.DataFrame:
    if max_users is None:
        return validation_df
    selected_users = validation_df["user_id"].drop_duplicates().head(max_users)
    return validation_df.loc[validation_df["user_id"].isin(selected_users)].copy()


def model_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row["method"] == "ItemKNN":
        return (row["method"], row["k_neighbors"], row["min_k"], row["similarity"])
    return (
        row["method"],
        row["n_factors"],
        row["n_epochs"],
        row["lr_all"],
        row["reg_all"],
        row["biased"],
    )


def run_experiments(args: argparse.Namespace) -> None:
    train_df, validation_df = load_data(args.dataset_dir)
    trainset, testset = build_surprise_sets(train_df, validation_df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    item_rows: list[dict[str, Any]] = []
    trained_models: dict[tuple[Any, ...], surprise.AlgoBase] = {}
    best_item: tuple[str, surprise.AlgoBase, dict[str, Any]] | None = None

    if not args.item_k_values:
        raise ValueError("item_k_values must include at least one value.")
    if not args.funk_factors:
        raise ValueError("funk_factors must include at least one value.")

    for k, min_k, sim_name in product(args.item_k_values, args.item_min_k_values, args.item_sims):
        row = {
            "method": "ItemKNN",
            "k_neighbors": k,
            "min_k": min_k,
            "similarity": sim_name,
        }
        try:
            model = train_item_knn(trainset, k=k, min_k=min_k, sim_name=sim_name)
            metrics = prediction_metrics(model, testset)
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "RMSE": None,
                    "MAE": None,
                }
            )
            print(f"ItemKNN failed for k={k}, min_k={min_k}, sim={sim_name}: {exc}")
            item_rows.append(row)
            continue

        row.update({"status": "ok", "error": None, **metrics})
        item_rows.append(row)
        trained_models[model_key(row)] = model
        if best_item is None or row["RMSE"] < best_item[2]["RMSE"]:
            best_item = ("ItemKNN", model, row)

    item_results = pd.DataFrame(item_rows)
    item_results.to_csv(args.output_dir / "item_based_sensitivity.csv", index=False)

    funk_rows: list[dict[str, Any]] = []
    best_funk: tuple[str, surprise.AlgoBase, dict[str, Any]] | None = None

    funk_grid = product(
        args.funk_factors,
        args.funk_lr_all_values,
        args.funk_reg_all_values,
        args.funk_biased_values,
    )
    for n_factors, lr_all, reg_all, biased in funk_grid:
        row = {
            "method": "FunkSVD",
            "n_factors": n_factors,
            "n_epochs": args.funk_epochs,
            "lr_all": lr_all,
            "reg_all": reg_all,
            "biased": biased,
        }
        try:
            model = train_funksvd(
                trainset,
                n_factors=n_factors,
                n_epochs=args.funk_epochs,
                lr_all=lr_all,
                reg_all=reg_all,
                biased=biased,
                random_state=args.random_state,
            )
            metrics = prediction_metrics(model, testset)
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "RMSE": None,
                    "MAE": None,
                }
            )
            print(
                "FunkSVD failed for "
                f"factors={n_factors}, lr={lr_all}, reg={reg_all}, biased={biased}: {exc}"
            )
            funk_rows.append(row)
            continue

        row.update({"status": "ok", "error": None, **metrics})
        funk_rows.append(row)
        trained_models[model_key(row)] = model
        if best_funk is None or row["RMSE"] < best_funk[2]["RMSE"]:
            best_funk = ("FunkSVD", model, row)

    funk_results = pd.DataFrame(funk_rows)
    funk_results.to_csv(args.output_dir / "funksvd_sensitivity.csv", index=False)

    print("\nItemKNN sensitivity")
    print(item_results.to_string(index=False))
    print("\nFunkSVD sensitivity")
    print(funk_results.to_string(index=False))

    prediction_results = pd.concat([item_results, funk_results], ignore_index=True, sort=False)
    prediction_results.to_csv(args.output_dir / "item_funksvd_prediction_metrics.csv", index=False)

    if args.skip_ranking:
        return

    if best_item is None or best_funk is None:
        raise ValueError("Both item_k_values and funk_factors must include at least one value.")

    ranking_validation_df = maybe_limit_ranking_users(validation_df, args.max_ranking_users)
    ranking_testset = build_ranking_testset(
        train_df=train_df,
        validation_df=ranking_validation_df,
        n_negatives=args.ranking_negatives,
        seed=args.random_state,
    )
    ranking_rows: list[dict[str, Any]] = []
    candidate_mode = (
        "full_catalog"
        if args.ranking_negatives is None
        else f"negative_sample_{args.ranking_negatives}"
    )

    configs_to_rank = [best_item, best_funk]
    selected_by = "lowest_validation_rmse"
    if args.rank_all_configs:
        configs_to_rank = [
            (row["method"], trained_models[model_key(row)], row)
            for row in [*item_rows, *funk_rows]
            if row.get("status") == "ok"
        ]
        selected_by = "all_configs"

    for method_name, model, base_row in configs_to_rank:
        ranking_metrics = calculate_metrics(
            model=model,
            testset=ranking_testset,
            k=args.top_k,
            threshold=args.relevance_threshold,
        )
        ranking_set_rmse = ranking_metrics.pop("RMSE")
        ranking_rows.append(
            {
                "method": method_name,
                "selected_by": selected_by,
                "ranking_candidate_mode": candidate_mode,
                "ranking_users": ranking_validation_df["user_id"].nunique(),
                "RankingSetRMSE": ranking_set_rmse,
                **base_row,
                **ranking_metrics,
            }
        )

    ranking_results = pd.DataFrame(ranking_rows)
    ranking_results.to_csv(args.output_dir / "item_funksvd_ranking_metrics.csv", index=False)
    print("\nRanking metrics for best model of each family")
    print(ranking_results.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "tarea_01" / "dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "tarea_01" / "results")
    parser.add_argument("--item-k-values", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--item-min-k-values", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--item-sims",
        nargs="+",
        default=["pearson"],
        choices=["cosine", "msd", "pearson"],
    )
    parser.add_argument("--funk-factors", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--funk-epochs", type=int, default=30)
    parser.add_argument("--funk-lr-all-values", type=float, nargs="+", default=[0.005])
    parser.add_argument("--funk-reg-all-values", type=float, nargs="+", default=[0.02])
    parser.add_argument("--funk-biased-values", type=parse_bool, nargs="+", default=[True])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--ranking-negatives", type=parse_optional_int, default=500)
    parser.add_argument("--max-ranking-users", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rank-all-configs", action="store_true")
    parser.add_argument("--skip-ranking", action="store_true")
    return parser


if __name__ == "__main__":
    run_experiments(build_parser().parse_args())
