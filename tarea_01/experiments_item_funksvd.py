"""Experimentos únicos para los notebooks originales de ItemKNN y FunkSVD.

Se usa TODO train y TODO validation. La selección sobre validation es
exploratoria y no constituye una evaluación sobre un test independiente.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import surprise

from tarea_01.evaluation import evaluate_model, prepare_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "tarea_01" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "tarea_01" / "informe_assets" / "data"


def parse_optional_int(value):
    return None if value.lower() in {"none", "null", "full", "all"} else int(value)


def parse_bool(value):
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Booleano no válido: {value}")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_data(dataset_dir=DATASET_DIR):
    return pd.read_csv(dataset_dir / "train.csv"), pd.read_csv(dataset_dir / "validation.csv")


def build_surprise_sets(train_df, validation_df):
    data = surprise.Dataset.load_from_df(
        train_df[["user_id", "recipe_id", "rating"]], surprise.Reader(rating_scale=(0, 5))
    )
    return data.build_full_trainset(), list(
        validation_df[["user_id", "recipe_id", "rating"]].itertuples(index=False, name=None)
    )


def train_item_knn(trainset, k=80, min_k=1, sim_name="msd"):
    return surprise.KNNBasic(
        k=k, min_k=min_k, sim_options={"name": sim_name, "user_based": False}, verbose=False
    ).fit(trainset)


def train_funksvd(
    trainset, n_factors=20, n_epochs=30, lr_all=0.003, reg_all=0.05, biased=True, random_state=42
):
    return surprise.SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        biased=biased,
        random_state=random_state,
    ).fit(trainset)


def prediction_metrics(model, testset):
    predictions = model.test(testset)
    return {
        "RMSE": float(surprise.accuracy.rmse(predictions, verbose=False)),
        "MAE": float(surprise.accuracy.mae(predictions, verbose=False)),
        "RatingFallbackFraction": float(
            np.mean([p.details["was_impossible"] for p in predictions])
        ),
    }


def configuration_id(row):
    if row["method"] == "ItemKNN":
        return f"item_{row['similarity']}_k{row['k_neighbors']}_min{row['min_k']}"
    return (
        f"funk_f{row['n_factors']}_lr{row['lr_all']}_reg{row['reg_all']}"
        f"_bias{int(row['biased'])}_e{row['n_epochs']}_s{row['random_state']}"
    )


def artifact_manifest(dataset_dir, parameters):
    sources = [dataset_dir / n for n in ("train.csv", "validation.csv", "item_metadata.csv")]
    sources += [Path(__file__), Path(__file__).with_name("evaluation.py")]
    return {
        "parameters": parameters,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "scikit-surprise", "scipy", "matplotlib", "nbconvert")
        },
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "selection": "Exploratory minimum validation RMSE; ranking optimum shown separately.",
    }


def run_family(
    method,
    dataset_dir=None,
    output_dir=None,
    *,
    item_k_values=(10, 30, 50, 80),
    item_sims=("pearson", "msd"),
    item_min_k_values=(1,),
    funk_factors=(20, 50, 100),
    funk_lr_all_values=(0.003, 0.005),
    funk_reg_all_values=(0.02, 0.05),
    funk_biased_values=(True,),
    funk_epochs=30,
    random_state=42,
    ranking_negatives=100,
    top_k=10,
    relevance_threshold=4.0,
    skip_ranking=False,
    rank_all_configs=True,
):
    """Guarda grilla, ganadores, predicciones y procedencia; sin redondear selección.

    Se calcula cada similitud solo una vez, variando k durante la agregación.
    Evita conservar una matriz de similitud por configuración.
    """
    if method not in {"ItemKNN", "FunkSVD"}:
        raise ValueError("method debe ser ItemKNN o FunkSVD")
    dataset_dir, output_dir = Path(dataset_dir or DATASET_DIR), Path(output_dir or OUTPUT_DIR)
    train_df, validation_df = load_data(dataset_dir)
    metadata_df = pd.read_csv(dataset_dir / "item_metadata.csv")
    trainset, testset = build_surprise_sets(train_df, validation_df)
    context = (
        None
        if skip_ranking
        else prepare_evaluation(
            train_df,
            validation_df,
            metadata_df,
            k=top_k,
            threshold=relevance_threshold,
            n_negatives=ranking_negatives,
            seed=random_state,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if method == "ItemKNN":
        configurations = [
            dict(
                method=method,
                k_neighbors=k,
                min_k=minimum,
                similarity=sim,
                random_state=random_state,
            )
            for sim, minimum, k in product(item_sims, item_min_k_values, item_k_values)
        ]
    else:
        configurations = [
            dict(
                method=method,
                n_factors=f,
                n_epochs=funk_epochs,
                lr_all=lr,
                reg_all=reg,
                biased=bias,
                random_state=random_state,
            )
            for f, lr, reg, bias in product(
                funk_factors, funk_lr_all_values, funk_reg_all_values, funk_biased_values
            )
        ]
    if not configurations:
        raise ValueError("La grilla no puede estar vacía")
    rows, ranked = [], []
    best_model = best_row = best_result = best_ranking_result = best_ranking_row = None
    previous_similarity, model = None, None
    for config in configurations:
        start = perf_counter()
        config["config_id"] = configuration_id(config)
        if method == "ItemKNN":
            similarity_key = config["similarity"], config["min_k"]
            if similarity_key != previous_similarity:
                model = train_item_knn(
                    trainset,
                    k=config["k_neighbors"],
                    min_k=config["min_k"],
                    sim_name=config["similarity"],
                )
                previous_similarity = similarity_key
            model.k = config["k_neighbors"]
        else:
            model = train_funksvd(
                trainset,
                **{
                    key: config[key]
                    for key in (
                        "n_factors",
                        "n_epochs",
                        "lr_all",
                        "reg_all",
                        "biased",
                        "random_state",
                    )
                },
            )
        fit_seconds = perf_counter() - start
        result = None
        if rank_all_configs and not skip_ranking:
            result = evaluate_model(model, context=context)
            metrics = result.metrics
            ranked.append({**config, **metrics})
        else:
            metrics = prediction_metrics(model, testset)
        row = {
            **config,
            **metrics,
            "FitSeconds": fit_seconds,
            "EvaluationSeconds": perf_counter() - start - fit_seconds,
            "status": "ok",
        }
        rows.append(row)
        if best_row is None or row["RMSE"] < best_row["RMSE"]:
            best_row, best_result = row.copy(), result
            best_model = model if method == "FunkSVD" else None
        if result is not None and (
            best_ranking_row is None or row[f"NDCG@{top_k}"] > best_ranking_row[f"NDCG@{top_k}"]
        ):
            best_ranking_row, best_ranking_result = row.copy(), result
        print(f"{config['config_id']}: RMSE={row['RMSE']:.6f}, MAE={row['MAE']:.6f}", flush=True)

    if not skip_ranking and best_result is None:
        if method == "ItemKNN":
            best_model = train_item_knn(
                trainset, best_row["k_neighbors"], best_row["min_k"], best_row["similarity"]
            )
        best_result = evaluate_model(best_model, context=context)
        ranked.append({**best_row, **best_result.metrics})

    sensitivity = pd.DataFrame(rows)
    sensitivity["is_selected_rmse"] = sensitivity.config_id == best_row["config_id"]
    sensitivity["is_selected_mae"] = (
        sensitivity.config_id == sensitivity.loc[sensitivity.MAE.idxmin(), "config_id"]
    )
    if best_ranking_row is not None:
        sensitivity["is_selected_ndcg"] = sensitivity.config_id == best_ranking_row["config_id"]
    stem = "item_based" if method == "ItemKNN" else "funksvd"
    prefix = "itemknn" if method == "ItemKNN" else "funksvd"
    sensitivity.to_csv(output_dir / f"{stem}_sensitivity.csv", index=False)
    ranking = pd.DataFrame(ranked)
    if not skip_ranking:
        ranking["is_selected_rmse"] = ranking.config_id == best_row["config_id"]
        if best_ranking_row is not None:
            ranking["is_selected_ndcg"] = ranking.config_id == best_ranking_row["config_id"]
        ranking.to_csv(output_dir / f"{prefix}_ranking_sensitivity.csv", index=False)
        ranking.loc[ranking.is_selected_rmse].to_csv(
            output_dir / f"{prefix}_ranking_metrics.csv", index=False
        )
        best_result.export(output_dir, prefix=f"{prefix}_selected")
        if best_ranking_result is not None:
            best_ranking_result.export(output_dir, prefix=f"{prefix}_best_ndcg")
    parameters = {
        "method": method,
        "configurations": configurations,
        "ranking_negatives": ranking_negatives,
        "random_state": random_state,
        "threshold": relevance_threshold,
        "top_k": top_k,
        "skip_ranking": skip_ranking,
        "rank_all_configs": rank_all_configs,
    }
    manifest = artifact_manifest(dataset_dir, parameters)
    manifest["selected_rmse"], manifest["selected_ndcg"] = best_row, best_ranking_row
    if best_result is not None:
        manifest["protocol"] = best_result.protocol
    (output_dir / f"{prefix}_manifest.json").write_text(
        json.dumps(
            _json_safe(manifest),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )
    combine_results(output_dir)
    return {
        "sensitivity": sensitivity,
        "ranking": ranking,
        "selection": pd.DataFrame([best_row]),
        "evaluation": best_result,
    }


def combine_results(output_dir=OUTPUT_DIR):
    """Consolida artefactos de la nueva carpeta; no lee resultados históricos."""
    output_dir = Path(output_dir)
    mappings = [
        (
            ["item_based_sensitivity.csv", "funksvd_sensitivity.csv"],
            "item_funksvd_prediction_metrics.csv",
        ),
        (
            ["itemknn_ranking_metrics.csv", "funksvd_ranking_metrics.csv"],
            "item_funksvd_ranking_metrics.csv",
        ),
        (
            ["itemknn_ranking_sensitivity.csv", "funksvd_ranking_sensitivity.csv"],
            "item_funksvd_ranking_sensitivity.csv",
        ),
    ]
    for names, output in mappings:
        paths = [output_dir / name for name in names]
        if all(path.exists() for path in paths):
            pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(
                output_dir / output, index=False
            )


def run_experiments(args):
    kwargs = vars(args).copy()
    for family in kwargs.pop("families"):
        run_family(family, **kwargs)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families", nargs="+", choices=["ItemKNN", "FunkSVD"], default=["ItemKNN", "FunkSVD"]
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--item-k-values", type=int, nargs="+", default=[10, 30, 50, 80])
    parser.add_argument("--item-min-k-values", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--item-sims", nargs="+", choices=["pearson", "msd", "cosine"], default=["pearson", "msd"]
    )
    parser.add_argument("--funk-factors", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--funk-epochs", type=int, default=30)
    parser.add_argument("--funk-lr-all-values", type=float, nargs="+", default=[0.003, 0.005])
    parser.add_argument("--funk-reg-all-values", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--funk-biased-values", type=parse_bool, nargs="+", default=[True])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--ranking-negatives", type=parse_optional_int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rank-all-configs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-ranking", action="store_true")
    return parser


if __name__ == "__main__":
    run_experiments(build_parser().parse_args())
