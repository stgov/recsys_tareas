"""Reproducible UserKNN and consolidated experiments on the shared protocol.

Hyperparameters are selected on validation, so these are development results,
not an independent estimate of generalization. No competition files are written
unless ``write_competition_preview`` is explicitly called.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import surprise
from surprise import AlgoBase, PredictionImpossible, accuracy
from threadpoolctl import threadpool_limits

from tarea_01.evaluation import evaluate_model, paired_bootstrap, prepare_evaluation
from tarea_01.ranking_models import (
    ContentBased,
    ImplicitWrapper,
    bert_features,
    ingredient_tfidf,
)

TASK_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = TASK_DIR / "informe_assets" / "data"
FIGURE_DIR = TASK_DIR / "informe_assets"


def load_inputs(dataset_dir=None):
    dataset_dir = Path(dataset_dir or TASK_DIR / "dataset")
    train = pd.read_csv(dataset_dir / "train.csv")
    validation = pd.read_csv(dataset_dir / "validation.csv")
    metadata = pd.read_csv(dataset_dir / "item_metadata.csv")
    data = surprise.Dataset.load_from_df(
        train[["user_id", "recipe_id", "rating"]], surprise.Reader(rating_scale=(0, 5))
    )
    return train, validation, metadata, data.build_full_trainset()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


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


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(
            _json_safe(value),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _select_row(table):
    valid = table.loc[table["status"].eq("ok") & table["RMSE"].notna()].copy()
    if valid.empty:
        raise ValueError("No successful configurations with validation RMSE.")
    return valid.sort_values(["RMSE", "MAE", "config_id"], kind="stable").iloc[0]


def run_user_knn(
    dataset_dir=None,
    output_dir=None,
    *,
    k_values=(5, 10, 30, 50, 70, 100),
    similarities=("pearson", "msd"),
    seed=42,
    n_negatives=100,
):
    """Compare UserKNN similarities/neighbours, then evaluate the RMSE selection."""
    output_dir = Path(output_dir or ARTIFACT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, metadata, trainset = load_inputs(dataset_dir)
    testset = list(
        validation[["user_id", "recipe_id", "rating"]].itertuples(index=False, name=None)
    )
    rows = []
    for similarity in similarities:
        # KNNBasic's similarities do not depend on k: reuse the fitted matrix.
        start = perf_counter()
        model = surprise.KNNBasic(
            k=int(k_values[0]),
            min_k=1,
            sim_options={"name": similarity, "user_based": True},
            verbose=False,
        ).fit(trainset)
        fit_seconds = perf_counter() - start
        for k in k_values:
            model.k = int(k)
            predictions = model.test(testset)
            rows.append(
                {
                    "method": "UserKNN",
                    "config_id": f"userknn_{similarity}_k{k}",
                    "k_neighbors": int(k),
                    "min_k": 1,
                    "similarity": similarity,
                    "random_state": seed,
                    "status": "ok",
                    "RMSE": float(accuracy.rmse(predictions, verbose=False)),
                    "MAE": float(accuracy.mae(predictions, verbose=False)),
                    "similarity_fit_seconds": fit_seconds,
                }
            )
    sensitivity = pd.DataFrame(rows)
    selected = _select_row(sensitivity).to_dict()
    sensitivity["is_selected_rmse"] = sensitivity["config_id"].eq(selected["config_id"])
    sensitivity["is_selected_mae"] = sensitivity["config_id"].eq(
        sensitivity.sort_values(["MAE", "RMSE", "config_id"], kind="stable").iloc[0]["config_id"]
    )
    sensitivity.to_csv(output_dir / "user_based_sensitivity.csv", index=False)
    model = model_from_selection(selected).fit(trainset)
    context = prepare_evaluation(train, validation, metadata, n_negatives=n_negatives, seed=seed)
    result = evaluate_model(model, context=context, ranking_clip=False)
    result.export(output_dir, "userknn")
    ranking = pd.DataFrame(
        [
            {
                **selected,
                **result.metrics,
                "selected_by": "lowest_validation_rmse",
                **result.protocol,
            }
        ]
    )
    ranking.to_csv(output_dir / "userknn_ranking_metrics.csv", index=False)
    selection = {
        "method": "UserKNN",
        "selected_by": "lowest_validation_rmse",
        "selection_split": "validation.csv",
        "independent_test": False,
        "params": selected,
        "protocol": result.protocol,
    }
    _write_json(output_dir / "userknn_selection.json", selection)
    return {
        "sensitivity": sensitivity,
        "ranking": ranking,
        "selection": selection,
        "evaluation": result,
    }


def plot_user_sensitivity(sensitivity, output_dir=None):
    output_dir = Path(output_dir or FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"pearson": "#0072B2", "msd": "#D55E00"}
    for ax, metric in zip(axes, ("RMSE", "MAE"), strict=True):
        for similarity, group in sensitivity.groupby("similarity", sort=True):
            group = group.sort_values("k_neighbors")
            ax.plot(
                group["k_neighbors"],
                group[metric],
                marker="o",
                color=colors.get(similarity),
                label=similarity.upper(),
            )
        best = sensitivity.loc[sensitivity[metric].idxmin()]
        ax.scatter(
            [best["k_neighbors"]],
            [best[metric]],
            s=120,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
        )
        ax.annotate(
            f"Mínimo: {best[metric]:.4f}; k={int(best['k_neighbors'])}",
            xy=(best["k_neighbors"], best[metric]),
            xytext=(0.05, 0.10),
            textcoords="axes fraction",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": "0.4"},
        )
        ax.set(title=f"{metric} en validación (menor es mejor)", xlabel="Vecinos k", ylabel=metric)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("UserKNN: similitud y tamaño del vecindario")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"userknn_sensitivity.{suffix}", dpi=180, bbox_inches="tight")
    return fig


class BestRated(AlgoBase):
    """Unregularized item mean: an additional baseline, not a required method."""

    def fit(self, trainset):
        super().fit(trainset)
        self.item_means = {
            i: float(np.mean([r for _, r in ratings])) for i, ratings in trainset.ir.items()
        }
        return self

    def estimate(self, u, i):
        if not self.trainset.knows_item(i):
            raise PredictionImpossible("Unknown item.")
        return self.item_means[i]


class MostPopular(AlgoBase):
    """Rank by the number of training interactions, including all ratings."""

    def fit(self, trainset):
        super().fit(trainset)
        self.item_counts = {i: len(ratings) for i, ratings in trainset.ir.items()}
        return self

    def estimate(self, u, i):
        if not self.trainset.knows_item(i):
            raise PredictionImpossible("Unknown item.")
        return float(self.item_counts[i])


class UniformRandom(AlgoBase):
    """Seeded uniform score per raw (user, item), independent of call order."""

    def __init__(self, random_state=42):
        super().__init__()
        self.random_state = int(random_state)

    def estimate(self, u, i):
        raw_u = self.trainset.to_raw_uid(u) if self.trainset.knows_user(u) else str(u)
        raw_i = self.trainset.to_raw_iid(i) if self.trainset.knows_item(i) else str(i)
        key = f"{self.random_state}|{raw_u}|{raw_i}".encode()
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        return value / float(2**64)


def model_from_selection(row):
    method = row["method"]
    if method in {"UserKNN", "ItemKNN"}:
        return surprise.KNNBasic(
            k=int(row["k_neighbors"]),
            min_k=int(row.get("min_k", 1)),
            sim_options={"name": row["similarity"], "user_based": method == "UserKNN"},
            verbose=False,
        )
    if method == "FunkSVD":
        biased = str(row["biased"]).lower() in {"true", "1"}
        return surprise.SVD(
            n_factors=int(row["n_factors"]),
            n_epochs=int(row["n_epochs"]),
            lr_all=float(row["lr_all"]),
            reg_all=float(row["reg_all"]),
            biased=biased,
            random_state=int(row.get("random_state", 42)),
        )
    raise ValueError(f"Unsupported method {method!r}")


def load_selections(output_dir):
    selections = {}
    for method, filename in (
        ("UserKNN", "user_based_sensitivity.csv"),
        ("ItemKNN", "item_based_sensitivity.csv"),
        ("FunkSVD", "funksvd_sensitivity.csv"),
    ):
        path = Path(output_dir) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Execute user_based.ipynb, item_based.ipynb and "
                "funksvd.ipynb before general.ipynb."
            )
        table = pd.read_csv(path)
        if "is_selected_rmse" not in table:
            raise ValueError(f"Legacy results in {path}; rerun the corresponding notebook.")
        selected = table.loc[table["is_selected_rmse"].astype(str).str.lower().eq("true")]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one selected configuration in {path}.")
        row = selected.iloc[0].to_dict()
        if row["method"] != method:
            raise ValueError(f"Unexpected method in {path}.")
        selections[method] = row
    return selections


def _run_ranking_sensitivity(
    method,
    configurations,
    factory,
    trainset,
    context,
    output_dir,
    *,
    filename,
    export_prefix,
):
    """Evaluate a ranking family and retain its best validation nDCG model."""
    rows = []
    best_key = best_model = best_result = best_row = None
    for configuration in configurations:
        config = dict(configuration)
        print(f"Evaluating {config['config_id']}...", flush=True)
        model, diagnostics = factory(config)
        start = perf_counter()
        model.fit(trainset)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        result = evaluate_model(
            model,
            context=context,
            ranking_clip=False,
            rating_metrics=False,
        )
        evaluate_seconds = perf_counter() - start
        row = {
            "method": method,
            **config,
            **diagnostics,
            **result.metrics,
            **result.protocol,
            "fit_seconds": fit_seconds,
            "evaluate_seconds": evaluate_seconds,
            "rating_metrics_applicable": False,
            "selected_by": "highest_validation_ndcg_at_10",
        }
        rows.append(row)
        key = (-float(row["NDCG@10"]), -float(row["MAP@10"]), str(row["config_id"]))
        if best_key is None or key < best_key:
            best_key, best_model, best_result, best_row = key, model, result, row
    sensitivity = pd.DataFrame(rows)
    sensitivity["is_selected_ndcg"] = sensitivity["config_id"].eq(best_row["config_id"])
    sensitivity.to_csv(output_dir / filename, index=False)
    best_result.export(output_dir, export_prefix)
    parameter_keys = set(configurations[0]).union(diagnostics)
    selection = {
        "method": method,
        "selected_by": "highest_validation_ndcg_at_10_then_map_at_10",
        "selection_split": "validation.csv",
        "independent_test": False,
        "params": {key: best_row.get(key) for key in sorted(parameter_keys)},
        "protocol": best_result.protocol,
    }
    _write_json(output_dir / f"{export_prefix}_selection.json", selection)
    return {
        "sensitivity": sensitivity,
        "row": best_row,
        "model": best_model,
        "evaluation": best_result,
        "selection": selection,
    }


def run_general(
    dataset_dir=None,
    output_dir=None,
    *,
    seed=42,
    n_negatives=100,
    als_factors=(20, 50, 100),
    als_regularization=(0.01, 0.1),
    als_alpha=40.0,
    als_iterations=20,
    bpr_factors=(20, 50, 100),
    bpr_iterations=50,
    content_min_df=(1, 5, 10),
    content_weightings=("binary", "rating_excess"),
    include_bert=True,
):
    """Evaluate every required ranking method under one candidate context."""
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking

    output_dir = Path(output_dir or ARTIFACT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(dataset_dir or TASK_DIR / "dataset")
    selections = load_selections(output_dir)
    train, validation, metadata, trainset = load_inputs(dataset_dir)
    context = prepare_evaluation(train, validation, metadata, n_negatives=n_negatives, seed=seed)
    specs = {
        "BaselineOnly": (surprise.BaselineOnly(verbose=False), True),
        "BestRated": (BestRated(), True),
        "MostPopular": (MostPopular(), False),
        "Random": (UniformRandom(random_state=seed), False),
        **{name: (model_from_selection(row), True) for name, row in selections.items()},
    }
    rows, evaluations, fitted = [], {}, {}
    for name, (model, rating_metrics) in specs.items():
        print(f"Evaluating {name}...", flush=True)
        start = perf_counter()
        model.fit(trainset)
        fit_seconds = perf_counter() - start
        result = evaluate_model(
            model, context=context, ranking_clip=False, rating_metrics=rating_metrics
        )
        result.export(output_dir, f"general_{name.lower()}")
        rows.append(
            {
                "method": name,
                **result.metrics,
                **result.protocol,
                "fit_seconds": fit_seconds,
                "rating_metrics_applicable": rating_metrics,
                "selected_by": "lowest_validation_rmse" if name in selections else "fixed_baseline",
                "config_id": selections.get(name, {}).get("config_id", name.lower()),
            }
        )
        evaluations[name], fitted[name] = result, model

    als_configurations = [
        {
            "config_id": f"als_f{factors}_reg{regularization:g}",
            "n_factors": int(factors),
            "regularization": float(regularization),
            "alpha": float(als_alpha),
            "iterations": int(als_iterations),
            "random_state": int(seed),
            "relevance_threshold": 4.0,
        }
        for regularization in als_regularization
        for factors in als_factors
    ]

    def als_factory(config):
        with threadpool_limits(limits=1, user_api="blas"):
            native = AlternatingLeastSquares(
                factors=config["n_factors"],
                regularization=config["regularization"],
                alpha=config["alpha"],
                iterations=config["iterations"],
                random_state=config["random_state"],
                num_threads=1,
                use_gpu=False,
            )
        return ImplicitWrapper(native, config["relevance_threshold"]), {}

    als = _run_ranking_sensitivity(
        "ALS",
        als_configurations,
        als_factory,
        trainset,
        context,
        output_dir,
        filename="als_sensitivity.csv",
        export_prefix="general_als",
    )

    bpr_configurations = [
        {
            "config_id": f"bpr_f{factors}",
            "n_factors": int(factors),
            "learning_rate": 0.05,
            "regularization": 0.01,
            "iterations": int(bpr_iterations),
            "random_state": int(seed),
            "relevance_threshold": 4.0,
        }
        for factors in bpr_factors
    ]

    def bpr_factory(config):
        native = BayesianPersonalizedRanking(
            factors=config["n_factors"],
            learning_rate=config["learning_rate"],
            regularization=config["regularization"],
            iterations=config["iterations"],
            random_state=config["random_state"],
            num_threads=1,
            use_gpu=False,
        )
        return ImplicitWrapper(native, config["relevance_threshold"]), {}

    bpr = _run_ranking_sensitivity(
        "BPR",
        bpr_configurations,
        bpr_factory,
        trainset,
        context,
        output_dir,
        filename="bpr_sensitivity.csv",
        export_prefix="general_bpr",
    )

    content_configurations = [
        {
            "config_id": f"content_ingredients_df{min_df}_{weighting}",
            "content_source": "ingredient_tfidf",
            "content_min_df": int(min_df),
            "weighting": weighting,
            "relevance_threshold": 4.0,
        }
        for min_df in content_min_df
        for weighting in content_weightings
    ]
    if include_bert:
        content_configurations.extend(
            [
                {
                    "config_id": f"content_bert_{weighting}",
                    "content_source": "bert",
                    "content_min_df": None,
                    "weighting": weighting,
                    "relevance_threshold": 4.0,
                }
                for weighting in content_weightings
            ]
        )
    feature_cache = {}

    def content_factory(config):
        cache_key = (config["content_source"], config["content_min_df"])
        if cache_key not in feature_cache:
            if config["content_source"] == "bert":
                feature_cache[cache_key] = bert_features(metadata, dataset_dir)
            else:
                feature_cache[cache_key] = ingredient_tfidf(
                    metadata, min_df=config["content_min_df"]
                )
        item_ids, features, diagnostics = feature_cache[cache_key]
        model = ContentBased(
            item_ids,
            features,
            relevance_threshold=config["relevance_threshold"],
            weighting=config["weighting"],
        )
        return model, diagnostics

    content = _run_ranking_sensitivity(
        "ContentBased",
        content_configurations,
        content_factory,
        trainset,
        context,
        output_dir,
        filename="content_based_sensitivity.csv",
        export_prefix="general_contentbased",
    )

    for name, family in (("ALS", als), ("BPR", bpr), ("ContentBased", content)):
        rows.append(family["row"])
        evaluations[name] = family["evaluation"]
        fitted[name] = family["model"]
        selections[name] = family["selection"]
    metrics = pd.DataFrame(rows)
    if metrics.groupby("method").size().ne(1).any():
        raise AssertionError("The consolidated table must contain one row per method.")
    candidate_hashes = {result.protocol["candidate_sha256"] for result in evaluations.values()}
    if len(candidate_hashes) != 1:
        raise AssertionError("Methods were not evaluated on identical candidate sets.")
    als_popular_bootstrap = {
        "comparison": "ALS minus MostPopular",
        "scope": "fixed validation users and selected configurations",
        "limitation": "does not include seed or hyperparameter-selection uncertainty",
        "metrics": {
            metric: paired_bootstrap(
                evaluations["ALS"],
                evaluations["MostPopular"],
                metric=metric,
                n_resamples=5000,
                seed=seed,
            )
            for metric in ("NDCG@10", "Recall@10")
        },
    }
    _write_json(output_dir / "als_vs_mostpopular_bootstrap.json", als_popular_bootstrap)
    metrics.to_csv(output_dir / "all_methods_metrics.csv", index=False)
    selections.update(
        {
            "BaselineOnly": {
                "bsl_options": "Surprise default alternating bias estimation; not implicit ALS"
            },
            "BestRated": {"aggregation": "unregularized_training_item_mean"},
            "MostPopular": {"count": "all_training_interactions"},
            "Random": {"seed": seed, "algorithm": "blake2b_uniform_per_raw_user_item"},
        }
    )
    _write_json(output_dir / "all_methods_selection.json", selections)
    common_protocol = next(iter(evaluations.values())).protocol.copy()
    common_protocol.pop("scoring_api", None)
    common_protocol.pop("rating_score", None)
    _write_json(
        output_dir / "all_methods_protocol.json",
        {
            **common_protocol,
            "scoring_api": "scalar or per-user batch; identifiers only, no labels",
            "rating_score": "clipped for explicit predictors; omitted for ranking-only models",
            "selection_split": "validation.csv",
            "independent_test": False,
            "required_methods_complete": True,
            "pending_required_methods": [],
            "missing_sensitivity": [],
            "sensitivity_files": [
                "als_sensitivity.csv",
                "bpr_sensitivity.csv",
                "content_based_sensitivity.csv",
            ],
        },
    )
    return {
        "metrics": metrics,
        "models": fitted,
        "evaluations": evaluations,
        "selection": selections,
        "context": context,
        "sensitivity": {
            "ALS": als["sensitivity"],
            "BPR": bpr["sensitivity"],
            "ContentBased": content["sensitivity"],
        },
        "bootstrap": als_popular_bootstrap,
    }


def plot_general_metrics(metrics, output_dir=None):
    output_dir = Path(output_dir or FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("RMSE", "NDCG@10", "Recall@10"),
        ("Error de ratings (menor es mejor)", "nDCG@10 muestreado", "Recall@10 muestreado"),
        strict=True,
    ):
        frame = metrics.loc[metrics[metric].notna(), ["method", metric]].sort_values(metric)
        colors = ["#D55E00" if name == "MostPopular" else "#0072B2" for name in frame["method"]]
        ax.barh(frame["method"], frame[metric], color=colors, height=0.65)
        for y, value in enumerate(frame[metric]):
            ax.text(value, y, f" {value:.4f}", va="center", fontsize=8)
        ax.set(title=title, xlabel=metric, xlim=(0, float(frame[metric].max()) * 1.25))
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Comparación consolidada · misma validación y mismos candidatos")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"all_methods_quality.{suffix}", dpi=180, bbox_inches="tight")
    return fig


def plot_required_sensitivities(sensitivity, output_dir=None):
    """Plot factor searches for implicit CF and representation search for content."""
    output_dir = Path(output_dir or FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    factor_fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row_index, method in enumerate(("ALS", "BPR")):
        table = sensitivity[method]
        group_column = "regularization" if method == "ALS" else None
        groups = table.groupby(group_column, sort=True) if group_column else [(None, table)]
        for column_index, metric in enumerate(("NDCG@10", "MAP@10")):
            ax = axes[row_index, column_index]
            for color, (group_name, group) in zip(colors, groups, strict=False):
                group = group.sort_values("n_factors")
                label = f"regularización={group_name:g}" if group_column else method
                ax.plot(
                    group["n_factors"],
                    group[metric],
                    marker="o",
                    color=color,
                    label=label,
                )
            best = table.loc[table["is_selected_ndcg"]].iloc[0]
            ax.scatter(
                [best["n_factors"]],
                [best[metric]],
                s=130,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                zorder=4,
            )
            ax.set(
                title=f"{method}: {metric}",
                xlabel="Factores latentes",
                ylabel=metric,
            )
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
    factor_fig.suptitle("Sensibilidad de factores · feedback implícito")
    for suffix in ("png", "pdf"):
        factor_fig.savefig(
            output_dir / f"implicit_factor_sensitivity.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )

    content = sensitivity["ContentBased"].copy()

    def content_label(row):
        weighting = "binario" if row["weighting"] == "binary" else "ponderado 1/2"
        if row["content_source"] == "bert":
            return f"BERT (entregado) · {weighting}"
        return f"Ingredientes TF-IDF · min_df={int(row['content_min_df'])} · {weighting}"

    content["representation"] = content.apply(content_label, axis=1)
    content = content.sort_values("NDCG@10")
    content_fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ("NDCG@10", "MAP@10"), strict=True):
        bars = ax.barh(content["representation"], content[metric], color="#0072B2", height=0.62)
        selected = content["is_selected_ndcg"].to_numpy(dtype=bool)
        for bar, is_selected, value in zip(bars, selected, content[metric], strict=True):
            if is_selected:
                bar.set_color("#D55E00")
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.4f}", va="center")
        ax.set(title=metric, xlabel=metric)
        ax.set_xlim(0, float(content[metric].max()) * 1.25)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    content_fig.suptitle("Content-Based: comparación de representaciones")
    for suffix in ("png", "pdf"):
        content_fig.savefig(
            output_dir / f"content_based_sensitivity.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    return {"implicit": factor_fig, "content": content_fig}


def write_competition_preview(result, dataset_dir=None, output_dir=None):
    """Explicit opt-in train-only predictions; no refit or submission is performed."""
    dataset_dir = Path(dataset_dir or TASK_DIR / "dataset")
    output_dir = Path(output_dir or TASK_DIR / "results" / "competition_preview")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    rating_name = metrics.sort_values(["RMSE", "method"]).iloc[0]["method"]
    ranking_name = metrics.sort_values(["NDCG@10", "method"], ascending=[False, True]).iloc[0][
        "method"
    ]
    rating_model, ranking_model = result["models"][rating_name], result["models"][ranking_name]
    template = pd.read_csv(dataset_dir / "rating_template_fill.csv")
    template["rating"] = [
        rating_model.predict(row.user_id, row.recipe_id, clip=True).est
        for row in template.itertuples(index=False)
    ]
    template.to_csv(output_dir / "competicion_rating_completado.csv", index=False)
    targets = json.loads((dataset_dir / "ranking_template_fill.json").read_text(encoding="utf-8"))
    trainset = ranking_model.trainset
    catalog = sorted(trainset.to_raw_iid(i) for i in trainset.all_items())
    seen = {
        trainset.to_raw_uid(u): {trainset.to_raw_iid(i) for i, _ in ratings}
        for u, ratings in trainset.ur.items()
    }
    recommendations = {}
    for raw_uid in targets:
        uid = int(raw_uid) if str(raw_uid).isdigit() else raw_uid
        scored = [
            (iid, ranking_model.predict(uid, iid, clip=False).est)
            for iid in catalog
            if iid not in seen.get(uid, set())
        ]
        recommendations[str(raw_uid)] = [
            int(iid) for iid, _ in sorted(scored, key=lambda x: (-x[1], x[0]))[:10]
        ]
    _write_json(output_dir / "competicion_ranking.json", recommendations)
    _write_json(
        output_dir / "preview_protocol.json",
        {
            "rating_model": rating_name,
            "ranking_model": ranking_name,
            "selection_split": "validation.csv",
            "ranking_selection": "NDCG@10 sampled",
            "fit_split": "train.csv",
            "candidates": "full training catalog excluding seen",
            "tie_break": "raw item id ascending",
            "submitted": False,
        },
    )
    return output_dir
