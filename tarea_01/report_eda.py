"""EDA reproducible: tablas auditables y figuras en español para el informe.

Ejecutar desde el repositorio: ``python -m tarea_01.report_eda``.
No altera los datos ni utiliza validación para ajustar una representación.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

TASK_ROOT = Path(__file__).resolve().parent
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def report_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.alpha": 0.2,
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Exporta PNG para Markdown y PDF vectorial con el mismo nombre base."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".png", ".pdf"):
        fig.savefig(path.with_suffix(extension), bbox_inches="tight", facecolor="white")


def parse_list_column(value: object) -> list:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Se esperaba una lista, se encontró {type(parsed).__name__}")
    return parsed


def ingredient_coverage(
    ingredients: pd.Series,
    thresholds: tuple[int, ...] = (1, 5, 10),
    min_ingredients: int = 3,
) -> pd.DataFrame:
    """Frecuencia documental: cada ingrediente cuenta una vez por receta."""
    documents = ingredients.map(lambda values: set(parse_list_column(values)))
    frequencies = Counter(term for document in documents for term in document)
    rows = []
    for threshold in thresholds:
        vocabulary = {term for term, count in frequencies.items() if count >= threshold}
        counts = documents.map(lambda values: len(values & vocabulary))
        rows.append(
            {
                "min_df": threshold,
                "vocabulary": len(vocabulary),
                "recipes": len(counts),
                "min_ingredients": min_ingredients,
                "recipes_with_min_ingredients": int((counts >= min_ingredients).sum()),
                "coverage_fraction": float((counts >= min_ingredients).mean()),
                "orphan_recipes": int((counts == 0).sum()),
                "orphan_fraction": float((counts == 0).mean()),
                "mean_ingredients": float(counts.mean()),
                "p10_ingredients": float(counts.quantile(0.1)),
                "p25_ingredients": float(counts.quantile(0.25)),
                "median_ingredients": float(counts.median()),
                "p75_ingredients": float(counts.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def top_recipes(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    top_k: int = 5,
    *,
    by: str = "rating",
    min_ratings: int = 1,
) -> pd.DataFrame:
    """Orden determinista: media, soporte e ID; nunca cortar antes de ordenar."""
    table = (
        train.groupby("recipe_id")
        .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "size"))
        .reset_index()
    )
    table = table.loc[table.n_ratings >= min_ratings]
    columns = ["mean_rating", "n_ratings", "recipe_id"]
    if by == "popularity":
        columns = ["n_ratings", "mean_rating", "recipe_id"]
    elif by != "rating":
        raise ValueError("by debe ser 'rating' o 'popularity'")
    table = table.sort_values(columns, ascending=[False, False, True]).head(top_k)
    return table.merge(
        metadata[["id", "name"]],
        left_on="recipe_id",
        right_on="id",
        how="left",
        validate="one_to_one",
    ).drop(columns="id")


def split_summary(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    train_universe = train.user_id.nunique() * train.recipe_id.nunique()
    rows = []
    for name, frame in (("train", train), ("validation", validation)):
        pairs = frame[["user_id", "recipe_id"]].drop_duplicates()
        own_universe = frame.user_id.nunique() * frame.recipe_id.nunique()
        dates = pd.to_datetime(frame.date, errors="coerce")
        rows.append(
            {
                "split": name,
                "interactions": len(frame),
                "unique_pairs": len(pairs),
                "users": frame.user_id.nunique(),
                "recipes": frame.recipe_id.nunique(),
                "density_own_universe": len(pairs) / own_universe,
                "density_train_universe": len(pairs) / train_universe,
                "mean_rating": frame.rating.mean(),
                "rating_4_5_fraction": frame.rating.isin([4, 5]).mean(),
                "rating_zero_fraction": frame.rating.eq(0).mean(),
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_rows": int(frame.duplicated().sum()),
                "duplicate_pairs": int(frame.duplicated(["user_id", "recipe_id"]).sum()),
                "invalid_dates": int(dates.isna().sum()),
                "invalid_ratings": int((~frame.rating.between(0, 5)).sum()),
                "date_min": str(dates.min().date()),
                "date_max": str(dates.max().date()),
            }
        )
    return pd.DataFrame(rows)


def plot_ratings_time(train: pd.DataFrame, validation: pd.DataFrame) -> plt.Figure:
    report_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained")
    for offset, (name, frame) in enumerate((("Train", train), ("Validación", validation))):
        fractions = frame.rating.value_counts(normalize=True).reindex(range(6), fill_value=0)
        bars = axes[0].bar(
            np.arange(6) + (offset - 0.5) * 0.36,
            fractions,
            width=0.36,
            color=COLORS[offset],
            label=name,
        )
        axes[0].bar_label(
            bars,
            labels=[f"{100 * v:.1f}%" for v in fractions],
            fontsize=7,
            padding=2 if offset == 0 else 10,
        )
        counts = pd.to_datetime(frame.date).dt.year.value_counts().sort_index()
        axes[1].plot(
            counts.index, counts.values, "o-", markersize=3, color=COLORS[offset], label=name
        )
    axes[0].set(
        title="Las evaluaciones se concentran en 4 y 5",
        xlabel="Rating",
        ylabel="Proporción de interacciones",
        xticks=range(6),
        ylim=(0, 1),
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(
        title="El split no es un corte temporal global",
        xlabel="Año",
        ylabel="Interacciones observadas",
    )
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    for ax in axes:
        ax.grid(axis="y")
        ax.legend(frameon=False)
    return fig


def plot_activity(counts: pd.Series, entity: str = "usuarios") -> plt.Figure:
    """Intervalos exactos (0,5], (5,10], ... (50,inf): 1-5,...,51+."""
    report_style()
    bins = [*range(0, 51, 5), np.inf]
    labels = [f"{low + 1}-{low + 5}" for low in range(0, 50, 5)] + ["51+"]
    distribution = pd.cut(counts, bins=bins, labels=labels, right=True).value_counts(sort=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), layout="constrained")
    x = np.arange(len(labels))
    bars = axes[0].bar(x, distribution.values, color=COLORS[0])
    axes[0].bar_label(bars, padding=3, fontsize=8)
    axes[0].set(
        ylabel=f"Número de {entity}",
        title=f"Actividad de {entity} en train",
        ylim=(0, distribution.max() * 1.22),
    )
    axes[1].plot(x, distribution.cumsum() / distribution.sum(), "o-", color=COLORS[2])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(
        ylabel=f"Fracción acumulada de {entity}",
        ylim=(0, 1.06),
        title=f"Media: {counts.mean():.1f}; mediana: {counts.median():.0f}",
    )
    for ax in axes:
        ax.set(xticks=x, xticklabels=labels, xlabel="Interacciones por entidad")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y")
    return fig


def plot_long_tail(train: pd.DataFrame) -> plt.Figure:
    report_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained")
    for index, (column, label) in enumerate((("user_id", "Usuarios"), ("recipe_id", "Recetas"))):
        values = train[column].value_counts().sort_values(ascending=False).to_numpy()
        rank = np.arange(1, len(values) + 1)
        axes[0].plot(rank, values, color=COLORS[index], label=label)
        axes[1].plot(
            np.r_[0, rank / len(values)],
            np.r_[0, values.cumsum() / values.sum()],
            color=COLORS[index],
            label=label,
        )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Posición por actividad (log)",
        ylabel="Interacciones (log)",
        title="Cola larga de actividad y popularidad",
    )
    axes[1].plot([0, 1], [0, 1], "--", color="#777777", linewidth=1, label="Uniforme")
    axes[1].set(
        xlabel="Fracción de entidades más activas",
        ylabel="Fracción de interacciones",
        title="Concentración de la señal de entrenamiento",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid()
    return fig


def plot_metadata(metadata: pd.DataFrame, coverage: pd.DataFrame) -> plt.Figure:
    report_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    minutes = pd.to_numeric(metadata.minutes, errors="coerce")
    regular = minutes.between(1, 240)
    axes[0].hist(minutes[regular], bins=np.arange(0, 256, 15), color=COLORS[0], edgecolor="white")
    axes[0].set(
        title="Tiempo de preparación: zona habitual", xlabel="Minutos (1 a 240)", ylabel="Recetas"
    )
    axes[0].text(
        0.97,
        0.96,
        f"Ventana: {regular.mean():.1%} de recetas\n"
        f"Ceros: {minutes.eq(0).sum()} · >240: {minutes.gt(240).sum()}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    counts = metadata.ingredients.map(parse_list_column).map(len)
    axes[1].hist(
        counts, bins=np.arange(-0.5, counts.max() + 1.5), color=COLORS[2], edgecolor="white"
    )
    axes[1].set(
        title="Complejidad de las recetas", xlabel="Ingredientes declarados", ylabel="Recetas"
    )
    x = np.arange(len(coverage))
    bars = axes[2].bar(x, coverage.coverage_fraction, color=COLORS[0], label="≥3 ingredientes")
    axes[2].bar_label(
        bars, labels=[f"{v:.2%}" for v in coverage.coverage_fraction], padding=3, fontsize=9
    )
    axes[2].set(
        xticks=x,
        xticklabels=[str(v) for v in coverage.min_df],
        ylim=(0, 1.12),
        title="Cobertura al filtrar ingredientes raros",
        xlabel="Frecuencia mínima por receta (min_df)",
        ylabel="Recetas con al menos 3 ingredientes",
    )
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(axis="y")
    return fig


def build_eda_artifacts(
    dataset_dir: Path = TASK_ROOT / "dataset", output_dir: Path = TASK_ROOT / "informe_assets"
) -> dict:
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(dataset_dir / "train.csv")
    validation = pd.read_csv(dataset_dir / "validation.csv")
    metadata = pd.read_csv(dataset_dir / "item_metadata.csv")
    splits = split_summary(train, validation)
    coverage = ingredient_coverage(metadata.ingredients)
    user_counts, item_counts = train.user_id.value_counts(), train.recipe_id.value_counts()
    train_dates = pd.to_datetime(train.date).groupby(train.user_id).max()
    validation_dates = pd.to_datetime(validation.date).groupby(validation.user_id).min()
    temporal = pd.concat(
        [train_dates.rename("last_train"), validation_dates.rename("first_validation")], axis=1
    ).dropna()
    minutes = pd.to_numeric(metadata.minutes, errors="coerce")
    ingredient_frequencies = Counter(
        term for terms in metadata.ingredients.map(parse_list_column) for term in set(terms)
    )
    tag_frequencies = Counter(
        term for terms in metadata.tags.map(parse_list_column) for term in set(terms)
    )
    summary = {
        "splits": splits.to_dict(orient="records"),
        "cross_split": {
            "shared_pairs": len(
                train[["user_id", "recipe_id"]]
                .drop_duplicates()
                .merge(
                    validation[["user_id", "recipe_id"]].drop_duplicates(),
                    on=["user_id", "recipe_id"],
                )
            ),
            "new_validation_users": len(set(validation.user_id) - set(train.user_id)),
            "new_validation_recipes": len(set(validation.recipe_id) - set(train.recipe_id)),
            "users_with_both_splits": len(temporal),
            "users_validation_before_last_train": int(
                (temporal.first_validation < temporal.last_train).sum()
            ),
            "users_boundary_same_day": int(
                (temporal.first_validation == temporal.last_train).sum()
            ),
            "global_date_ranges_overlap": bool(
                pd.to_datetime(validation.date).min() < pd.to_datetime(train.date).max()
            ),
        },
        "activity": {},
        "metadata": {
            "recipes": len(metadata),
            "duplicate_ids": int(metadata.id.duplicated().sum()),
            "missing_by_column": {str(k): int(v) for k, v in metadata.isna().sum().items()},
            "train_items_without_metadata": len(set(train.recipe_id) - set(metadata.id)),
            "validation_items_without_metadata": len(set(validation.recipe_id) - set(metadata.id)),
            "minutes_zero": int(minutes.eq(0).sum()),
            "minutes_negative": int(minutes.lt(0).sum()),
            "minutes_gt_240": int(minutes.gt(240).sum()),
            "minutes_gt_1440": int(minutes.gt(1440).sum()),
            "minutes_max": float(minutes.max()),
            "minutes_median": float(minutes.median()),
            "minutes_p99": float(minutes.quantile(0.99)),
            "ingredients_vocabulary": len(ingredient_frequencies),
            "ingredients_singleton": sum(value == 1 for value in ingredient_frequencies.values()),
            "ingredients_frequency_lt5": sum(
                value < 5 for value in ingredient_frequencies.values()
            ),
            "tags_vocabulary": len(tag_frequencies),
        },
        "ingredient_coverage": coverage.to_dict(orient="records"),
        "sources": {
            name: {"sha256": hashlib.sha256((dataset_dir / name).read_bytes()).hexdigest()}
            for name in ("train.csv", "validation.csv", "item_metadata.csv")
        },
    }
    for label, counts in (("users", user_counts), ("recipes", item_counts)):
        summary["activity"][label] = {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "at_most_5_fraction": float(counts.le(5).mean()),
            "at_most_10_fraction": float(counts.le(10).mean()),
            "top_1pct_interaction_fraction": float(
                counts.head(int(np.ceil(len(counts) * 0.01))).sum() / counts.sum()
            ),
            "top_10pct_interaction_fraction": float(
                counts.head(int(np.ceil(len(counts) * 0.1))).sum() / counts.sum()
            ),
        }
    (data_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    splits.to_csv(data_dir / "eda_split_summary.csv", index=False)
    coverage.to_csv(data_dir / "ingredient_coverage.csv", index=False)
    top_recipes(train, metadata, by="popularity").to_csv(
        data_dir / "top_popular_recipes.csv", index=False
    )
    top_recipes(train, metadata, min_ratings=10).to_csv(
        data_dir / "top_rated_recipes_min10.csv", index=False
    )
    rating_distribution = (
        pd.DataFrame(
            {
                "train": train.rating.value_counts(normalize=True),
                "validation": validation.rating.value_counts(normalize=True),
            }
        )
        .sort_index()
        .rename_axis("rating")
    )
    rating_distribution.to_csv(data_dir / "rating_distribution.csv")
    metadata.loc[minutes.eq(0) | minutes.gt(1440), ["id", "name", "minutes"]].to_csv(
        data_dir / "metadata_minutes_review.csv", index=False
    )
    for stem, fig in (
        ("eda_ratings_time", plot_ratings_time(train, validation)),
        ("eda_user_activity", plot_activity(user_counts, "usuarios")),
        ("eda_item_activity", plot_activity(item_counts, "recetas")),
        ("eda_long_tail", plot_long_tail(train)),
        ("eda_metadata", plot_metadata(metadata, coverage)),
    ):
        save_figure(fig, output_dir / stem)
        plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=TASK_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=TASK_ROOT / "informe_assets")
    args = parser.parse_args()
    summary = build_eda_artifacts(args.dataset_dir, args.output_dir)
    print(pd.DataFrame(summary["splits"]).to_string(index=False))
    print(json.dumps(summary["cross_split"], indent=2, ensure_ascii=False))
    print(f"EDA exportado en {args.output_dir}")


if __name__ == "__main__":
    main()
