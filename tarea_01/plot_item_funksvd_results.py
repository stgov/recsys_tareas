"""Create report-ready plots from ItemKNN and FunkSVD experiment CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return None
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df.loc[df["status"] == "ok"].copy()
    return df


def save_metric_curve(
    df: pd.DataFrame,
    x_col: str,
    group_cols: list[str],
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    group_key = group_cols[0] if len(group_cols) == 1 else group_cols
    for group_values, group_df in df.groupby(group_key, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        label = ", ".join(
            f"{col}={value}" for col, value in zip(group_cols, group_values, strict=True)
        )
        group_df = group_df.sort_values(x_col)
        ax.plot(group_df[x_col], group_df[metric], marker="o", label=label)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def config_label(row: pd.Series) -> str:
    if row["method"] == "ItemKNN":
        return f"ItemKNN k={int(row['k_neighbors'])} {row['similarity']}"
    return (
        f"FunkSVD f={int(row['n_factors'])} "
        f"lr={row['lr_all']} reg={row['reg_all']} biased={row['biased']}"
    )


def save_ranking_quality_plot(df: pd.DataFrame, output_path: Path) -> None:
    metric_cols = [
        col
        for col in ["Precision@10", "Recall@10", "NDCG@10", "MAP@10"]
        if col in df.columns
    ]
    if not metric_cols:
        return

    plot_df = df.copy()
    plot_df["config"] = plot_df.apply(config_label, axis=1)
    ax = plot_df.set_index("config")[metric_cols].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Ranking quality on validation")
    ax.set_xlabel("configuration")
    ax.set_ylabel("score")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_discovery_plot(df: pd.DataFrame, output_path: Path) -> None:
    metric_cols = [col for col in ["Novelty", "CatalogCoverage"] if col in df.columns]
    if not metric_cols:
        return

    plot_df = df.copy()
    plot_df["config"] = plot_df.apply(config_label, axis=1)
    ax = plot_df.set_index("config")[metric_cols].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Novelty and catalog coverage on validation")
    ax.set_xlabel("configuration")
    ax.set_ylabel("score")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_diversity_plot(df: pd.DataFrame, output_path: Path) -> None:
    if "Diversity" not in df.columns:
        return

    plot_df = df.copy()
    plot_df["config"] = plot_df.apply(config_label, axis=1)
    ax = plot_df.set_index("config")["Diversity"].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Average distinct tags in Top-10 recommendations")
    ax.set_xlabel("configuration")
    ax.set_ylabel("distinct tags")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "tarea_01" / "results")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "tarea_01" / "figures")
    return parser


def main(args: argparse.Namespace) -> None:
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    item_df = read_csv_if_exists(args.results_dir / "item_based_sensitivity.csv")
    if item_df is not None and not item_df.empty:
        save_metric_curve(
            item_df,
            x_col="k_neighbors",
            group_cols=["similarity", "min_k"],
            metric="RMSE",
            title="ItemKNN sensitivity: RMSE vs k",
            output_path=args.figures_dir / "itemknn_rmse_by_k.png",
        )
        save_metric_curve(
            item_df,
            x_col="k_neighbors",
            group_cols=["similarity", "min_k"],
            metric="MAE",
            title="ItemKNN sensitivity: MAE vs k",
            output_path=args.figures_dir / "itemknn_mae_by_k.png",
        )

    funk_df = read_csv_if_exists(args.results_dir / "funksvd_sensitivity.csv")
    if funk_df is not None and not funk_df.empty:
        save_metric_curve(
            funk_df,
            x_col="n_factors",
            group_cols=["lr_all", "reg_all", "biased"],
            metric="RMSE",
            title="FunkSVD sensitivity: RMSE vs latent factors",
            output_path=args.figures_dir / "funksvd_rmse_by_factors.png",
        )
        save_metric_curve(
            funk_df,
            x_col="n_factors",
            group_cols=["lr_all", "reg_all", "biased"],
            metric="MAE",
            title="FunkSVD sensitivity: MAE vs latent factors",
            output_path=args.figures_dir / "funksvd_mae_by_factors.png",
        )

    ranking_df = read_csv_if_exists(args.results_dir / "item_funksvd_ranking_metrics.csv")
    if ranking_df is not None and not ranking_df.empty:
        save_ranking_quality_plot(ranking_df, args.figures_dir / "ranking_quality.png")
        save_discovery_plot(ranking_df, args.figures_dir / "ranking_discovery.png")
        save_diversity_plot(ranking_df, args.figures_dir / "ranking_diversity.png")

    print(f"Figures written to {args.figures_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
