"""Figuras del informe: sensibilidad, error de ratings y ranking por separado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tarea_01.report_eda import COLORS, report_style, save_figure  # noqa: E402


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Sin resultados todavía: {path}")
        return None
    frame = pd.read_csv(path)
    if "status" in frame:
        frame = frame.loc[frame.status == "ok"].copy()
    return frame


def config_label(row: pd.Series) -> str:
    if (
        row["method"] == "ItemKNN"
        and pd.notna(row.get("similarity"))
        and pd.notna(row.get("k_neighbors"))
    ):
        return f"ItemKNN · {str(row['similarity']).upper()} · k={int(row['k_neighbors'])}"
    if row["method"] == "FunkSVD" and pd.notna(row.get("n_factors")):
        return f"FunkSVD · {int(row['n_factors'])} factores"
    return str(row["method"])


def _labels(frame: pd.DataFrame) -> pd.Series:
    labels = frame.apply(config_label, axis=1)
    if labels.duplicated().any() and "config_id" in frame:
        labels = labels + " (" + frame.config_id.astype(str) + ")"
    return labels


def _footer(frame: pd.DataFrame) -> str:
    parts = []
    if "ranking_candidate_mode" in frame:
        modes = frame.ranking_candidate_mode.dropna().unique()
        if len(modes) != 1:
            raise ValueError("No comparar protocolos de candidatos diferentes en una figura.")
        mode = str(modes[0])
        if mode.startswith("negative_sample_"):
            parts.append(f"Validación + hasta {mode.rsplit('_', 1)[-1]} negativos por usuario")
        else:
            parts.append("Candidatos: " + mode)
    if "ranking_users" in frame:
        users = frame.ranking_users.dropna().unique()
        if len(users) == 1:
            parts.append(f"{int(users[0]):,} usuarios")
    return " · ".join(parts)


def _save_close(fig: plt.Figure, output_path: Path) -> None:
    save_figure(fig, Path(output_path))
    plt.close(fig)


def save_sensitivity_plot(df: pd.DataFrame, method: str, output_path: Path) -> None:
    """Dos paneles: RMSE/MAE con iguales configuraciones y mínimo señalado."""
    report_style()
    plotted = df
    scope_note = ""
    if method == "ItemKNN":
        x_col, xlabel, group_cols = "k_neighbors", "Vecinos máximos (k)", ["similarity", "min_k"]
        selected_similarity = str(df.loc[df.RMSE.idxmin(), "similarity"])
        plotted = df.loc[df.similarity.eq(selected_similarity)].copy()
        scope_note = (
            f"Solo se muestra {selected_similarity.upper()}; "
            "Pearson se conserva en tablas y diagnósticos"
        )
    else:
        x_col, xlabel = "n_factors", "Factores latentes"
        group_cols = ["lr_all", "reg_all", "biased"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    for index, (values, group) in enumerate(plotted.groupby(group_cols, dropna=False, sort=True)):
        group = group.sort_values(x_col)
        if method == "ItemKNN":
            label = f"{str(values[0]).upper()} · min_k={int(values[1])}"
        else:
            label = f"lr={values[0]:g} · reg={values[1]:g}"
            if df.biased.nunique() > 1:
                label += f" · sesgos={'sí' if values[2] else 'no'}"
        for ax, metric in zip(axes, ["RMSE", "MAE"], strict=True):
            ax.plot(
                group[x_col],
                group[metric],
                marker="o",
                markersize=5,
                color=COLORS[index % len(COLORS)],
                linestyle=["-", "--", "-.", ":"][(index // len(COLORS)) % 4],
                label=label,
            )
            ax.set(xlabel=xlabel, ylabel=metric, title=f"{metric} en validación ↓")
            ax.set_xticks(sorted(plotted[x_col].unique()))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid()
    for ax, metric in zip(axes, ["RMSE", "MAE"], strict=True):
        best = plotted.loc[plotted[metric].idxmin()]
        ax.scatter(
            [best[x_col]], [best[metric]], s=120, facecolors="none", edgecolors="#222222", zorder=5
        )
        ax.set_title(f"{metric} en validación ↓ · mínimo {best[metric]:.4f}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 2), frameon=False)
    fig.suptitle(
        f"{method}: sensibilidad del error de ratings\n"
        "Menor valor es mejor · Ejes verticales ampliados; no comienzan en cero"
        + (f"\n{scope_note}" if scope_note else ""),
        fontsize=12,
    )
    _save_close(fig, output_path)


def plot_family_sensitivity(df: pd.DataFrame, method: str, output_dir: Path) -> Path:
    path = Path(output_dir) / (
        "itemknn_sensitivity.png" if method == "ItemKNN" else "funksvd_sensitivity.png"
    )
    save_sensitivity_plot(df, method, path)
    return path


def save_metric_curve(
    df: pd.DataFrame, x_col: str, group_cols: list[str], metric: str, title: str, output_path: Path
) -> None:
    """Compatibilidad con notebooks anteriores; las curvas nuevas usan dos paneles."""
    report_style()
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for values, group in df.groupby(group_cols, dropna=False):
        ordered = group.sort_values(x_col)
        ax.plot(ordered[x_col], ordered[metric], "o-", label=str(values))
    ax.set(title=title, xlabel=x_col, ylabel=metric)
    ax.grid()
    ax.legend(frameon=False, fontsize=8)
    _save_close(fig, output_path)


def _metric_panels(
    frame: pd.DataFrame,
    specs: list[tuple[str, str, bool]],
    title: str,
    output_path: Path,
    *,
    footer: str = "",
) -> None:
    specs = [spec for spec in specs if spec[0] in frame and frame[spec[0]].notna().any()]
    if not specs:
        return
    report_style()
    cols = min(len(specs), 2)
    rows = int(np.ceil(len(specs) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.2 * cols, max(3.3, len(frame) * 0.5) * rows),
        squeeze=False,
        layout="constrained",
    )
    labels = _labels(frame)
    y = np.arange(len(frame))
    palette = [COLORS[i % len(COLORS)] for i in range(len(frame))]
    for ax, (column, label, percentage) in zip(axes.flat, specs):
        values = frame[column]
        bars = ax.barh(y, values.fillna(0), color=palette, height=0.65)
        texts = [
            "sin resultado" if pd.isna(v) else (f"{v:.1%}" if percentage else f"{v:.4f}")
            for v in values
        ]
        ax.bar_label(bars, labels=texts, padding=4, fontsize=9)
        maximum = float(values.max())
        upper = 1.05 if percentage and maximum > 0.8 else max(maximum * 1.22, 0.01)
        ax.set(
            yticks=y,
            yticklabels=labels,
            xlabel=label,
            title=label,
            xlim=(0, upper),
        )
        ax.invert_yaxis()
        ax.grid(axis="x")
        if percentage:
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    for ax in list(axes.flat)[len(specs) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    if footer:
        fig.supxlabel(footer, fontsize=8)
    _save_close(fig, output_path)


def save_ranking_quality_plot(df: pd.DataFrame, output_path: Path) -> None:
    _metric_panels(
        df,
        [
            ("Precision@10", "Precision@10 ↑", True),
            ("Recall@10", "Recall@10 ↑", True),
            ("NDCG@10", "nDCG@10 ↑", False),
            ("MAP@10", "MAP@10 ↑", False),
        ],
        "Calidad del ranking en validación",
        output_path,
        footer=_footer(df),
    )


def plot_family_ranking(df: pd.DataFrame, method: str, output_dir: Path) -> Path:
    """Curvas compactas de nDCG/MAP; evita una leyenda por cada configuración."""
    report_style()
    x_col = "k_neighbors" if method == "ItemKNN" else "n_factors"
    groups = ["similarity"] if method == "ItemKNN" else ["lr_all", "reg_all"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    for index, (values, group) in enumerate(df.groupby(groups, sort=True)):
        values = values if isinstance(values, tuple) else (values,)
        label = (
            str(values[0]).upper()
            if method == "ItemKNN"
            else f"lr={values[0]:g} · reg={values[1]:g}"
        )
        group = group.sort_values(x_col)
        for ax, metric in zip(axes, ["NDCG@10", "MAP@10"], strict=True):
            ax.plot(
                group[x_col], group[metric], "o-", color=COLORS[index % len(COLORS)], label=label
            )
            ax.set(
                xlabel="Vecinos máximos (k)" if method == "ItemKNN" else "Factores latentes",
                ylabel=metric,
                title=f"{metric} ↑",
                xticks=sorted(df[x_col].unique()),
            )
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(2, len(labels)), frameon=False)
    footer = _footer(df)
    subtitle = "Ejes ampliados" + (f" · {footer}" if footer else "")
    fig.suptitle(f"{method}: sensibilidad de ranking\n{subtitle}", fontsize=11)
    path = Path(output_dir) / (
        "itemknn_ranking_sensitivity.png"
        if method == "ItemKNN"
        else "funksvd_ranking_sensitivity.png"
    )
    _save_close(fig, path)
    if method == "ItemKNN":
        save_item_diagnostics(df, Path(output_dir) / "itemknn_neighbors.png")
    return path


def save_item_diagnostics(df: pd.DataFrame, output_path: Path) -> None:
    """Diagnóstico de ranking: fallback y vecinos en predicciones exitosas."""
    if not {"ActualKMean", "RankingFallbackFraction"}.issubset(df.columns):
        return
    report_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    for index, (similarity, group) in enumerate(df.groupby("similarity", sort=True)):
        group = group.sort_values("k_neighbors")
        for ax, metric in zip(axes, ["ActualKMean", "RankingFallbackFraction"], strict=True):
            ax.plot(
                group.k_neighbors,
                group[metric],
                "o-",
                color=COLORS[index % len(COLORS)],
                label=str(similarity).upper(),
            )
            ax.set(xlabel="Vecinos máximos solicitados (k)", xticks=sorted(df.k_neighbors.unique()))
            ax.grid()
            ax.legend(frameon=False)
    axes[0].set(
        title="Vecinos efectivos disponibles",
        ylabel="Media de vecinos efectivos",
        ylim=(0, max(1, df.ActualKMean.max() * 1.2)),
    )
    axes[1].set(
        title="Predicciones que recurren al fallback",
        ylabel="Proporción de candidatos de ranking",
        ylim=(0, min(1, df.RankingFallbackFraction.max() * 1.2 + 0.01)),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("ItemKNN: disponibilidad de señal colaborativa", fontsize=13)
    footer = _footer(df)
    note = "Vecinos efectivos: predicciones que reportan actual_k"
    fig.supxlabel(note + (f" · {footer}" if footer else ""), fontsize=8)
    _save_close(fig, output_path)


def save_prediction_plot(df: pd.DataFrame, output_path: Path) -> None:
    _metric_panels(
        df,
        [("RMSE", "RMSE ↓", False), ("MAE", "MAE ↓", False)],
        "Error sobre ratings observados de validación",
        output_path,
        footer="Menor valor es mejor · No incluye ceros artificiales del ranking",
    )


def save_discovery_plot(df: pd.DataFrame, output_path: Path) -> None:
    _metric_panels(
        df,
        [
            ("NoveltyBitsPerItem", "Novedad: bits por ítem ↑", False),
            ("CatalogCoverage", "Cobertura del catálogo de train ↑", True),
        ],
        "Descubrimiento y alcance del catálogo",
        output_path,
        footer=_footer(df),
    )


def save_diversity_plot(df: pd.DataFrame, output_path: Path) -> None:
    _metric_panels(
        df,
        [
            ("TagDiversity", "Tags distintos por lista ↑", False),
            ("InteractionILD", "ILD de interacciones ↑", False),
        ],
        "Diversidad semántica y diversidad colaborativa",
        output_path,
        footer="",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "tarea_01" / "informe_assets" / "data"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=PROJECT_ROOT / "tarea_01" / "informe_assets"
    )
    return parser


def main(args: argparse.Namespace) -> None:
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    for filename, method, stem in (
        ("item_based_sensitivity.csv", "ItemKNN", "itemknn_sensitivity"),
        ("funksvd_sensitivity.csv", "FunkSVD", "funksvd_sensitivity"),
    ):
        frame = read_csv_if_exists(args.results_dir / filename)
        if frame is not None and not frame.empty:
            save_sensitivity_plot(frame, method, args.figures_dir / stem)
    sensitivity = read_csv_if_exists(args.results_dir / "item_funksvd_ranking_sensitivity.csv")
    if sensitivity is not None:
        for method, frame in sensitivity.groupby("method"):
            plot_family_ranking(frame, method, args.figures_dir)
    ranking_path = args.results_dir / "all_methods_metrics.csv"
    if not ranking_path.exists():
        ranking_path = args.results_dir / "item_funksvd_ranking_metrics.csv"
    ranking = read_csv_if_exists(ranking_path)
    if ranking is not None and not ranking.empty:
        save_ranking_quality_plot(ranking, args.figures_dir / "ranking_quality")
        save_discovery_plot(ranking, args.figures_dir / "ranking_discovery")
        save_diversity_plot(ranking, args.figures_dir / "ranking_diversity")
    prediction_path = args.results_dir / "all_methods_metrics.csv"
    if not prediction_path.exists():
        prediction_path = args.results_dir / "item_funksvd_prediction_metrics.csv"
    prediction = read_csv_if_exists(prediction_path)
    if prediction is not None and not prediction.empty:
        save_prediction_plot(prediction, args.figures_dir / "prediction_quality")
    print(f"Figuras PNG + PDF exportadas en {args.figures_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
