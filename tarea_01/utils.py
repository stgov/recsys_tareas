"""utils.py

Módulo de utilidades para evaluación de modelos de recomendación con Surprise.
Calcula métricas de predicción (RMSE) y de ranking / catálogo (Recall@k,
NDCG@k, MAP@k, Diversity, Novelty).

Versión optimizada:
- Negative sampling en la generación de Top-N (evita puntuar todo el catálogo
  por usuario, que es el cuello de botella real con modelos KNN).
- Paralelización por usuario con joblib (ya es dependencia de surprise).
- Reemplazo de test_df.iterrows() por operaciones vectorizadas/groupby.
- Eliminado código muerto (user_ratings no se usaba).
"""

import math
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import pandas as pd
import surprise
from joblib import Parallel, delayed
from surprise import accuracy

# En este dataset los ids de usuario e ítem son siempre enteros
UserId = int
ItemId = int


def _predict_for_user(
    model: surprise.AlgoBase,
    user_id: UserId,
    candidates: npt.NDArray,
    n: int,
) -> tuple[UserId, list[ItemId]]:
    """Genera las Top-N predicciones para un único usuario. Aislado en su

    propia función para poder paralelizar con joblib.
    """
    preds = [model.predict(user_id, i, verbose=False) for i in candidates]
    preds.sort(key=lambda x: x.est, reverse=True)
    return user_id, [p.iid for p in preds[:n]]


def get_top_n_recommendations(
    model: surprise.AlgoBase,
    trainset: surprise.Trainset,
    test_df: pd.DataFrame,
    all_items: list[ItemId],
    n: int = 10,
    user_col: str = "user_id",
    item_col: str = "recipe_id",
    n_negatives: int | None = 200,
    n_jobs: int = -1,
    seed: int = 42,
) -> dict[UserId, list[ItemId]]:
    """Genera las Top-N recomendaciones predichas para cada usuario en test

    que no hayan sido vistas en el conjunto de entrenamiento.

    Parameters
    ----------
    n_negatives : int or None
        Si se especifica, en vez de puntuar TODO el catálogo no visto para
        cada usuario, se muestrea un subconjunto de `n_negatives` ítems
        (negative sampling). Es el enfoque estándar en evaluación de
        recomendadores y reduce drásticamente el costo computacional sin
        afectar mayormente el ranking relativo entre modelos. Poner None
        para el comportamiento original (evaluar todo el catálogo).
    n_jobs : int
        Núcleos a usar en la paralelización por usuario (-1 = todos).
    """
    rng = np.random.default_rng(seed)
    all_items_arr = np.array(all_items)

    user_seen: dict[UserId, set[ItemId]] = defaultdict(set)
    for user_id, item_id, _ in trainset.all_ratings():
        raw_u = trainset.to_raw_uid(user_id)
        raw_i = trainset.to_raw_iid(item_id)
        user_seen[raw_u].add(raw_i)

    test_users = test_df[user_col].unique()

    def build_candidates(u: UserId) -> npt.NDArray:
        seen = user_seen.get(u, set())
        unseen = all_items_arr[[item_id not in seen for item_id in all_items_arr]]
        if unseen.size == 0:
            unseen = all_items_arr
        if n_negatives is not None and unseen.size > n_negatives:
            unseen = rng.choice(unseen, size=n_negatives, replace=False)
        return unseen

    results: list[tuple[UserId, list[ItemId]]] = Parallel(n_jobs=n_jobs)(
        delayed(_predict_for_user)(model, u, build_candidates(u), n) for u in test_users
    )

    return dict(results)


def calculate_metrics(
    model: surprise.AlgoBase,
    trainset: surprise.Trainset,
    testset: list[tuple[UserId, ItemId, float]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
    user_col: str = "user_id",
    item_col: str = "recipe_id",
    rating_col: str = "rating",
    n_negatives: int | None = 200,
    n_jobs: int = -1,
) -> dict[str, float]:
    """Evalúa un modelo entrenado de Surprise retornando un diccionario con:

    - RMSE
    - Recall@k
    - NDCG@k
    - MAP@k
    - Diversity
    - Novelty
    """
    # 1. RMSE en testset
    predictions = model.test(testset)
    rmse_val = float(accuracy.rmse(predictions, verbose=False))

    # 2. Relevancias reales en Test (Ground Truth) - vectorizado con groupby
    relevant_df = test_df.loc[test_df[rating_col] >= threshold]
    user_relevant: dict[UserId, set[ItemId]] = (
        relevant_df.groupby(user_col)[item_col].apply(set).to_dict()
    )

    # 3. Probabilidades a priori para Novelty: P(i) = count(i) / total_interactions
    total_interactions: int = len(train_df)
    item_counts: dict[ItemId, int] = train_df[item_col].value_counts().to_dict()
    item_novelty_score: dict[ItemId, float] = {
        item: -math.log2(count / total_interactions)
        for item, count in item_counts.items()
        if count > 0
    }
    default_novelty: float = -math.log2(1 / (total_interactions + 1))

    # 4. Top-N Recomendaciones (con negative sampling + paralelización)
    all_items: list[ItemId] = list(
        set(train_df[item_col].unique()).union(set(test_df[item_col].unique()))
    )
    top_n = get_top_n_recommendations(
        model,
        trainset,
        test_df,
        all_items,
        n=k,
        user_col=user_col,
        item_col=item_col,
        n_negatives=n_negatives,
        n_jobs=n_jobs,
    )

    # 5. Cálculo de métricas de ranking por usuario
    recalls: list[float] = []
    ndcgs: list[float] = []
    maps: list[float] = []
    novelties: list[float] = []
    recommended_items_per_user: list[list[ItemId]] = []

    for u, recs in top_n.items():
        relevant = user_relevant.get(u, set())
        recommended_items_per_user.append(recs)

        # Recall@k
        if len(relevant) > 0:
            hits = len(set(recs).intersection(relevant))
            recalls.append(hits / len(relevant))
        else:
            recalls.append(0.0)

        # NDCG@k & MAP@k (relevancia binaria con threshold)
        dcg = 0.0
        running_hits = 0
        ap_sum = 0.0

        for rank, item in enumerate(recs, start=1):
            if item in relevant:
                dcg += 1.0 / math.log2(rank + 1)
                running_hits += 1
                ap_sum += running_hits / rank

        # IDCG
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcgs.append((dcg / idcg) if idcg > 0 else 0.0)
        maps.append((ap_sum / ideal_hits) if ideal_hits > 0 else 0.0)

        # Novelty por usuario
        user_nov = [item_novelty_score.get(item, default_novelty) for item in recs]
        if user_nov:
            novelties.append(np.mean(user_nov))

    # 6. Catalog Coverage / List Diversity (Cobertura Normalizada)
    all_recs_flat = [item for sublist in recommended_items_per_user for item in sublist]
    unique_recommended = len(set(all_recs_flat))
    diversity_val = unique_recommended / len(all_items) if all_items else 0.0

    # Normalizar Novelty para aproximación en escala 0-1
    max_possible_novelty = -math.log2(1 / (total_interactions + 1))
    avg_novelty_norm = float(np.mean(novelties) / max_possible_novelty) if novelties else 0.0

    return {
        "RMSE": round(rmse_val, 4),
        f"Recall@{k}": round(float(np.mean(recalls)), 4),
        f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
        f"MAP@{k}": round(float(np.mean(maps)), 4),
        "Diversity": round(diversity_val, 4),
        "Novelty": round(avg_novelty_norm, 4),
    }
