import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from surprise import AlgoBase, accuracy


def calculate_metrics(
    *,
    model: AlgoBase,
    testset: list[tuple],
    k: int = 10,
    threshold: float = 3.5,
) -> dict[str, float]:
    predictions = model.test(testset)
    rmse_val: float = float(accuracy.rmse(predictions, verbose=False))

    user_raw = [pred.uid for pred in predictions]
    item_raw = [pred.iid for pred in predictions]
    true_r = np.array([pred.r_ui for pred in predictions], dtype=np.float32)
    est_r = np.array([pred.est for pred in predictions], dtype=np.float32)

    unique_users, user_indices = np.unique(user_raw, return_inverse=True)
    unique_items, item_indices = np.unique(item_raw, return_inverse=True)

    n_users = len(unique_users)
    n_items = len(unique_items)

    relevant_mask = (true_r >= threshold).astype(np.int8)
    ground_truth_sparse = csr_matrix(
        (relevant_mask, (user_indices, item_indices)),
        shape=(n_users, n_items),
        dtype=np.int8,
    )

    # 4. Obtención del Top-K ordenado por usuario
    # Ordenar primero y luego tomar los top-k por usuario con cumcount (100% vectorizado)
    df_preds = pd.DataFrame(
        {
            "user_idx": user_indices,
            "item_idx": item_indices,
            "est_r": est_r,
        }
    )
    df_sorted = df_preds.sort_values(by=["user_idx", "est_r"], ascending=[True, False])
    df_sorted["rank"] = df_sorted.groupby("user_idx").cumcount()
    top_k_df = df_sorted[df_sorted["rank"] < k].copy().reset_index(drop=True)

    # 5. Evaluar si cada ítem en el Top-K es relevante
    gt_array = ground_truth_sparse.toarray()
    top_k_df["is_rel"] = gt_array[top_k_df["user_idx"].values, top_k_df["item_idx"].values]

    # 6. Cálculo de Precision@K, Recall@K, MAP@K y NDCG@K por usuario
    # Descuento logarítmico para rankings (posición 1-indexed: rank + 1)
    ranks = top_k_df["rank"].values + 1
    top_k_df["discount"] = 1.0 / np.log2(ranks + 1)
    top_k_df["dcg_contrib"] = top_k_df["is_rel"] * top_k_df["discount"]

    # Precisión acumulada en cada posición para Average Precision (AP)
    top_k_df["cum_rel"] = top_k_df.groupby("user_idx")["is_rel"].cumsum()
    top_k_df["prec_at_r"] = top_k_df["cum_rel"] / ranks
    top_k_df["ap_contrib"] = top_k_df["prec_at_r"] * top_k_df["is_rel"]

    # Agregación por usuario
    user_metrics = (
        top_k_df.groupby("user_idx")
        .agg(
            hits=("is_rel", "sum"),
            rec_k=("is_rel", "count"),
            dcg=("dcg_contrib", "sum"),
            ap_sum=("ap_contrib", "sum"),
        )
        .reindex(range(n_users), fill_value=0)
    )

    # Total de ítems relevantes en ground truth por usuario
    rel_per_user = np.asarray(ground_truth_sparse.sum(axis=1)).flatten()

    # IDCG@K (DCG ideal con los ítems relevantes disponibles)
    ideal_hits = np.minimum(rel_per_user, k)
    discounts = 1.0 / np.log2(np.arange(1, k + 1) + 1)
    cum_discounts = np.cumsum(discounts)
    idcg = np.zeros(n_users, dtype=np.float64)
    has_ideal = ideal_hits > 0
    idcg[has_ideal] = cum_discounts[ideal_hits[has_ideal] - 1]

    # Usuarios válidos para métricas (al menos 1 ítem relevante en el testset)
    valid_users = rel_per_user > 0

    precision_users = np.zeros(n_users)
    recall_users = np.zeros(n_users)
    map_users = np.zeros(n_users)
    ndcg_users = np.zeros(n_users)

    if np.any(valid_users):
        precision_users[valid_users] = (
            user_metrics.loc[valid_users, "hits"].values
            / user_metrics.loc[valid_users, "rec_k"].values
        )
        recall_users[valid_users] = (
            user_metrics.loc[valid_users, "hits"].values / rel_per_user[valid_users]
        )
        map_users[valid_users] = (
            user_metrics.loc[valid_users, "ap_sum"].values / ideal_hits[valid_users]
        )
        ndcg_users[valid_users] = np.where(
            idcg[valid_users] > 0,
            user_metrics.loc[valid_users, "dcg"].values / idcg[valid_users],
            0.0,
        )

    # 7. Novelty (Auto-información basada en la frecuencia de ítems en el catálogo de test)
    item_counts = pd.Series(item_indices).value_counts()
    item_probs = item_counts / len(item_indices)
    item_self_info = -np.log2(item_probs)

    top_k_novelty = top_k_df["item_idx"].map(item_self_info).mean() if len(top_k_df) > 0 else 0.0

    # 8. Diversity (Similitud de Coseno intra-lista promedio usando perfiles Sparse)
    # Matriz Ítem-Usuario dispersa normalizada para similitud de coseno
    item_user_sparse = csr_matrix(
        (
            np.ones(len(item_indices), dtype=np.float32),
            (item_indices, user_indices),
        ),
        shape=(n_items, n_users),
    )
    norms = np.asarray(np.sqrt(item_user_sparse.multiply(item_user_sparse).sum(axis=1))).flatten()
    norms[norms == 0] = 1.0
    item_user_norm = item_user_sparse.multiply(1.0 / norms[:, None]).tocsr()

    user_diversities = []
    for _, items in top_k_df.groupby("user_idx")["item_idx"]:
        item_list = items.values
        n_rec = len(item_list)
        if n_rec > 1:
            sub_mat = item_user_norm[item_list, :]
            # Similitud coseno entre los ítems recomendados al usuario
            sim_matrix = (sub_mat @ sub_mat.T).toarray()
            # Promedio de similitudes fuera de la diagonal
            upper_tri = sim_matrix[np.triu_indices(n_rec, k=1)]
            user_diversities.append(1.0 - float(np.mean(upper_tri)))
        else:
            user_diversities.append(1.0)

    diversity_val = float(np.mean(user_diversities)) if user_diversities else 1.0

    return {
        "RMSE": round(rmse_val, 4),
        f"Recall@{k}": round(
            float(np.mean(recall_users[valid_users])) if np.any(valid_users) else 0.0,
            4,
        ),
        f"Precision@{k}": round(
            float(np.mean(precision_users[valid_users])) if np.any(valid_users) else 0.0,
            4,
        ),
        f"NDCG@{k}": round(
            float(np.mean(ndcg_users[valid_users])) if np.any(valid_users) else 0.0,
            4,
        ),
        f"MAP@{k}": round(
            float(np.mean(map_users[valid_users])) if np.any(valid_users) else 0.0,
            4,
        ),
        "Diversity": round(diversity_val, 4),
        "Novelty": round(float(top_k_novelty), 4),
    }
