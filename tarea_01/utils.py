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

    # 1. Predicción base
    predictions = model.test(testset)
    rmse_val = float(accuracy.rmse(predictions, verbose=False))

    # 2. Creación eficiente del DataFrame de resultados
    df_preds = pd.DataFrame(
        [
            {"uid": pred.uid, "iid": pred.iid, "est_r": float(pred.est), "true_r": float(pred.r_ui)}
            for pred in predictions
        ]
    )

    # Identificar ítems relevantes en el ground truth
    df_preds["is_rel"] = (df_preds["true_r"] >= threshold).astype(np.int8)

    # 3. Obtención del Top-K ordenado por usuario
    df_sorted = df_preds.sort_values(by=["uid", "est_r"], ascending=[True, False])
    df_sorted["rank"] = df_sorted.groupby("uid").cumcount() + 1
    top_k_df = df_sorted[df_sorted["rank"] <= k].copy()

    # 4. Cálculo de DCG y AP por ítem
    top_k_df["discount"] = 1.0 / np.log2(top_k_df["rank"] + 1)
    top_k_df["dcg_contrib"] = top_k_df["is_rel"] * top_k_df["discount"]

    top_k_df["cum_rel"] = top_k_df.groupby("uid")["is_rel"].cumsum()
    top_k_df["prec_at_r"] = top_k_df["cum_rel"] / top_k_df["rank"]
    top_k_df["ap_contrib"] = top_k_df["prec_at_r"] * top_k_df["is_rel"]

    # 5. Agregación de métricas a nivel de usuario
    user_metrics = top_k_df.groupby("uid").agg(
        hits=("is_rel", "sum"), dcg=("dcg_contrib", "sum"), ap_sum=("ap_contrib", "sum")
    )

    # Conteo real de ítems relevantes por usuario en todo el testset
    user_totals = df_preds.groupby("uid")["is_rel"].sum().rename("total_rel")
    user_metrics = user_metrics.join(user_totals, how="right").fillna(0)

    # Filtrar solo usuarios que tengan al menos 1 ítem relevante para evaluar
    valid_users_mask = user_metrics["total_rel"] > 0
    valid_metrics = user_metrics[valid_users_mask].copy()

    if len(valid_metrics) > 0:
        # Precision divide por K, Recall divide por los relevantes posibles
        valid_metrics["precision"] = valid_metrics["hits"] / k
        valid_metrics["recall"] = valid_metrics["hits"] / valid_metrics["total_rel"]

        # IDCG (vectorizado) para calcular NDCG
        ideal_hits = np.minimum(valid_metrics["total_rel"].astype(int), k)
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        cum_idcg_pad = np.insert(np.cumsum(discounts), 0, 0.0)  # offset index 0
        valid_metrics["idcg"] = cum_idcg_pad[ideal_hits]

        valid_metrics["ndcg"] = np.where(
            valid_metrics["idcg"] > 0, valid_metrics["dcg"] / valid_metrics["idcg"], 0.0
        )

        # MAP
        valid_metrics["map"] = valid_metrics["ap_sum"] / ideal_hits

        # Promedios finales
        mean_precision = valid_metrics["precision"].mean()
        mean_recall = valid_metrics["recall"].mean()
        mean_ndcg = valid_metrics["ndcg"].mean()
        mean_map = valid_metrics["map"].mean()
    else:
        mean_precision = mean_recall = mean_ndcg = mean_map = 0.0

    # 6. Novelty (basado en el modelo de entrenamiento, no en el testset)
    trainset = model.trainset
    n_train_ratings = trainset.n_ratings

    # Mapeo de popularidad cruda
    raw_to_count = {
        trainset.to_raw_iid(inner_iid): len(ur_list) for inner_iid, ur_list in trainset.ir.items()
    }

    # Asignar 0.1 artificial a ítems desconocidos para evitar log(0)
    top_k_df["train_count"] = top_k_df["iid"].map(raw_to_count).fillna(0.1)
    top_k_df["prob"] = top_k_df["train_count"] / n_train_ratings
    top_k_df["self_info"] = -np.log2(top_k_df["prob"])
    novelty_val = top_k_df["self_info"].mean() if len(top_k_df) > 0 else 0.0

    # 7. Diversity (Similitud del catálogo de Entrenamiento en Sparse)
    rows, cols = [], []
    for inner_iid, ur_list in trainset.ir.items():
        for inner_uid, _ in ur_list:
            rows.append(inner_iid)
            cols.append(inner_uid)

    train_sparse = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(trainset.n_items, trainset.n_users),
    )

    norms = np.asarray(np.sqrt(train_sparse.multiply(train_sparse).sum(axis=1))).flatten()
    norms[norms == 0] = 1.0
    train_norm = train_sparse.multiply(1.0 / norms[:, None]).tocsr()

    user_diversities = []
    for uid, group in top_k_df.groupby("uid"):
        # Convertir IDs crudos de test a IDs internos de train
        inner_iids = [
            trainset.to_inner_iid(raw)
            for raw in group["iid"]
            if trainset.knows_item(trainset.to_inner_iid(raw))
            if raw in raw_to_count
        ]

        n_rec = len(inner_iids)
        if n_rec > 1:
            sub_mat = train_norm[inner_iids, :]
            # El producto cruzado de KxK ítems es diminuto, toarray() aquí es seguro
            sim_matrix = (sub_mat @ sub_mat.T).toarray()
            upper_tri = sim_matrix[np.triu_indices(n_rec, k=1)]
            user_diversities.append(1.0 - float(np.mean(upper_tri)))
        else:
            user_diversities.append(1.0)

    diversity_val = np.mean(user_diversities) if user_diversities else 1.0

    return {
        "RMSE": round(rmse_val, 4),
        f"Recall@{k}": round(float(mean_recall), 4),
        f"Precision@{k}": round(float(mean_precision), 4),
        f"NDCG@{k}": round(float(mean_ndcg), 4),
        f"MAP@{k}": round(float(mean_map), 4),
        "Diversity": round(float(diversity_val), 4),
        "Novelty": round(float(novelty_val), 4),
    }
