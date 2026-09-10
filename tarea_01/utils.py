# ocupe AI para optimizar el codigo de forma vertorial
# https://share.gemini.google/4TfjE57QA4Ql

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from surprise import AlgoBase, accuracy


def calculate_metrics(
    *,
    model: AlgoBase,
    testset_rating: list[tuple],
    testset_ranking: list[tuple],
    k: int = 10,
    threshold: float = 3.5,
) -> dict[str, float]:
    predictions_rating = model.test(testset_rating)
    rmse_val = float(accuracy.rmse(predictions_rating, verbose=False))
    mae_val = float(accuracy.mae(predictions_rating, verbose=False))

    predictions_ranking = model.test(testset_ranking)
    records = [(p.uid, p.iid, p.est, p.r_ui) for p in predictions_ranking]
    df_preds = pd.DataFrame.from_records(records, columns=["uid", "iid", "est_r", "true_r"])

    df_preds["is_rel"] = (df_preds["true_r"] >= threshold).astype(np.int8)

    df_sorted = df_preds.sort_values(by=["uid", "est_r"], ascending=[True, False])
    df_sorted["rank"] = df_sorted.groupby("uid").cumcount() + 1
    top_k_df = df_sorted[df_sorted["rank"] <= k].copy()

    top_k_df["discount"] = 1.0 / np.log2(top_k_df["rank"] + 1)
    top_k_df["dcg_contrib"] = top_k_df["is_rel"] * top_k_df["discount"]

    top_k_df["cum_rel"] = top_k_df.groupby("uid")["is_rel"].cumsum()
    top_k_df["prec_at_r"] = top_k_df["cum_rel"] / top_k_df["rank"]
    top_k_df["ap_contrib"] = top_k_df["prec_at_r"] * top_k_df["is_rel"]

    user_metrics = top_k_df.groupby("uid").agg(
        hits=("is_rel", "sum"), dcg=("dcg_contrib", "sum"), ap_sum=("ap_contrib", "sum")
    )

    user_totals = df_preds.groupby("uid")["is_rel"].sum().rename("total_rel")
    user_metrics = user_metrics.join(user_totals, how="right").fillna(0)

    valid_users_mask = user_metrics["total_rel"] > 0
    valid_metrics = user_metrics[valid_users_mask].copy()

    if len(valid_metrics) > 0:
        valid_metrics["precision"] = valid_metrics["hits"] / k
        valid_metrics["recall"] = valid_metrics["hits"] / valid_metrics["total_rel"]

        ideal_hits = np.minimum(valid_metrics["total_rel"].astype(int), k)
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        cum_idcg_pad = np.insert(np.cumsum(discounts), 0, 0.0)
        valid_metrics["idcg"] = cum_idcg_pad[ideal_hits]

        valid_metrics["ndcg"] = np.where(
            valid_metrics["idcg"] > 0, valid_metrics["dcg"] / valid_metrics["idcg"], 0.0
        )
        valid_metrics["map"] = valid_metrics["ap_sum"] / ideal_hits

        mean_precision = valid_metrics["precision"].mean()
        mean_recall = valid_metrics["recall"].mean()
        mean_ndcg = valid_metrics["ndcg"].mean()
        mean_map = valid_metrics["map"].mean()
    else:
        mean_precision = mean_recall = mean_ndcg = mean_map = 0.0

    trainset = model.trainset
    top_k_df["inner_iid"] = top_k_df["iid"].map(trainset._raw2inner_id_items)

    item_counts = np.zeros(trainset.n_items, dtype=np.float32)
    for inner_iid, ur_list in trainset.ir.items():
        item_counts[inner_iid] = len(ur_list)

    valid_novelty_mask = top_k_df["inner_iid"].notna()
    inner_iids_array = top_k_df.loc[valid_novelty_mask, "inner_iid"].astype(int)

    top_k_df["train_count"] = 0.1
    top_k_df.loc[valid_novelty_mask, "train_count"] = item_counts[inner_iids_array]

    top_k_df["prob"] = top_k_df["train_count"] / trainset.n_ratings
    top_k_df["self_info"] = -np.log2(top_k_df["prob"])
    novelty_val = top_k_df["self_info"].mean() if len(top_k_df) > 0 else 0.0

    train_ratings = np.array(list(trainset.all_ratings()), dtype=np.int32)

    if len(train_ratings) > 0:
        uids_train = train_ratings[:, 0]
        iids_train = train_ratings[:, 1]

        train_sparse = csr_matrix(
            (np.ones(len(train_ratings), dtype=np.float32), (iids_train, uids_train)),
            shape=(trainset.n_items, trainset.n_users),
        )

        norms = np.array(
            train_sparse.sum(axis=1)
        ).flatten()  # En matriz 1/0, suma == suma de cuadrados
        norms_inv = np.divide(1.0, np.sqrt(norms), out=np.zeros_like(norms), where=norms > 0)
        train_norm = diags(norms_inv).dot(train_sparse)

        valid_recs = top_k_df[top_k_df["inner_iid"].notna()].copy()
        valid_recs["inner_iid"] = valid_recs["inner_iid"].astype(int)

        if len(valid_recs) > 0:
            eval_uids, _ = pd.factorize(valid_recs["uid"])
            eval_iids = valid_recs["inner_iid"].values
            n_eval_users = eval_uids.max() + 1

            R = csr_matrix(
                (np.ones(len(valid_recs), dtype=np.float32), (eval_uids, eval_iids)),
                shape=(n_eval_users, trainset.n_items),
            )

            Z = R.dot(train_norm)
            Z.data **= 2
            Z_sq_norms = np.array(Z.sum(axis=1)).flatten()
            k_u = np.array(R.sum(axis=1)).flatten()

            valid_k_mask = k_u > 1
            diversities = np.ones(n_eval_users, dtype=np.float32)
            k_valid = k_u[valid_k_mask]

            mean_sim = (Z_sq_norms[valid_k_mask] - k_valid) / (k_valid * (k_valid - 1))
            mean_sim = np.clip(mean_sim, 0.0, 1.0)  # Precisión de flotantes

            diversities[valid_k_mask] = 1.0 - mean_sim
            diversity_val = diversities.mean()
        else:
            diversity_val = 1.0
    else:
        diversity_val = 1.0

    return {
        "RMSE": round(rmse_val, 4),
        "MAE": round(mae_val, 4),
        f"Recall@{k}": round(float(mean_recall), 4),
        f"Precision@{k}": round(float(mean_precision), 4),
        f"NDCG@{k}": round(float(mean_ndcg), 4),
        f"MAP@{k}": round(float(mean_map), 4),
        "Diversity": round(float(diversity_val), 4),
        "Novelty": round(float(novelty_val), 4),
    }
