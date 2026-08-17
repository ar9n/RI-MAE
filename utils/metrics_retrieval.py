import numpy as np
from sklearn.neighbors import NearestNeighbors


def evaluate(features, labels, batch_size=128):
    """
    Exact retrieval metrics with batched cosine similarity.

    Still O(N^2), but much faster than looping over queries one by one.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)

    n = features.shape[0]
    if n <= 1:
        return {"NN": 0.0, "FT": 0.0, "ST": 0.0, "mAP": 0.0}

    # Normalize once
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8

    # Encode labels as ints for faster comparisons
    _, labels_int = np.unique(labels, return_inverse=True)
    class_counts = np.bincount(labels_int)
    num_rel_all = class_counts[labels_int] - 1

    ranks = np.arange(1, n, dtype=np.float32)

    nn_sum = 0.0
    ft_sum = 0.0
    st_sum = 0.0
    ap_sum = 0.0
    valid_queries = 0

    for start in range(0, n, batch_size):

        end = min(start + batch_size, n)
        batch = features[start:end]

        # [B, N]
        sim = batch @ features.T

        # remove self-match
        row_idx = np.arange(end - start)
        sim[row_idx, start + row_idx] = -np.inf

        # full ranking (exact)
        ranked = np.argsort(-sim, axis=1)[:, : n - 1]

        rel = (labels_int[ranked] == labels_int[start:end, None])

        num_rel = num_rel_all[start:end]
        valid_mask = num_rel > 0
        if not np.any(valid_mask):
            continue

        rel = rel[valid_mask]
        num_rel = num_rel[valid_mask]

        cumsum_rel = np.cumsum(rel, axis=1, dtype=np.int32)
        precision_at_k = cumsum_rel / ranks[None, :]

        nn_sum += precision_at_k[:, 0].sum()

        ft_idx = num_rel - 1
        st_idx = np.minimum(2 * num_rel - 1, n - 2)  # avoid out-of-bounds

        rows = np.arange(rel.shape[0])
        ft_sum += (cumsum_rel[rows, ft_idx] / num_rel).sum()
        st_sum += (cumsum_rel[rows, st_idx] / num_rel).sum()
        ap_sum += ((precision_at_k * rel).sum(axis=1) / num_rel).sum()

        valid_queries += rel.shape[0]

    if valid_queries == 0:
        return {"NN": 0.0, "FT": 0.0, "ST": 0.0, "mAP": 0.0}

    return {
        "P@1": nn_sum / valid_queries,
        "FT": ft_sum / valid_queries,
        "ST": st_sum / valid_queries,
        "mAP": ap_sum / valid_queries,
    }