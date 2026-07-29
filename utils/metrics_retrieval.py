import numpy as np
from sklearn.neighbors import NearestNeighbors


def evaluate_nn_precision(features, labels):
    nbrs = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="auto").fit(features)
    _, indices = nbrs.kneighbors(features)

    correct = 0
    n = len(labels)

    for i in range(n):
        if labels[i] == labels[indices[i, 1]]:
            correct += 1

    return correct / n if n > 0 else 0.0

def evaluate(features, labels):
    """
    Compute retrieval mAP using cosine similarity with chunked matrix multiplication.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)

    n = features.shape[0]
    if n <= 1:
        return 0.0

    # L2-normalize for cosine similarity via dot product
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-8)

    metric_list = []
    ranks = np.arange(1, n, dtype=np.float32) 

    for idx in range(n):
    
        query = features[idx]

        # Cosine similarity
        sim = query @ features.T

        # exclude self
        sim[idx] = -np.inf 

        # descending order, excluding the query itself
        ranked = np.argsort(-sim)

        rel = (labels[ranked] == labels[idx]).astype(np.float32)[:(n-1)]
        cumsum_rel = np.cumsum(rel)
        num_rel = np.sum(rel)
        if num_rel <= 0:
            continue
        
        precision_at_k = cumsum_rel / ranks
        recall_at_k = cumsum_rel / num_rel

        nn = float(precision_at_k[0])
        ft = float(recall_at_k[int(num_rel)-1])
        st = float(recall_at_k[2*int(num_rel)-1])
        ap = float(np.sum(precision_at_k * rel) / num_rel)
        metric_list.append([nn, ft, st, ap])
        #if idx % 100 == 0:
            #print(f"Processed {idx}/{n} queries, current metrics: {np.mean(metric_list, axis=0)}")
            #print(f"Relevant samples for query {idx}: {num_rel}, Precision: {precision_at_k}, Similarity: {sim[ranked]}")

    average_metrics = np.mean(metric_list, axis=0)

    return {"NN": average_metrics[0], "FT": average_metrics[1], "ST": average_metrics[2], "mAP": average_metrics[3]}