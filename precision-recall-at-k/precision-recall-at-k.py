import numpy as np

def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = np.asarray(recommended[:k])

    relevant = np.asarray(relevant)

    hits = np.intersect1d(top_k, relevant)

    precision_k = len(hits) / k

    recall_k = len(hits) / len(relevant) if len(relevant) > 0 else 0.0

    return [float(precision_k), float(recall_k)]