import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans


def calculate_entropy(embedding, num_bins=10):
    """
        args:
            embedding: ndarray. (expert_num, bs, hidden_dim)
        Return:
            ndarray. (hidden,)
    """
    embedding = embedding.reshape(-1, embedding.shape[-1])
    entropies = []
    for i in range(embedding.shape[-1]):
        hist_probabilities, _ = np.histogram(embedding[:, i], bins=num_bins, density=True)
        hist_probabilities = hist_probabilities[hist_probabilities > 0]
        hist_probabilities = hist_probabilities / np.sum(hist_probabilities)
        entropy = -np.sum(hist_probabilities * np.log2(hist_probabilities))
        entropies.append(entropy)
    return np.array(entropies)


def _weighted_euclidean_distance(embedding, entropies):
    """
        embedding: ndarray. (expert_num, bs, hidden)
        entropies: ndarray. (hidden,)
    """
    weights = 1 / (entropies + 1e-9)
    weights = np.clip(weights, 0, 1e2)
    weights /= weights.sum(axis=0, keepdims=True)

    assert embedding.shape[1] % len(weights) == 0
    mul = int(embedding.shape[1] / len(weights))
    weights = np.concatenate([weights for _ in range(mul)], axis=0)

    def _distance_func(u, v):
        euclidean = np.clip(u - v, -1e2, 1e2) ** 2
        weighed_euclidean = weights * euclidean
        return np.sqrt(np.sum(weighed_euclidean))

    dist_matrix = pdist(embedding, metric=_distance_func)
    return dist_matrix


def cluster_for_prune(embedding, cluster_number, pruning_method="hierarchical_prune", **kwargs):
    """
        args:
            embedding: ndarray. (expert_num, bs, hidden)
            entropies: ndarray. (hidden,)
    """
    expert_num, bs, hidden = embedding.shape
    embedding = embedding.reshape(expert_num, -1)
    if pruning_method == "kmeans_prune":
        uns = KMeans(n_clusters=cluster_number, random_state=0)
        basic_cluster_result = uns.fit(embedding)
        basic_cluster_label = basic_cluster_result.labels_.tolist()
    elif pruning_method.startswith("hierarchical_prune"):
        if pruning_method.endswith("with_entropy"):
            assert "entropy" in kwargs
            dist_matrix = _weighted_euclidean_distance(embedding, kwargs["entropy"])
        else:
            dist_matrix = squareform(pdist(embedding))
        Z = linkage(dist_matrix, method="ward")
        basic_cluster_label = fcluster(Z, t=cluster_number, criterion='maxclust').tolist()
    else:
        raise NotImplementedError("Unsupervised pruning method has not been implemented yet.")

    return basic_cluster_label

