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
    if "kmeans" in pruning_method:
        uns = KMeans(n_clusters=cluster_number, random_state=0)
        basic_cluster_result = uns.fit(embedding)
        basic_cluster_label = basic_cluster_result.labels_.tolist()

        cluster_centers = uns.cluster_centers_
        distances_to_center = []
        for i in range(len(embedding)):
            distances_to_center.append(np.linalg.norm(embedding[i] - cluster_centers[basic_cluster_label[i]]))
        distances_to_center = np.array(distances_to_center, dtype=float)

    elif "hierarchical" in pruning_method:
        if pruning_method.startswith("entropy"):
            assert "entropy" in kwargs
            dist_matrix = _weighted_euclidean_distance(embedding, kwargs["entropy"])
        else:
            dist_matrix = squareform(pdist(embedding))
        Z = linkage(dist_matrix, method="ward")
        basic_cluster_label = fcluster(Z, t=cluster_number, criterion='maxclust').tolist()

        cluster_centers = []
        for cluster_id in range(1, cluster_number + 1):
            cluster_samples = embedding[np.array(basic_cluster_label) == cluster_id]
            cluster_center = np.mean(cluster_samples, axis=0)
            cluster_centers.append(cluster_center)
        cluster_centers = np.array(cluster_centers)

        distances_to_center = []
        for i, sample in enumerate(embedding):
            cluster_id = basic_cluster_label[i] - 1
            distance = np.linalg.norm(sample - cluster_centers[cluster_id])
            distances_to_center.append(distance)
        distances_to_center = np.array(distances_to_center, dtype=float)

    else:
        raise NotImplementedError("Unsupervised pruning method has not been implemented yet.")

    return (basic_cluster_label, distances_to_center)
