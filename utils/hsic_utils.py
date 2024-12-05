from utils.HSIC import hsic_gam

import numpy as np
from sklearn.cluster import SpectralClustering


def split_graph(cka_similarity, num_subgraphs):
    clustering = SpectralClustering(
        n_clusters=num_subgraphs,
        affinity='precomputed',
        random_state=42
    )
    labels = clustering.fit_predict(cka_similarity)  # 注意: 距离矩阵需要取负值作为亲和度
    subgraphs = []
    for i in range(num_subgraphs):
        subgraphs.append(np.where(labels == i)[0].tolist())
    return subgraphs

def hsic_split_graph(expert_data, prune_rate=0.2):
    cka_similarity = np.zeros((expert_number, expert_number))
    for i in range(expert_number - 1):
        for j in range(i + 1, expert_number):
            i_j_sim, _ = hsic_gam(expert_data[i], expert_data[j])
            i_i_sim, _ = hsic_gam(expert_data[i], expert_data[i])
            j_j_sim, _ = hsic_gam(expert_data[j], expert_data[j])
            val = i_j_sim / (i_i_sim * j_j_sim)
            cka_similarity[i, j] = val
            cka_similarity[j, i] = val

    num_subgraphs = int((1 - prune_rate) * expert_number)
    subgraphs = split_graph(cka_similarity, num_subgraphs)

    for i, subgraph in enumerate(subgraphs):
        print(f"subgraph {i + 1}: {subgraph}")
    return subgraphs


if __name__ == '__main__':
    expert_number = 16
    batch_size = 32
    hidden_size = 64
    expert_data = np.random.rand(*(expert_number, batch_size, hidden_size))
    subgraphs = hsic_split_graph(expert_data)
