import torch
import os
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def compute_similarity(embedding):
    distance_matrix = squareform(pdist(embedding.to(torch.float16).detach().numpy()))
    similarity_matrix = 1 - distance_matrix / np.max(distance_matrix)
    return similarity_matrix


if __name__ == "__main__":
    model = "deepseek"
    math_expert_output_dir = "pruned_result/DeepSeek-V2-Lite/sample_1000/MathInstruct_expert_output_hidden"
    save_dir = "visual/deepseek_MathInstruct_expert_similarity"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    layer_output = []
    for layer_idx in range(1, 27):
        file_name = f"layer_{layer_idx}.pth"
        file_path = f"{math_expert_output_dir}/{file_name}"
        data = torch.load(file_path)
        data = data.mean(dim=1)
        # 1 7 13 19 25
        if layer_idx % 6 == 1:
            similarity_matrix = compute_similarity(data)
            sns.heatmap(similarity_matrix, fmt='d', cmap='YlGnBu')
            save_path = f"{save_dir}/layerwise_{file_name}.png"
            plt.savefig(save_path)  # 保存图像
            plt.close()
        layer_output.append(data)

    layer_output = torch.stack(layer_output, dim=0)
    global_level_output = layer_output.mean(dim=1)
    global_similarity_matrix = compute_similarity(global_level_output)
    sns.heatmap(global_similarity_matrix, fmt='d', cmap='YlGnBu')
    save_path = f"{save_dir}/global_similarity_matrix.png"
    plt.savefig(save_path)  # 保存图像
    plt.close()
