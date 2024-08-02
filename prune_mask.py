import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.cluster import KMeans
import pandas as pd
import json

from data_utils import dataset_local_load


def calibration_generation(dataset_dir, dataset_name, sample_number=50):
    train_dataset_map, valid_dataset_map = dataset_local_load(dataset_dir)
    train_dataset = train_dataset_map[dataset_name]
    train_df = pd.DataFrame(train_dataset)
    train_df = train_df.sample(n=sample_number, random_state=1, axis=0)
    return train_df

class PreTrainedDeepseekV2PrunerByDomain:
    def __init__(self, model, calibration_data):
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.unsupervised_method = KMeans(n_clusters=model.config.num_experts_per_tok, random_state=0)
        self.calibration_data = calibration_data

        self.generate_unsupervised_map()
        self.generate_pruned_map()

    def generate_unsupervised_map(self):
        prompt = list(self.calibration_data["prompt"])
        # completion = list(self.calibration_data["completion"])
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        output = self.model(input_ids, attention_mask)
        pre_ffn_hidden_states = output.pre_ffn_hidden_states
        assert len(pre_ffn_hidden_states) == len(self.model.layers)

        self.unsupervised_map = {}
        for idx, hidden_state in enumerate(pre_ffn_hidden_states):
            if "DeepseekV2MLP" in str(type(self.model.layers[idx].mlp)):
                continue

            score_weight = self.model.layers[idx].mlp.gate.weight
            scores = F.linear(hidden_state.type(torch.float32), score_weight.type(torch.float32), None)
            scores = scores.softmax(dim=-1, dtype=torch.float32).sum(0).sum(0).tolist()

            experts = self.model.layers[idx].mlp.experts
            experts_output = []
            for expert in experts:
                basic_output = expert(hidden_state).sum(1).flatten().type(torch.float16)
                experts_output.append(basic_output)

            experts_output = torch.stack(experts_output)
            kmeans_result = self.unsupervised_method.fit(experts_output.detach().numpy())
            cluster_label = kmeans_result.labels_.tolist()
            assert len(cluster_label) == self.model.config.n_routed_experts

            self.unsupervised_map[idx] = [[cluster_label[i], scores[i]] for i in range(len(cluster_label))]

    def generate_pruned_map(self, prune_rate=0.5):
        if self.unsupervised_map is None:
            raise ValueError("No existed unsupervised map.")
        self.pruned_map = {}
        for layer_idx, cluster_info in self.unsupervised_map.items():
            new_map = {}
            for expert_idx, (cluster_label, score) in enumerate(cluster_info):
                if cluster_label not in new_map:
                    new_map[cluster_label] = {}
                new_map[cluster_label][expert_idx] = score

            for cluster_label, cluster_info in new_map.items():
                cluster_length = len(cluster_info)
                if cluster_length > 1:
                    cluster_pruned_info = sorted(cluster_info.items(), key=lambda x: x[1])[:int(prune_rate * cluster_length)]
                    if layer_idx not in self.pruned_map:
                        self.pruned_map[layer_idx] = []
                    self.pruned_map[layer_idx] += [info[0] for info in cluster_pruned_info]


if __name__ == "__main__":
    model_name = "/home/zhongyuan_peng/.cache/modelscope/hub/deepseek-ai/DeepSeek-V2-Lite"
    dataset_dir = "dataset"
    dataset_name_list = ["MathInstruct", "finance_alpaca", "code_alpaca_20k"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)

    for dataset_name in dataset_name_list:
        train_df = calibration_generation(dataset_dir, dataset_name)
        pruner = PreTrainedDeepseekV2PrunerByDomain(model, train_df)
        pruner.generate_unsupervised_map()
        pruner.generate_pruned_map()

        save_file_path = f"{dataset_name}_pruning.json"
        with open(save_file_path, 'w') as f:
            json.dump(pruner.pruned_map, f)
