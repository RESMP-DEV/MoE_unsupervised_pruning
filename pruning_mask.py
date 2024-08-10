import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.cluster import KMeans
import pandas as pd
import json
import os

from data_utils import dataset_local_load


def calibration_generation(dataset_dir, dataset_name, sample_number=50):
    train_dataset_map, valid_dataset_map = dataset_local_load(dataset_dir)
    train_dataset = train_dataset_map[dataset_name]
    train_df = pd.DataFrame(train_dataset)
    train_df = train_df.sample(n=sample_number, random_state=1, axis=0)
    return train_df


class PreTrainedDeepseekV2PrunerByDomain:
    def __init__(self, model, calibration_data, batch_size=16):
        self.model = model.model
        self.tokenizer = tokenizer
        self.unsupervised_method = KMeans(n_clusters=model.config.num_experts_per_tok, random_state=0)
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_unsupervised_map(self):
        print("unsupervising...")
        prompt = list(self.calibration_data["prompt"])
        completion = list(self.calibration_data["completion"])

        pre_ffn_hidden_states = None
        batch_total = len(prompt) // self.batch_size + 1
        for batch in range(batch_total):
            start = min(batch * self.batch_size, len(prompt))
            end = min((batch + 1) * self.batch_size, len(prompt))
            if end == start:
                continue
            basic_prompt = prompt[start:end]
            basic_completion = completion[start:end]
            inputs = self.tokenizer(basic_prompt, basic_completion, return_tensors='pt', max_length=128,
                                    padding='max_length', truncation=True)
            # data to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            # inference and expert score compute
            self.model.eval()
            # inference
            output = self.model(input_ids, attention_mask, output_hidden_states=True, pre_ffn_hidden=True)
            print(f"batch {batch}/{batch_total - 1} feed forward finish.")
            # ffn state merge
            if batch == 0:
                pre_ffn_hidden_states = list(output.pre_ffn_hidden_states)
                assert len(pre_ffn_hidden_states) == len(self.model.layers)
            else:
                update_pre_ffn_hidden_states = list(output.pre_ffn_hidden_states)
                for layer_idx, layer in enumerate(pre_ffn_hidden_states):
                    pre_ffn_hidden_states[layer_idx] = torch.cat([layer, update_pre_ffn_hidden_states[layer_idx]],
                                                                 dim=0)
                assert len(pre_ffn_hidden_states) == len(self.model.layers)

        # expert score compute
        self.unsupervised_map = {}
        for idx, hidden_state in enumerate(pre_ffn_hidden_states):
            if "DeepseekV2MLP" in str(type(self.model.layers[idx].mlp)):
                continue

            score_weight = self.model.layers[idx].mlp.gate.weight
            scores = F.linear(hidden_state.type(torch.float32), score_weight.type(torch.float32), None)
            scores = scores.softmax(dim=-1, dtype=torch.float32).sum(0).sum(0).tolist()
            print(f"layer {idx} feed forward finish.")

            experts = self.model.layers[idx].mlp.experts
            experts_output = []
            for expert in experts:
                # .flatten() -- [bs, seq_len, hidden] ==> [bs * seq_len * hidden]
                # basic_output = expert(hidden_state).flatten().type(torch.float16)
                # .sum(1).flatten() -- [bs, seq_len, hidden] ==> [bs, hidden] == > [bs * hidden]
                basic_output = expert(hidden_state).sum(1).flatten().type(torch.float16)
                experts_output.append(basic_output)
            print(f"layer {idx} expert forward finish.")

            experts_output = torch.stack(experts_output)
            kmeans_result = self.unsupervised_method.fit(experts_output.cpu().detach().numpy())
            cluster_label = kmeans_result.labels_.tolist()
            assert len(cluster_label) == self.model.config.n_routed_experts

            self.unsupervised_map[idx] = {i: (cluster_label[i], scores[i]) for i in range(len(cluster_label))}

    def generate_pruned_map(self, prune_rate=0.5):
        print("pruning...")
        if self.unsupervised_map is None:
            raise ValueError("No existed unsupervised map.")
        self.pruned_map = {}
        for layer_idx, cluster_info in self.unsupervised_map.items():
            new_map = {}
            for expert_idx, (cluster_label, score) in cluster_info.items():
                if cluster_label not in new_map:
                    new_map[cluster_label] = {}
                new_map[cluster_label][expert_idx] = score

            for cluster_label, cluster_info in new_map.items():
                cluster_length = len(cluster_info)
                if cluster_length > 1:
                    cluster_pruned_info = sorted(cluster_info.items(), key=lambda x: x[1])[
                                          :int(prune_rate * cluster_length)]
                    if layer_idx not in self.pruned_map:
                        self.pruned_map[layer_idx] = []
                    self.pruned_map[layer_idx] += [info[0] for info in cluster_pruned_info]


if __name__ == "__main__":
    model_name = "/home/zhongyuan_peng/.cache/modelscope/hub/deepseek-ai/DeepSeek-V2-Lite"
    dataset_dir = "dataset"
    dataset_name_list = ["MathInstruct", "finance_alpaca", "code_alpaca_20k"]
    sample_number = 200

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)

    for dataset_name in dataset_name_list:
        train_df = calibration_generation(dataset_dir, dataset_name, sample_number=sample_number)
        pruner = PreTrainedDeepseekV2PrunerByDomain(model, train_df)

        unsupervised_save_path = f"pruned_result/{dataset_name}_unsupervised.json"
        if not os.path.exists(unsupervised_save_path):
            pruner.generate_unsupervised_map()
            with open(unsupervised_save_path, 'w') as f:
                json.dump(pruner.unsupervised_map, f)
            print(f"unsupervised map saved to {unsupervised_save_path}")
        else:
            with open(unsupervised_save_path, 'r') as f:
                pruner.unsupervised_map = json.load(f)
            print(f"unsupervised map is loaded from {unsupervised_save_path}")

        prune_save_file_path = f"pruned_result/{dataset_name}_prune.json"
        if not os.path.exists(prune_save_file_path):
            pruner.generate_pruned_map()
            with open(prune_save_file_path, 'w') as f:
                json.dump(pruner.pruned_map, f)
            print(f"pruned map saved to {prune_save_file_path}")
