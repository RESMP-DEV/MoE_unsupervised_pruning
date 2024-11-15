import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from datasets import load_dataset
import pandas as pd
import numpy as np
import json
import os
import math

from data_utils import dataset_local_load


def domain_calibration_generation(dataset_dir, dataset_name, sample_number=50):
    train_dataset_map, valid_dataset_map = dataset_local_load(dataset_dir)
    train_dataset = train_dataset_map[dataset_name]
    train_df = pd.DataFrame(train_dataset)
    train_df = train_df.sample(n=sample_number, random_state=1, axis=0)
    return train_df

def mix_calibration_generation(dataset_dir, dataset_name_list, sample_number=50):
    train_dataset_map, valid_dataset_map = dataset_local_load(dataset_dir)
    res_df = pd.DataFrame()
    for dataset_name in dataset_name_list:
        train_dataset = train_dataset_map[dataset_name]
        train_df = pd.DataFrame(train_dataset)
        train_df = train_df.sample(n=sample_number, random_state=1, axis=0)
        res_df = pd.concat([res_df, train_df])
    return res_df


def c4_calibration_generation(dataset_path, sample_number=10000):
    dataset = load_dataset('json', data_files={'train': f'{dataset_path}'})
    df = pd.DataFrame(dataset["train"])[["text"]]
    df = df.sample(n=sample_number, random_state=1, axis=0)
    return df

class PreTrainedMoEPruner:
    def __init__(self, model, tokenizer, calibration_data, batch_size=16, pruning_method="kmeans_prune",
                 cluster_number=12, sentence_emd="avg_pooling"):
        self.model = model.model
        self.tokenizer = tokenizer
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.pruning_method = pruning_method
        self.cluster_number = cluster_number
        self.sentence_emd = sentence_emd
        self.device = model.device

    def seq_to_sentence(self, emb):
        # [bs, seq_len, hidden] == > [bs, hidden]
        if self.sentence_emd == "avg_pooling":
            return emb.mean(1)
        else:
            raise NotImplementedError("Sentence embedding method has not been implemented yet.")

    def forward(self):
        print("forwarding...")
        if "prompt" in self.calibration_data.columns:
            prompt = list(self.calibration_data["prompt"])
            # delete corresponding prompt text
            prompt = ["### Instruction:" + basic_prompt.split("### Instruction:")[1] for basic_prompt in prompt]
            completion = list(self.calibration_data["completion"])
            calib_dataset_length = len(self.calibration_data["prompt"])
            data_flag = 1
        elif "text" in self.calibration_data.columns:
            text = list(self.calibration_data["text"])
            calib_dataset_length = len(self.calibration_data["text"])
            data_flag = 2
        else:
            raise NotImplementedError("Calibration data has no available column.")

        self.pre_ffn_hidden_states = None
        batch_total = calib_dataset_length // self.batch_size + 1
        for batch in range(batch_total):
            start = min(batch * self.batch_size, calib_dataset_length)
            end = min((batch + 1) * self.batch_size, calib_dataset_length)
            if end == start:
                continue
            if data_flag == 1:
                basic_prompt = prompt[start:end]
                basic_completion = completion[start:end]
                inputs = self.tokenizer(basic_prompt, basic_completion, return_tensors='pt', max_length=128,
                                        padding='max_length', truncation=True)
            elif data_flag == 2:
                basic_text = text[start:end]
                inputs = self.tokenizer(basic_text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
            else:
                raise ValueError("data_flag error")
            # data to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            # inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_ids, attention_mask, output_hidden_states=True, pre_ffn_hidden=True)
            print(f"batch {batch}/{batch_total - 1} feed forward finish.")
            # ffn state merge
            if batch == 0:
                self.pre_ffn_hidden_states = [self.seq_to_sentence(layer).cpu() for layer in list(output.pre_ffn_hidden_states)]
                assert len(self.pre_ffn_hidden_states) == len(self.model.layers)
            else:
                update_pre_ffn_hidden_states = list(output.pre_ffn_hidden_states)
                for layer_idx, layer in enumerate(self.pre_ffn_hidden_states):
                    self.pre_ffn_hidden_states[layer_idx] = torch.cat([layer, self.seq_to_sentence(update_pre_ffn_hidden_states[layer_idx]).cpu()],
                                                                 dim=0)

                assert len(self.pre_ffn_hidden_states) == len(self.model.layers)

    def generate_unsupervised_map(self):
        # expert score compute
        print("unsupervise pruning...")
        self.unsupervised_map = {}
        for idx, hidden_state in enumerate(self.pre_ffn_hidden_states):
            if "MLP" in str(type(self.model.layers[idx].mlp)):
                continue

            # for fast debugging
            # if idx == 2:
            #     break

            with torch.no_grad():
                score_weight = self.model.layers[idx].mlp.gate.weight
                scores = F.linear(hidden_state.to(score_weight.device).type(torch.float32), score_weight.type(torch.float32), None)
                scores = scores.softmax(dim=-1, dtype=torch.float32)
                print(f"layer {idx} feed forward finish.")

                experts = self.model.layers[idx].mlp.experts
                experts_output = []
                for expert in experts:
                    basic_output = expert(hidden_state.to(self.device)).cpu()
                    experts_output.append(basic_output)

            print(f"layer {idx} experts forward finish.")

            experts_output = torch.stack(experts_output)  # [expert_num, bs, hidden]
            self.unsupervised_map[idx] = []
            calib_dataset_length = len(self.calibration_data["prompt"])
            batch_total = calib_dataset_length // self.batch_size + 1
            for batch in range(batch_total):
                start = min(batch * self.batch_size, calib_dataset_length)
                end = min((batch + 1) * self.batch_size, calib_dataset_length)
                if end == start:
                    continue

                basic_output = experts_output[:, start:end, :].view(len(experts), -1).to(torch.float16)
                basic_score = scores[start:end, :].mean(0).tolist()
                basic_unsupervised_data = basic_output.cpu().detach().numpy()
                if self.pruning_method == "kmeans_prune":
                    uns = KMeans(n_clusters=self.cluster_number, random_state=0)
                    basic_cluster_result = uns.fit(basic_unsupervised_data)
                    basic_cluster_label = basic_cluster_result.labels_.tolist()
                elif self.pruning_method == "hierarchical_prune":
                    Z = linkage(basic_unsupervised_data, method="ward")
                    basic_cluster_label = fcluster(Z, t=self.cluster_number, criterion='maxclust').tolist()
                else:
                    raise NotImplementedError("Unsupervised pruning method has not been implemented yet.")

                basic_cluster_final_result = [(basic_cluster_label[i], basic_score[i]) for i in
                                              range(len(basic_cluster_label))]
                self.unsupervised_map[idx].append(basic_cluster_final_result)
            print(f"layer {idx} {self.pruning_method} unsupervised learning finish.")

    def generate_seer_map(self, prune_rate=0.5):
        print("seer pruning...")
        seer_map_score_stat = {}
        for idx, hidden_state in enumerate(self.pre_ffn_hidden_states):
            if "MLP" in str(type(self.model.layers[idx].mlp)):
                continue

            # for fast debugging
            # if idx == 2:
            #     break

            with torch.no_grad():
                score_weight = self.model.layers[idx].mlp.gate.weight
                scores = F.linear(hidden_state.to(score_weight.device).type(torch.float32), score_weight.type(torch.float32), None)
                scores = scores.softmax(dim=-1, dtype=torch.float32)
                scores = scores.sum(dim=0, dtype=torch.float32)
                seer_map_score_stat[idx] = scores.cpu().detach().numpy().tolist()
                print(f"layer {idx} feed forward finish.")

        # Global Expert Pruning
        seer_score_info = []
        for k, v in seer_map_score_stat.items():
            for i, score in enumerate(v):
                seer_score_info.append([score, [k, i]])

        selected_seer_score_info = sorted(seer_score_info, key=lambda x: x[0])[:int(prune_rate * len(seer_score_info))]
        self.seer_map = {}
        for score, (layer_idx, expert_idx) in selected_seer_score_info:
            if layer_idx not in self.seer_map:
                self.seer_map[layer_idx] = []
            self.seer_map[layer_idx].append(expert_idx)
        print(f"global expert pruning finish.")

    def dynamic_coef_compute(self, cluster_length):
        if hasattr(self.model.config, "n_routed_experts"):
            total_expert_num = self.model.config.n_routed_experts
        elif hasattr(self.model.config, "num_experts"):
            total_expert_num = self.model.config.num_experts
        else:
            raise AttributeError("No suitable attribute representing number of experts")
        length_pctg = cluster_length / total_expert_num
        dynamic_coef = math.exp(0.5 - length_pctg) - math.exp(length_pctg - 0.5) / (math.exp(length_pctg - 0.5) + math.exp(0.5 - length_pctg))
        dynamic_coef *= 0.6
        dynamic_coef = dynamic_coef + 0.5
        return dynamic_coef

    def generate_pruned_map(self, prune_rate=0.5):
        print("pruning...")
        if self.unsupervised_map is None:
            raise ValueError("No existed unsupervised map.")
        self.pruned_map = {}
        self.selected_expert_map = {}
        for layer_idx, cluster_info in self.unsupervised_map.items():
            selected_expert_map = {i:0 for i in range(len(cluster_info[0]))}
            for b_idx, batch_cluster_info in enumerate(cluster_info):
                cluster_map = {}
                for expert_idx, (cluster_label, score) in enumerate(batch_cluster_info):
                    if cluster_label not in cluster_map:
                        cluster_map[cluster_label] = {}
                    cluster_map[cluster_label][expert_idx] = score

                for cluster_label, cluster_result_info in cluster_map.items():
                    cluster_length = len(cluster_result_info)

                    if cluster_length < 3:
                        selected_cluster_info = cluster_result_info.items()
                    else:
                        dynamic_coef = self.dynamic_coef_compute(cluster_length)
                        selected_cluster_info = sorted(cluster_result_info.items(), key=lambda x: x[1], reverse=True)[
                                              :int(dynamic_coef * (1 - prune_rate) * cluster_length)]
                    for (selected_expert_idx, selected_expert_score) in selected_cluster_info:
                        selected_expert_map[selected_expert_idx] += 1

            self.selected_expert_map[layer_idx] = selected_expert_map
            pruned_experts = sorted(selected_expert_map.items(), key=lambda x: x[1])[:int(prune_rate * len(selected_expert_map))]
            self.pruned_map[layer_idx] = [info[0] for info in pruned_experts]

def domain_pruning():
    print("Domain Pruning...")
    base_path = f"pruned_result/{model_name}/sample_{sample_number}_cluster_{cluster_number}/{pruning_method}/rate_{prune_rate}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for dataset_name in dataset_name_list:
        print(f"Dataset: {dataset_name}")
        train_df = domain_calibration_generation(dataset_dir, dataset_name, sample_number=sample_number)
        pruner = PreTrainedMoEPruner(model, tokenizer, train_df, batch_size=batch_size, pruning_method=pruning_method,
                                     cluster_number=cluster_number)
        pruner.forward()

        unsupervised_save_path = f"{base_path}/{dataset_name}_unsupervised.json"
        if os.path.exists(unsupervised_save_path):
            with open(unsupervised_save_path, 'r') as f:
                pruner.unsupervised_map = json.load(f)
            print(f"unsupervised map is loaded from {unsupervised_save_path}")
        else:
            pruner.generate_unsupervised_map()
            with open(unsupervised_save_path, 'w') as f:
                json.dump(pruner.unsupervised_map, f)
            print(f"unsupervised map saved to {unsupervised_save_path}")

        prune_save_file_path = f"{base_path}/{dataset_name}_prune.json"

        if os.path.exists(prune_save_file_path):
            print(f"pruned map is located at {prune_save_file_path}")
        else:
            print(f"pruning_method: {pruning_method}, prune_rate: {prune_rate}")
            pruner.generate_pruned_map(prune_rate=prune_rate)
            with open(prune_save_file_path, 'w') as f:
                json.dump(pruner.pruned_map, f)
            print(f"pruned map saved to {prune_save_file_path}")


def mix_pruning():
    print("Mix Pruning...")
    base_path = f"pruned_result/{model_name}/sample_{sample_number}_cluster_{cluster_number}/{pruning_method}/rate_{prune_rate}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    train_df = mix_calibration_generation(dataset_dir, dataset_name_list, sample_number=sample_number)
    pruner = PreTrainedMoEPruner(model, tokenizer, train_df, batch_size=batch_size,
                                 pruning_method=pruning_method, cluster_number=cluster_number)
    pruner.forward()
    unsupervised_save_path = f"{base_path}/mix_unsupervised.json"
    if os.path.exists(unsupervised_save_path):
        with open(unsupervised_save_path, 'r') as f:
            pruner.unsupervised_map = json.load(f)
        print(f"unsupervised map is loaded from {unsupervised_save_path}")
    else:
        pruner.generate_unsupervised_map()
        with open(unsupervised_save_path, 'w') as f:
            json.dump(pruner.unsupervised_map, f)
        print(f"unsupervised map saved to {unsupervised_save_path}")

    prune_save_file_path = f"{base_path}/mix_prune.json"
    if os.path.exists(prune_save_file_path):
        print(f"pruned map is located at {prune_save_file_path}")
    else:
        print(f"pruning_method: {pruning_method}, prune_rate: {prune_rate}")
        pruner.generate_pruned_map(prune_rate=prune_rate)
        with open(prune_save_file_path, 'w') as f:
            json.dump(pruner.pruned_map, f)
        print(f"pruned map saved to {prune_save_file_path}")


def seer_pruning():
    print("Seer Pruning...")
    base_path = f"pruned_result/{model_name}/sample_{sample_number}/{pruning_method}/rate_{prune_rate}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    assert len(dataset_name_list) == 1
    dataset_name = None
    for file in os.listdir(dataset_dir):
        if file.startswith(dataset_name_list[0]):
            dataset_name = file
            break
    assert dataset_name is not None

    dataset_path = f"{dataset_dir}/{dataset_name}"
    train_df = c4_calibration_generation(dataset_path, sample_number=sample_number)
    pruner = PreTrainedMoEPruner(model, tokenizer, train_df, batch_size=batch_size, pruning_method=pruning_method,
                                 cluster_number=cluster_number)
    pruner.forward()
    seer_base_path = f"{base_path}/{dataset_name_list[0]}_prune.json"
    if os.path.exists(seer_base_path):
        with open(seer_base_path, 'r') as f:
            pruner.unsupervised_map = json.load(f)
        print(f"seer map is loaded from {seer_base_path}")
    else:
        pruner.generate_seer_map(prune_rate=prune_rate)
        with open(seer_base_path, 'w') as f:
            json.dump(pruner.seer_map, f)
        print(f"seer map saved to {seer_base_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model directory to load.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory.")
    parser.add_argument("--dataset_name_list", type=str, required=True, help="Dataset name list.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--sample_number", type=int, default=500, help="Number of samples.")
    parser.add_argument("--cluster_number", type=int, default=6, help="Number of cluster.")
    parser.add_argument("--by_domain", type=int, default=1, help="By domain or mix")
    parser.add_argument("--pruning_method", type=str, default="in_class_prune", help="Pruning method.")
    parser.add_argument("--prune_rate", type=float, default=0.5, help="Pruning rate.")
    args = parser.parse_args()

    model_path = args.model_path
    dataset_dir = args.dataset_dir
    dataset_name_list = args.dataset_name_list.split(",")
    batch_size = args.batch_size
    sample_number = args.sample_number
    cluster_number = args.cluster_number
    by_domain = args.by_domain
    pruning_method = args.pruning_method
    prune_rate = args.prune_rate

    model_name = model_path.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)

    # seer pruning
    if pruning_method == "seer_prune":
        seer_pruning()
    elif by_domain:
        # domain pruning
        domain_pruning()
    else:
        # mix pruning
        mix_pruning()