import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import pandas as pd
import numpy as np
import json
import os
import math

from utils.data_utils import dataset_local_load
from utils.hsic_utils import hsic_split_graph
from utils.uns_utils import calculate_entropy, cluster_for_prune


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


def c4_calibration_generation(dataset_path, sample_number=1000):
    dataset = load_dataset('json', data_files={'train': f'{dataset_path}'})
    df = pd.DataFrame(dataset["train"])[["text"]]
    df = df.sample(n=sample_number, random_state=1, axis=0)
    return df


class PreTrainedMoEPruner:
    def __init__(self, model, tokenizer, calibration_data, batch_size=16, layerwise_pruning_method="hierarchical_prune",
                 global_pruning_method="hierarchical_prune",
                 cluster_number=12, sentence_emd="avg_pooling"):
        self.model = model.model
        self.tokenizer = tokenizer
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.layerwise_pruning_method = layerwise_pruning_method
        self.global_pruning_method = global_pruning_method
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

    def generate_layerwise_unsupervised_map(self, expert_output_save_dir=None):
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

            experts_output = torch.stack(experts_output)  # (expert_num, bs, hidden)
            if expert_output_save_dir:
                expert_output_save_path = f"{expert_output_save_dir}/layer_{idx}.pth"
                if not os.path.exists(expert_output_save_path):
                    torch.save(experts_output, expert_output_save_path)
                    print(f"layer {idx} expert output save finish.")
                else:
                    print(f"layer {idx} expert output has already existed.")

            if "prompt" in self.calibration_data.columns:
                calib_dataset_length = len(self.calibration_data["prompt"])
            elif "text" in self.calibration_data.columns:
                calib_dataset_length = len(self.calibration_data["text"])
            else:
                raise NotImplementedError("Calibration data has no available column.")

            entropy = None
            if self.layerwise_pruning_method.startswith("entropy"):
                emb = experts_output.permute(1, 0, 2)  # (bs, expert_num, hidden)
                entropy = calculate_entropy(emb.to(torch.float16))

            self.unsupervised_map[idx] = []
            batch_total = calib_dataset_length // self.batch_size + 1
            for batch in range(batch_total):
                start = min(batch * self.batch_size, calib_dataset_length)
                end = min((batch + 1) * self.batch_size, calib_dataset_length)
                if end == start:
                    continue

                basic_score = scores[start:end, :].mean(0).tolist()  # router score, which has been abandoned
                basic_output = experts_output[:, start:end, :].to(torch.float16)
                basic_unsupervised_data = basic_output.cpu().detach().numpy()

                basic_cluster_label, distances_to_center = cluster_for_prune(basic_unsupervised_data, self.cluster_number, pruning_method=self.layerwise_pruning_method, entropy=entropy)
                basic_cluster_final_result = [(basic_cluster_label[i], distances_to_center[i]) for i in
                                              range(len(basic_cluster_label))]
                self.unsupervised_map[idx].append(basic_cluster_final_result)
            print(f"layer {idx} {self.layerwise_pruning_method} unsupervised learning finish.")

    def seer_prune(self, prune_rate=0.2):
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

    def hsic_prune(self, prune_rate=0.2):
        print("hsic pruning...")
        self.hsic_map = {}
        for idx, hidden_state in enumerate(self.pre_ffn_hidden_states):
            if "MLP" in str(type(self.model.layers[idx].mlp)):
                continue

            # for fast debugging
            # if idx == 2:
            #     break

            with torch.no_grad():
                score_weight = self.model.layers[idx].mlp.gate.weight
                scores = F.linear(hidden_state.to(score_weight.device).type(torch.float32),
                                  score_weight.type(torch.float32), None)
                scores = scores.softmax(dim=-1, dtype=torch.float32)
                print(f"layer {idx} feed forward finish.")

                experts = self.model.layers[idx].mlp.experts
                experts_output = []
                for expert in experts:
                    basic_output = expert(hidden_state.to(self.device)).cpu()
                    experts_output.append(basic_output)
            print(f"layer {idx} experts forward finish.")

            experts_output = torch.stack(experts_output)  # [expert_num, bs, hidden]
            unsupervised_data = experts_output.to(torch.float32).cpu().detach().numpy()
            subgraphs = hsic_split_graph(unsupervised_data, prune_rate)
            self.hsic_map[idx] = []
            for subgraph in subgraphs:
                if len(subgraph) == 1:
                    continue
                score = scores[:, subgraph].sum(0)
                max_value, max_index = torch.max(score, dim=0)
                for i in range(len(subgraph)):
                    if i != max_index.item():
                        self.hsic_map[idx].append(subgraph[i])
            print(f"layer {idx} {self.layerwise_pruning_method} pruning finish.")

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

    def generate_layerwise_pruned_map(self, prune_rate=0.2):
        print("layerwise pruning...")
        if self.unsupervised_map is None:
            raise ValueError("No existed unsupervised map.")
        self.layerwise_pruned_map = {}
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
                        selected_cluster_info = sorted(cluster_result_info.items(), key=lambda x: x[1])[
                                              :int(dynamic_coef * (1 - prune_rate) * cluster_length)]
                    for (selected_expert_idx, selected_expert_score) in selected_cluster_info:
                        selected_expert_map[selected_expert_idx] += 1

            self.selected_expert_map[layer_idx] = selected_expert_map
            pruned_experts = sorted(selected_expert_map.items(), key=lambda x: x[1])[:int(prune_rate * len(selected_expert_map))]
            self.layerwise_pruned_map[layer_idx] = sorted([info[0] for info in pruned_experts])

    def generate_global_pruned_map(self, expert_output_save_dir, prune_rate=0.2):
        remain_experts = {}
        expert_outputs = {}
        layerwise_remain_length = None
        for idx in range(len(self.model.layers)):
            if "MLP" in str(type(self.model.layers[idx].mlp)):
                continue

            expert_output_save_path = f"{expert_output_save_dir}/layer_{idx}.pth"
            expert_output = torch.load(expert_output_save_path)  # (expert_num, bs, hidden)
            layerwise_pruned_expert = sorted(self.layerwise_pruned_map[str(idx)])
            if layerwise_remain_length is None:
                layerwise_remain_length = expert_output.shape[0] - len(layerwise_pruned_expert)
            else:
                assert layerwise_remain_length == expert_output.shape[0] - len(layerwise_pruned_expert)
            all_expert = torch.arange(expert_output.shape[0])
            remain_expert = all_expert[~torch.isin(all_expert, torch.tensor(layerwise_pruned_expert))]
            remain_experts[idx] = remain_expert
            remain_expert_output = expert_output[remain_expert]
            expert_outputs[idx] = remain_expert_output

        remain_experts_list = sorted(remain_experts.items(), key=lambda x: x[0])
        expert_outputs_list = [ele[1] for ele in sorted(expert_outputs.items(), key=lambda x: x[0])]
        expert_outputs = torch.cat(expert_outputs_list, dim=0)

        print("global pruning...")
        unsupervised_cross_layer_cluster_result = []
        experts_num, dataset_length, hidden_size = expert_outputs.size()
        batch_total = dataset_length // self.batch_size + 1
        for batch in range(batch_total):
            start = min(batch * self.batch_size, dataset_length)
            end = min((batch + 1) * self.batch_size, dataset_length)
            if end == start:
                continue

            basic_output = expert_outputs[:, start:end, :].to(torch.float16)
            basic_unsupervised_data = basic_output.cpu().detach().numpy()

            basic_cluster_label, distances_to_center = cluster_for_prune(basic_unsupervised_data, self.cluster_number,
                                                                         pruning_method=self.global_pruning_method)
            basic_cluster_final_result = [(basic_cluster_label[i], distances_to_center[i]) for i in
                                          range(len(basic_cluster_label))]
            unsupervised_cross_layer_cluster_result.append(basic_cluster_final_result)
            print(f"batch {batch}/{batch_total} done")

        selected_global_expert_map = {i: 0 for i in range(experts_num)}
        for b_idx, batch_cluster_info in enumerate(unsupervised_cross_layer_cluster_result):
            cluster_map = {}
            for cross_layer_expert_idx, (cluster_label, score) in enumerate(batch_cluster_info):
                if cluster_label not in cluster_map:
                    cluster_map[cluster_label] = {}
                cluster_map[cluster_label][cross_layer_expert_idx] = score

            for cluster_label, cluster_result_info in cluster_map.items():
                cluster_length = len(cluster_result_info)

                if cluster_length < 3:
                    selected_cluster_info = cluster_result_info.items()
                else:
                    dynamic_coef = self.dynamic_coef_compute(cluster_length)
                    selected_cluster_info = sorted(cluster_result_info.items(), key=lambda x: x[1])[
                                            :int(dynamic_coef * (1 - prune_rate) * cluster_length)]
                for (selected_expert_idx, selected_expert_score) in selected_cluster_info:
                    selected_global_expert_map[selected_expert_idx] += 1

        pruned_experts = sorted(selected_global_expert_map.items(), key=lambda x: x[1])[:int(prune_rate * experts_num)]
        self.global_pruned_map = {}
        for (idx, _) in pruned_experts:
            loc = idx // layerwise_remain_length
            layer_idx, remain_experts = remain_experts_list[loc]
            expert_idx = idx % layerwise_remain_length
            if layer_idx not in self.global_pruned_map:
                self.global_pruned_map[layer_idx] = []
            self.global_pruned_map[layer_idx].append(int(remain_experts[expert_idx]))

        # merge with layerwise result
        for layer_idx, prune_list in self.global_pruned_map.items():
            layerwise_prune_list = self.layerwise_pruned_map[str(layer_idx)]
            assert not set(layerwise_prune_list) & set(prune_list)
            self.global_pruned_map[layer_idx] = sorted(layerwise_prune_list + prune_list)


def domain_pruning():
    print("Domain Pruning...")
    base_path = f"pruned_result/{model_name}/sample_{sample_number}_cluster_{cluster_number}/{layerwise_pruning_method}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for dataset_name in dataset_name_list:
        print(f"Dataset: {dataset_name}")
        train_df = domain_calibration_generation(dataset_dir, dataset_name, sample_number=sample_number)
        pruner = PreTrainedMoEPruner(model, tokenizer, train_df,
                                     batch_size=batch_size,
                                     layerwise_pruning_method=layerwise_pruning_method,
                                     global_pruning_method=global_pruning_method,
                                     cluster_number=cluster_number)

        unsupervised_save_path = f"{base_path}/{dataset_name}_unsupervised.json"
        expert_output_save_dir = f"{base_path}/{dataset_name}_expert_output_hidden"
        if not os.path.exists(expert_output_save_dir):
            os.makedirs(expert_output_save_dir)
        if os.path.exists(unsupervised_save_path):
            with open(unsupervised_save_path, 'r') as f:
                pruner.unsupervised_map = json.load(f)
            print(f"unsupervised map is loaded from {unsupervised_save_path}")
        else:
            pruner.forward()
            pruner.generate_layerwise_unsupervised_map(expert_output_save_dir)
            with open(unsupervised_save_path, 'w') as f:
                json.dump(pruner.unsupervised_map, f)
            print(f"unsupervised map saved to {unsupervised_save_path}")

        layerwise_prune_save_dir = f"{base_path}/rate_{layerwise_prune_rate}"
        if not os.path.exists(layerwise_prune_save_dir):
            os.makedirs(layerwise_prune_save_dir)
        layerwise_prune_save_file_path = f"{layerwise_prune_save_dir}/{dataset_name}_prune.json"
        if os.path.exists(layerwise_prune_save_file_path):
            with open(layerwise_prune_save_file_path, 'r') as f:
                pruner.layerwise_pruned_map = json.load(f)
            print(f"pruned map is loaded from {layerwise_prune_save_file_path}")
        else:
            print(f"layerwise pruning_method: {layerwise_pruning_method}, prune_rate: {layerwise_prune_rate}")
            pruner.generate_layerwise_pruned_map(prune_rate=layerwise_prune_rate)
            with open(layerwise_prune_save_file_path, 'w') as f:
                json.dump(pruner.layerwise_pruned_map, f)
            print(f"layerwise pruned map saved to {layerwise_prune_save_file_path}")

        if global_pruning_method:
            global_prune_save_dir = f"{base_path}/rate_{layerwise_prune_rate}/rate_{global_prune_rate}"
            if not os.path.exists(global_prune_save_dir):
                os.makedirs(global_prune_save_dir)
            global_prune_save_file_path = f"{global_prune_save_dir}/{dataset_name}_prune.json"
            if os.path.exists(global_prune_save_file_path):
                with open(global_prune_save_file_path, 'r') as f:
                    pruner.global_pruned_map = json.load(f)
                print(f"global pruned map is loaded from {global_prune_save_file_path}")
            else:
                print(f"global pruning_method: {global_pruning_method}, prune_rate: {global_prune_rate}")
                pruner.generate_global_pruned_map(expert_output_save_dir, prune_rate=global_prune_rate)
                with open(global_prune_save_file_path, 'w') as f:
                    json.dump(pruner.global_pruned_map, f)
                print(f"global pruned map saved to {global_prune_save_file_path}")


def agnostic_pruning():
    print("Agnostic Pruning...")
    base_path = f"pruned_result/{model_name}/sample_{sample_number}_cluster_{cluster_number}/{layerwise_pruning_method}"
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

    pruner = PreTrainedMoEPruner(model, tokenizer, train_df,
                                 batch_size=batch_size,
                                 layerwise_pruning_method=layerwise_pruning_method,
                                 global_pruning_method=global_pruning_method,
                                 cluster_number=cluster_number)
    pruner.forward()
    unsupervised_save_path = f"{base_path}/{dataset_name_list[0]}_unsupervised.json"
    expert_output_save_dir = f"{base_path}/{dataset_name_list[0]}_expert_output_hidden"
    if not os.path.exists(expert_output_save_dir):
        os.makedirs(expert_output_save_dir)
    if os.path.exists(unsupervised_save_path):
        with open(unsupervised_save_path, 'r') as f:
            pruner.unsupervised_map = json.load(f)
        print(f"unsupervised map is loaded from {unsupervised_save_path}")
    else:
        pruner.generate_layerwise_unsupervised_map(expert_output_save_dir)
        with open(unsupervised_save_path, 'w') as f:
            json.dump(pruner.unsupervised_map, f)
        print(f"unsupervised map saved to {unsupervised_save_path}")

    prune_save_dir = f"{base_path}/rate_{layerwise_prune_rate}"
    if not os.path.exists(prune_save_dir):
        os.makedirs(prune_save_dir)
    prune_save_file_path = f"{prune_save_dir}/{dataset_name_list[0]}_prune.json"
    if os.path.exists(prune_save_file_path):
        print(f"pruned map is located at {prune_save_file_path}")
    else:
        print(f"layerwise pruning_method: {layerwise_pruning_method}, prune_rate: {layerwise_prune_rate}")
        pruner.generate_layerwise_pruned_map(prune_rate=layerwise_prune_rate)
        with open(prune_save_file_path, 'w') as f:
            json.dump(pruner.pruned_map, f)
        print(f"layerwise pruned map saved to {prune_save_file_path}")

    if global_pruning_method:
        raise NotImplementedError("Global pruning has not support for agnostic pruning")


def seer_pruning():
    print("Seer Pruning...")
    base_path = f"pruned_result/{model_name}/sample_{sample_number}/{layerwise_pruning_method}/rate_{layerwise_prune_rate}"
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
    pruner = PreTrainedMoEPruner(model, tokenizer, train_df, batch_size=batch_size, layerwise_pruning_method=layerwise_pruning_method,
                                 cluster_number=cluster_number)
    pruner.forward()
    seer_base_path = f"{base_path}/{dataset_name_list[0]}_prune.json"
    if os.path.exists(seer_base_path):
        with open(seer_base_path, 'r') as f:
            pruner.unsupervised_map = json.load(f)
        print(f"seer map is loaded from {seer_base_path}")
    else:
        pruner.seer_prune(prune_rate=layerwise_prune_rate)
        with open(seer_base_path, 'w') as f:
            json.dump(pruner.seer_map, f)
        print(f"seer map saved to {seer_base_path}")


def hsic_pruning():
    print("HSIC Pruning...")
    base_path = f"pruned_result/{model_name}/sample_{sample_number}/{layerwise_pruning_method}/rate_{layerwise_prune_rate}"
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
    pruner = PreTrainedMoEPruner(model, tokenizer, train_df, batch_size=batch_size, layerwise_pruning_method=layerwise_pruning_method,
                                 cluster_number=cluster_number)
    pruner.forward()
    hsic_base_path = f"{base_path}/{dataset_name_list[0]}_prune.json"
    if os.path.exists(hsic_base_path):
        with open(hsic_base_path, 'r') as f:
            pruner.unsupervised_map = json.load(f)
        print(f"hsic map is loaded from {hsic_base_path}")
    else:
        pruner.hsic_prune(prune_rate=layerwise_prune_rate)
        with open(hsic_base_path, 'w') as f:
            json.dump(pruner.hsic_map, f)
        print(f"hsic map saved to {hsic_base_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model directory to load.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory.")
    parser.add_argument("--dataset_name_list", type=str, required=True, help="Dataset name list.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--sample_number", type=int, default=500, help="Number of samples.")
    parser.add_argument("--cluster_number", type=int, default=6, help="Number of cluster.")
    parser.add_argument("--by_domain", type=int, default=1, help="By domain or mix")
    parser.add_argument("--layerwise_pruning_method", type=str, default="hierarchical_prune", help="Layerwise pruning method.")
    parser.add_argument("--global_pruning_method", type=str, default="hierarchical_prune", help="Global pruning method.")
    parser.add_argument("--layerwise_prune_rate", type=float, default=0.2, help="Layerwise pruning rate.")
    parser.add_argument("--global_prune_rate", type=float, default=0.1, help="Global pruning rate.")
    args = parser.parse_args()

    model_path = args.model_path
    dataset_dir = args.dataset_dir
    dataset_name_list = args.dataset_name_list.split(",")
    batch_size = args.batch_size
    sample_number = args.sample_number
    cluster_number = args.cluster_number
    by_domain = args.by_domain
    layerwise_pruning_method = args.layerwise_pruning_method
    global_pruning_method = args.global_pruning_method
    layerwise_prune_rate = args.layerwise_prune_rate
    global_prune_rate = args.global_prune_rate

    model_name = model_path.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)

    if layerwise_pruning_method == "seer_prune":
        # seer pruning
        seer_pruning()
    elif layerwise_pruning_method == "hsic_prune":
        # hsic pruning
        hsic_pruning()
    elif by_domain:
        # domain pruning
        domain_pruning()
    else:
        # agnostic pruning
        agnostic_pruning()
