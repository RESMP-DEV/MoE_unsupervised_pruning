import json
import os
import random
import pandas as pd
from datasets import Dataset, load_dataset

FIELD_DICT = {
    "MathInstruct": "math", "finance_alpaca": "finance", "code_alpaca_20k": "coding"
}

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task in the field of {field}, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task in the field of {field}. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def train_valid_split_and_sample(dataset_dir_path, train_length, valid_length):
    files = sorted(os.listdir(dataset_dir_path))
    for file_name in files:
        dataset_name = file_name.split(".")[0]
        if (("code" in dataset_name.lower() or "finance" in dataset_name.lower() or "math" in dataset_name.lower())
                and not file_name.endswith("train.json")
                and not file_name.endswith("valid.json")):
            file_name_path = os.path.join(dataset_dir_path, file_name)
            train_sample_path = os.path.join(dataset_dir_path, f"{dataset_name}_{train_length}_train.json")
            valid_sample_path = os.path.join(dataset_dir_path, f"{dataset_name}_{valid_length}_valid.json")
            if os.path.exists(train_sample_path) and os.path.exists(valid_sample_path):
                print(f"{dataset_name} train valid data already exists.")
                continue
            with open(file_name_path, "r") as f:
                dataset_list = json.load(f)
                if len(dataset_list) < train_length + valid_length:
                    raise ValueError("dataset length is not enough!")
                train_valid_sample_split_point = int(len(dataset_list) * train_length / (train_length + valid_length))
                train_sample_range = dataset_list[:train_valid_sample_split_point]
                valid_sample_range = dataset_list[train_valid_sample_split_point:]
                train_sample = random.sample(train_sample_range, train_length)
                valid_sample = random.sample(valid_sample_range, valid_length)

            with open(train_sample_path, "w") as f:
                json.dump(train_sample, f)
            with open(valid_sample_path, "w") as f:
                json.dump(valid_sample, f)


def format_parse(message_map, dataset_name):
    message_map["field"] = FIELD_DICT[dataset_name]
    if "input" in message_map and message_map["input"] != "":
        prompt = PROMPT_DICT["prompt_input"]
    else:
        prompt = PROMPT_DICT["prompt_no_input"]
    return {"prompt": prompt.format_map(message_map), "completion": message_map["output"]}


def dataset_local_load(dataset_dir_path, train_length=5000, valid_length=500):
    files = sorted(os.listdir(dataset_dir_path))
    train_dataset_map = {}
    for file_name in files:
        if file_name.endswith(f"_{train_length}_train.json"):
            file_name_path = os.path.join(dataset_dir_path, file_name)
            dataset_name = file_name.split(f"_{train_length}_train.json")[0]
            with open(file_name_path, "r") as f:
                dataset_list = json.load(f)
                res_list = []
                for data in dataset_list:
                    if "source" in data and "/CoT/" not in data["source"]:
                        continue
                    res_list.append(format_parse(data, dataset_name))
            train_dataset_map[dataset_name] = res_list

    valid_dataset_map = {}
    for file_name in files:
        if file_name.endswith(f"_{valid_length}_valid.json"):
            file_name_path = os.path.join(dataset_dir_path, file_name)
            dataset_name = file_name.split(f"_{valid_length}_valid.json")[0]
            with open(file_name_path, "r") as f:
                dataset_list = json.load(f)
                res_list = []
                for data in dataset_list:
                    if "source" in data and "/CoT/" not in data["source"]:
                        continue
                    res_list.append(format_parse(data, dataset_name))
            valid_dataset_map[dataset_name] = res_list

    return train_dataset_map, valid_dataset_map


def dataset_map_merge(dataset_map):
    dataset_df = pd.DataFrame()
    for i, (k, v) in enumerate(dataset_map.items()):
        if i == 0:
            dataset_df = pd.DataFrame(v)
        else:
            dataset_df = pd.concat([dataset_df, pd.DataFrame(v)], axis=0)
    return Dataset.from_pandas(dataset_df.reset_index(drop=True))


def merge_dataset_lists_to_json(dataset_lists, save_path):
    if os.path.exists(save_path):
        print("file already exists")
        return
    dataset_list = sum(dataset_lists, [])
    with open(save_path, "w") as f:
        json.dump(dataset_list, f)


if __name__ == "__main__":
    dataset_dir = "dataset"
    train_length = 5000
    valid_length = 500
    # train_valid_split_and_sample(dataset_dir, train_length, valid_length)

    train_dataset_map, valid_dataset_map = dataset_local_load(dataset_dir, train_length=train_length, valid_length=valid_length)
    # train_dataset = dataset_map_merge(train_dataset_map)
    # valid_dataset = dataset_map_merge(valid_dataset_map)
