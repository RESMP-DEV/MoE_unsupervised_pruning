import argparse
import os
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from trl import SFTTrainer
from datasets import Dataset

from data_utils import dataset_local_load, dataset_map_merge


def load_model(pruned_mask):
    print(f"model id or path: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.pre_ffn_hidden = True
    config.pruned_mask = pruned_mask

    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("=================model params=================")
    for k, v in llm.config.__dict__.items():
        print(f"{k}: {v}")
    print("=================model params end=================")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="right")
    return llm, tokenizer


def domain_finetuning():
    train_dataset_map, valid_dataset_map = dataset_local_load(args.dataset_dir, train_length=args.train_length,
                                                              valid_length=args.valid_length)
    print("data loaded.")
    for dataset in train_dataset_map:
        print(f"{dataset} training dataset size: {len(train_dataset_map[dataset])}")
        print(f"{dataset} validation dataset size: {len(valid_dataset_map[dataset])}")

    output_dir = os.path.join(args.output_dir, f"train_{args.train_length}_valid_{args.valid_length}")
    output_dir = os.path.join(output_dir, "pruning_finetuning")
    output_dir = os.path.join(output_dir, f"sample_{args.prune_sample_number}_cluster_{args.prune_cluster_number}")
    output_dir = os.path.join(output_dir, f"{args.prune_method}_{args.prune_rate}")
    print(f"output_dir: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    trained_dataset = []
    dataset_name_list = sorted(list(train_dataset_map.keys()))
    for dataset_name in dataset_name_list:
        pruning_file_path = f"pruned_result/sample_{args.prune_sample_number}_cluster_{args.prune_cluster_number}/{args.prune_method}_{args.prune_rate}/{dataset_name}_prune.json"
        print(f"pruning_file_path: {pruning_file_path}")
        with open(pruning_file_path, "r") as f:
            pruned_mask = json.load(f)

        model, tokenizer = load_model(pruned_mask)
        print("model loaded.")

        if len(trained_dataset) != 0:
            for i in range(len(trained_dataset)):
                peft_model_path = os.path.join(output_dir, "_".join(trained_dataset[:(i + 1)]))
                model = PeftModel.from_pretrained(model, peft_model_path)
                model = model.merge_and_unload()
                print(f"model peft merge from {peft_model_path}")
        model = get_peft_model(model, peft_config)

        trained_dataset.append(dataset_name)
        output_path = os.path.join(output_dir, "_".join(trained_dataset))

        if os.path.exists(output_path):
            print(f"{output_path} already exists.")
            continue
        # for name, param in model.named_parameters():
        #     print(f"Parameter Name: {name} | Size: {param.size()} | Type: {param.data.dtype} | Trainable: {param.requires_grad} \n")
        train_args = TrainingArguments(
            output_path,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            # gradient_checkpointing=True,
            # gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_32bit",
            save_steps=500,
            eval_steps=200,
            logging_steps=1,
            learning_rate=1e-5,
            weight_decay=0.001,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            report_to=["tensorboard"],
        )

        train_data_list = train_dataset_map[dataset_name]
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data_list).reset_index(drop=True))
        valid_data_list = valid_dataset_map[dataset_name]
        valid_dataset = Dataset.from_pandas(pd.DataFrame(valid_data_list).reset_index(drop=True))
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            max_seq_length=2048,
            tokenizer=tokenizer,
            args=train_args,
        )

        print(f"{dataset_name} prune-masked training...")
        trainer.train()
        print(f"{dataset_name} prune-masked training end.")
        trainer.save_model(output_path)

        torch.cuda.empty_cache()


def seer_finetuning():
    train_dataset_map, valid_dataset_map = dataset_local_load(args.dataset_dir, train_length=args.train_length,
                                                              valid_length=args.valid_length)
    train_dataset = dataset_map_merge(train_dataset_map)
    valid_dataset = dataset_map_merge(valid_dataset_map)
    print("data loaded.")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")

    output_dir = os.path.join(args.output_dir, f"train_{args.train_length}_valid_{args.valid_length}")
    output_dir = os.path.join(output_dir, "pruning_finetuning")
    output_dir = os.path.join(output_dir, f"sample_{args.prune_sample_number}")
    output_dir = os.path.join(output_dir, f"{args.prune_method}_{args.prune_rate}")
    print(f"output_dir: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    pruning_file_path = f"pruned_result/sample_{args.prune_sample_number}/{args.prune_method}_{args.prune_rate}/c4_prune.json"
    print(f"pruning_file_path: {pruning_file_path}")
    with open(pruning_file_path, "r") as f:
        pruned_mask = json.load(f)

    model, tokenizer = load_model(pruned_mask)
    print("model loaded.")

    model = get_peft_model(model, peft_config)
    train_args = TrainingArguments(
        output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        save_steps=500,
        eval_steps=200,
        logging_steps=1,
        learning_rate=1e-5,
        weight_decay=0.001,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        report_to=["tensorboard"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=train_args,
    )
    print("training...")
    trainer.train()
    print("training end.")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model directory to load.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Training result directory.")
    parser.add_argument("--train_length", type=int, required=True, help="Train dataset length.")
    parser.add_argument("--valid_length", type=int, required=True, help="Valid dataset length.")
    parser.add_argument("--prune_sample_number", type=int, required=True, help="Pruning dataset sample number.")
    parser.add_argument("--prune_cluster_number", type=int, required=True, help="Pruning cluster number.")
    parser.add_argument("--prune_method", type=str, default="in_class_prune", help="Pruning method.")
    parser.add_argument("--prune_rate", type=float, default=0.5, help="Pruning rate.")
    args = parser.parse_args()

    # domain fine-tuning
    if "class_prune" in args.prune_method:
        domain_finetuning()

    # seer fine-tuning
    if args.prune_method == "seer_prune":
        seer_finetuning()
