import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from trl import SFTTrainer

from data_utils import dataset_local_load, dataset_map_merge


def load_model(pruned_mask):
    print(f"model id or path: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    return llm, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model directory to load.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Training result directory.")
    args = parser.parse_args()

    train_dataset_map, valid_dataset_map = dataset_local_load(args.dataset_dir)
    print("data loaded.")

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
    for dataset_name in train_dataset_map:
        pruning_file_path = f"pruned_result/{dataset_name}_pruning.json"
        with open(pruning_file_path, "r") as f:
            pruned_mask = json.load(f)

        model, tokenizer = load_model(pruned_mask)
        print("model loaded.")

        if len(trained_dataset) != 0:
            peft_model_path = os.path.join(args.output_dir, "_".join(trained_dataset))
            model = PeftModel.from_pretrained(model, peft_model_path)
        else:
            model = get_peft_model(model, peft_config)

        # for name, param in model.named_parameters():
        #     print(f"Parameter Name: {name} | Size: {param.size()} | Type: {param.data.dtype} | Trainable: {param.requires_grad} \n")

        trained_dataset.append(dataset_name)
        output_dir = os.path.join(args.output_dir, "_".join(trained_dataset))
        train_args = TrainingArguments(
            output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_32bit",
            save_steps=1000,
            eval_steps=200,
            logging_steps=1,
            learning_rate=1e-5,
            weight_decay=0.001,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            report_to=["tensorboard"],
        )

        train_dataset = train_dataset_map[dataset_name]
        valid_dataset = valid_dataset_map[dataset_name]
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
        trainer.save_model(args.output_dir)



