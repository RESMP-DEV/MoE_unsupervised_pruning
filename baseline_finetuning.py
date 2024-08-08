import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainingArguments
import torch
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

from data_utils import dataset_local_load, dataset_map_merge


def load_model():
    print(f"model id or path: {args.model_path}")
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True
    )

    print("=================model params=================")
    for k, v in llm.config.__dict__.items():
        print(f"{k}: {v}")
    print("=================model params end=================")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    return llm, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model directory to load.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Training result directory.")
    args = parser.parse_args()

    train_dataset_map, valid_dataset_map = dataset_local_load(args.dataset_dir)
    train_dataset = dataset_map_merge(train_dataset_map)
    valid_dataset = dataset_map_merge(valid_dataset_map)
    print("data loaded.")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")

    model, tokenizer = load_model()
    print("model loaded.")

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

    model = get_peft_model(model, peft_config)
    # for name, param in model.named_parameters():
    #     print(f"Parameter Name: {name} | Size: {param.size()} | Type: {param.data.dtype} | Trainable: {param.requires_grad} \n")

    train_args = TrainingArguments(
        args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        save_steps=1000,
        eval_steps=200,
        logging_steps=200,
        learning_rate=1e-5,
        weight_decay=0.001,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
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
    trainer.save_model(args.output_dir)



