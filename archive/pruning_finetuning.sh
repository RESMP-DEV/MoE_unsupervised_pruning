model_path="/vepfs/hongcheng/moe/modelscope_cache/deepseek-ai/DeepSeek-V2-Lite"
dataset_dir="dataset"
output_dir="result"
train_length=5000
valid_length=500
prune_sample_number=5000
prune_cluster_number=12
#prune_method="in_class_prune"
prune_method="seer_prune"
prune_rate=0.2
CUDA_VISIBLE_DEVICES="2,3" python pruning_finetuning.py \
                 --model_path $model_path \
                 --dataset_dir $dataset_dir \
                 --output_dir $output_dir \
                 --train_length $train_length \
                 --valid_length $valid_length \
                 --prune_sample_number $prune_sample_number \
                 --prune_cluster_number $prune_cluster_number \
                 --prune_method $prune_method \
                 --prune_rate $prune_rate
