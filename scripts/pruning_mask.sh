model_path="/home/test/.cache/modelscope/hub/Qwen/Qwen1___5-MoE-A2___7B"
#model_path="/home/test/.cache/modelscope/hub/deepseek-ai/DeepSeek-V2-Lite"
dataset_dir="dataset"
#dataset_name_list="c4"
dataset_name_list="MathInstruct,code_alpaca_20k,finance_alpaca"
batch_size=32
sample_number=1000
cluster_number=16 # only available when using uns method
by_domain=1 # only available when using uns method
#pruning_method="kmeans_prune"
pruning_method="hierarchical_prune"
#pruning_method="hierarchical_prune_with_entropy"
#pruning_method="seer_prune"
#pruning_method="hsic_prune"
prune_rate=0.2
CUDA_VISIBLE_DEVICES="4,5" python pruning_mask.py \
                 --model_path $model_path \
                 --dataset_dir $dataset_dir \
                 --dataset_name_list $dataset_name_list \
                 --batch_size $batch_size \
                 --sample_number $sample_number \
                 --cluster_number $cluster_number \
                 --by_domain $by_domain \
                 --pruning_method $pruning_method \
                 --prune_rate $prune_rate
