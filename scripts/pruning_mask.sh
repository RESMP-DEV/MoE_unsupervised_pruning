model_path="/vepfs/hongcheng/moe/modelscope_cache/deepseek-ai/DeepSeek-V2-Lite"
dataset_dir="dataset"
#dataset_name_list="c4"
dataset_name_list="MathInstruct,code_alpaca_20k,finance_alpaca"
batch_size=32
sample_number=32
cluster_number=12 # only available when using uns method
by_domain=0 # only available when using uns method
pruning_method="kmeans_prune"
#pruning_method="hierarchical_prune"
#pruning_method="seer_prune"
#pruning_method="hsic_prune"
prune_rate=0.2
CUDA_VISIBLE_DEVICES="2,3" python pruning_mask.py \
                 --model_path $model_path \
                 --dataset_dir $dataset_dir \
                 --dataset_name_list $dataset_name_list \
                 --batch_size $batch_size \
                 --sample_number $sample_number \
                 --cluster_number $cluster_number \
                 --by_domain $by_domain \
                 --pruning_method $pruning_method \
                 --prune_rate $prune_rate
