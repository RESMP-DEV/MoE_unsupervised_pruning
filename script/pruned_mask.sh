model_path="/vepfs/hongcheng/moe/modelscope_cache/deepseek-ai/DeepSeek-V2-Lite"
dataset_dir="dataset"
dataset_name_list="MathInstruct,finance_alpaca,code_alpaca_20k"
CUDA_VISIBLE_DEVICES="0,1" python pruned_mask.py \
                 --model_path $model_path \
                 --dataset_dir $dataset_dir \
                 --dataset_name_list dataset_name_list