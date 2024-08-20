model_path="/vepfs/hongcheng/moe/modelscope_cache/deepseek-ai/DeepSeek-V2-Lite"
dataset_dir="dataset"
output_dir="result/baseline_finetuning"
CUDA_VISIBLE_DEVICES="0,1" python baseline_finetuning.py \
                 --model_path $model_path \
                 --dataset_dir $dataset_dir \
                 --output_dir $output_dir