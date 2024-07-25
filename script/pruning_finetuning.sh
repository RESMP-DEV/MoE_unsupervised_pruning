model_path="/home/zhongyuan_peng/.cache/modelscope/hub/deepseek-ai/DeepSeek-V2-Lite"
dataset_dir="dataset"
output_dir="result/pruning_finetuning"
CUDA_VISIBLE_DEVICES="4,5,6,7" python pruning_finetuning.py \
                 --model_path $model_path \
                 --dataset_dir $dataset_dir \
                 --output_dir $output_dir