#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

# export PATH="/data/jzhang/conda/anaconda3/bin:$PATH"
# export OUTPUT_PATH="./output-ldm-dreambooth"

# eval "$(conda shell.bash hook)"
# bash
# conda activate smplitex

# python text2image.py --guidance_scale 2 --inference_steps 20 \
#     --prompt "a sks texturemap of Elon Musk wearing a black top and white trousers" \
#     --model_path "simplitex-trained-model" \
#     --output_path=$OUTPUT_PATH


# CUDA_VISIBLE_DEVICES=5 python text2image.py --guidance_scale 2 --inference_steps 75 \
#     --prompt "a sks texturemap of Bill Gates wearing a black top and white trousers" \
#     --model_path "simplitex-trained-model" \
#     --output_path="./output-ldm-dreambooth" 


celebrities=(
    'Yann LeCun',    
)

# Loop through each celebrity in the list
for celebrity in "${celebrities[@]}"
do
    # Execute the python command with the current celebrity name in the prompt
    CUDA_VISIBLE_DEVICES=6 python text2image.py --guidance_scale 2 --inference_steps 75 \
    --prompt "a sks texturemap of ${celebrity}" \
    --model_path "simplitex-trained-model" \
    --output_path="./datasets"
done

