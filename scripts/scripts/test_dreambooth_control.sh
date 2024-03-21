#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

# export CONTORL_MODEL="lllyasviel/control_v11p_sd15_seg"
# export OUTPUT_PATH="./23_02_2024_results_dreambooth_control_1"
export OUTPUT_PATH="./datasets_controlnet_celebrities"

export PATH="/data/wjwang/anaconda3/envs/py3.9/bin:$PATH"
# eval "$(conda shell.bash hook)"
# bash
# conda activate smplitex

# celebrities=(
#     steve jobs,
#     Leonardo DiCaprio,
#     Barack Obama,
#     Justin Bieber,
#     Bob Dylan,
#     "Bill Gates",
# )

# for celebrity in "${celebrities[@]}"
# do
#     echo $celebrity
#     CUDA_VISIBLE_DEVICES=7 python text2image_controlnet.py --guidance_scale 2 --inference_steps 75 \
#         --prompt "a sks texturemap of ${celebrity}" \
#         --model_path "simplitex-trained-model-controlnet" \
#         --output_path=$OUTPUT_PATH
# done


P='Kaiming He'
echo $P
CUDA_VISIBLE_DEVICES=7 python text2image_controlnet.py --guidance_scale 2 --inference_steps 75 \
    --prompt "a sks texturemap of $P wearing a black top and white trousers " \
    --model_path "simplitex-trained-model-controlnet" \
    --output_path=$OUTPUT_PATH
