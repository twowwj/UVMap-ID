#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/wjwang/anaconda3/envs/py3.9/bin:$PATH"
export MODEL_NAME="/workspace/wwang/ControlNetDreamBooth-main/model/Realistic_Vision_V4.0_noVAE"
export INSTANCE_DIR="/workspace/wwang/ControlNetDreamBooth-main/UV_APP"
export CLASS_DIR="class_dir"
export OUTPUT_DIR="./simplitex-trained-model-ipa-lora"

export VAE_MODEL_PATH="/workspace/wwang/ControlNetDreamBooth-main/model/sd-vae-ft-mse"
export IP_CKPT_PATH="/workspace/wwang/ControlNetDreamBooth-main/ip_adapter/models/ip-adapter-faceid_sd15.bin"

CUDA_VISIBLE_DEVICES=7 python train_dreambooth_ipa_lora_1.py --pretrained_model_name_or_path=$MODEL_NAME  \
        --ip_ckpt=$IP_CKPT_PATH --vae_model_name_or_path=$VAE_MODEL_PATH \
        --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR --class_data_dir=$CLASS_DIR \
        --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="a sks texturemap"  \
        --class_prompt="a texturemap" --resolution=512 --train_batch_size=1 \
        --gradient_accumulation_steps=2 --gradient_checkpointing --learning_rate=1e-6 \
        --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=10 --max_train_steps=15000 \
        --checkpointing_steps=1500 --train_text_encoder --use_8bit_adam \
        --validation_prompt="a sks texturemap" --images_steps 100 \
        --validation_image "/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set/Yann LeCun/1.jpg" \
        --validation_image_embed "/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set/Yann LeCun/1.npy" \
#         eval "$(conda shell.bash hook)"
# bash
# conda activate py3.9