#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/wjwang/anaconda3/envs/py3.9/bin:$PATH"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./data_train"
export CLASS_DIR="class_dir"
export OUTPUT_DIR="./"



accelerate launch --mixed_precision="fp16" train_dreambooth.py --pretrained_model_name_or_path=$MODEL_NAME  \
        --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR --class_data_dir=$CLASS_DIR \
        --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="a sks texturemap"  \
        --class_prompt="a texturemap" --resolution=512 --train_batch_size=1 \
        --gradient_accumulation_steps=2 --gradient_checkpointing --learning_rate=1e-6 \
        --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=10 --max_train_steps=1500 \
        --checkpointing_steps=1500 --train_text_encoder --use_8bit_adam \
        --validation_prompt="a sks texturemap" --images_steps 10

        eval "$(conda shell.bash hook)"
bash
conda activate py3.9