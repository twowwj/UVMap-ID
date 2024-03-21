export MODEL_NAME="simplitex-trained-model-controlnet"
export CONTORL_MODEL="lllyasviel/control_v11p_sd15_seg"
export INSTANCE_DIR="/workspace/wwang/ControlNetDreamBooth-main/UV_APP"
export CLASS_DIR="class_dir_control"
export OUTPUT_DIR="./simplitex-trained-model-controlnet-app_unet"

# accelerate launch --mixed_precision="fp16" train_dreambooth_control.py --pretrained_model_name_or_path=$MODEL_NAME  \
#         --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR --class_data_dir=$CLASS_DIR \
#         --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="a sks texturemap"  \
#         --class_prompt="a texturemap" --resolution=512 --train_batch_size=1 \
#         --gradient_accumulation_steps=2 --gradient_checkpointing --learning_rate=1e-6 \
#         --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=10 --max_train_steps=3000 \
#         --checkpointing_steps=3000 --train_text_encoder --use_8bit_adam --controlnet_model_name_or_path=$CONTORL_MODEL \
#         --validation_image "./data_train/02339-1630119034-sks texturemap, ((Marco Reus)).jpg.npy"

CUDA_VISIBLE_DEVICES=7 python train_dreambooth_control_app.py --pretrained_model_name_or_path=$MODEL_NAME  \
        --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR --class_data_dir=$CLASS_DIR \
        --instance_prompt="a sks texturemap"  \
        --class_prompt="a texturemap" --resolution=512 --train_batch_size=8 \
        --gradient_accumulation_steps=1 --gradient_checkpointing --learning_rate=1e-5 \
        --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=10 --max_train_steps=30000 \
        --checkpointing_steps=3000 --train_text_encoder --use_8bit_adam --controlnet_model_name_or_path=$CONTORL_MODEL \
        --validation_image "/workspace/wwang/ControlNetDreamBooth-main/UV_APP/ori_img_aligned/Albert Einstein/37.jpg" \
        --images_steps 50 \
        --validation_prompt="a sks texturemap" \
        --validation_seg="/workspace/wwang/ControlNetDreamBooth-main/UV_APP/all.npy" \




