export MODEL_NAME="simplitex-trained-model-controlnet"
export CONTORL_MODEL="lllyasviel/control_v11p_sd15_seg"
export INSTANCE_DIR="/workspace/wwang/ControlNetDreamBooth-main/UV_APP"
export CLASS_DIR="class_dir_control"
export OUTPUT_DIR="./simplitex-trained-model-controlnet-faceid-ipa-loar_3.26"

# CUDA_VISIBLE_DEVICES=7 python train_dreambooth_control_faceid.py --pretrained_model_name_or_path=$MODEL_NAME  \
#         --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR --class_data_dir=$CLASS_DIR \
#         --instance_prompt="a sks texturemap"  \
#         --class_prompt="a texturemap" --resolution=512 --train_batch_size=8 \
#         --gradient_accumulation_steps=1 --gradient_checkpointing --learning_rate=1e-5 \
#         --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=10 --max_train_steps=30000 \
#         --checkpointing_steps=3000 --train_text_encoder --use_8bit_adam --controlnet_model_name_or_path=$CONTORL_MODEL \
#         --validation_image "/workspace/wwang/ControlNetDreamBooth-main/UV_APP/ori_img/Albert Einstein/7.jpg" \
#         --validation_image_embed "/workspace/wwang/ControlNetDreamBooth-main/UV_APP/ori_img_aligned_emb/Albert Einstein/7.npy" \
#         --images_steps 50 \
#         --validation_prompt="a sks texturemap" \
#         --validation_seg="/workspace/wwang/ControlNetDreamBooth-main/UV_APP/all.npy" \

CUDA_VISIBLE_DEVICES=6 python train_dreambooth_control_faceid_ipa_lora.py --pretrained_model_name_or_path=$MODEL_NAME  \
        --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR --class_data_dir=$CLASS_DIR \
        --instance_prompt="a sks texturemap"  \
        --instance_prompt2="a sks texturemap wearing red clothes"  \
        --class_prompt="a texturemap" --resolution=512 --train_batch_size=8 \
        --gradient_accumulation_steps=1 --gradient_checkpointing --learning_rate=1e-5 \
        --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=10 --max_train_steps=60000 \
        --checkpointing_steps=3000 --train_text_encoder --use_8bit_adam --controlnet_model_name_or_path=$CONTORL_MODEL \
        --validation_image "/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set/Yann LeCun/1.jpg" \
        --validation_image_embed "/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set/Yann LeCun/1.npy" \
        --images_steps 100 \
        --validation_prompt="a sks texturemap" \
        --validation_seg="/workspace/wwang/ControlNetDreamBooth-main/UV_APP/all.npy" \





