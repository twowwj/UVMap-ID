export OUTPUT_PATH="./3_20_test_faceid—nicu"
export PATH="/data/wjwang/anaconda3/envs/py3.9/bin:$PATH"
export CONTORL_MODEL="lllyasviel/control_v11p_sd15_seg"
export PROJECT_MODEL='./simplitex-trained-model-controlnet-faceid/'

CUDA_VISIBLE_DEVICES=5 python text2image_controlnet_faceid.py --guidance_scale 2 --inference_steps 75 \
    --pretrained_model_name_or_path "simplitex-trained-model-controlnet" \
    --faceid_model_path "simplitex-trained-model-controlnet-faceid" \
    --controlnet_model_name_or_path=$CONTORL_MODEL \
    --faceid_model_path=$PROJECT_MODEL \
    --instance_prompt "a sks texturemap" \
    --output_path=$OUTPUT_PATH \
    --validation_images='/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set' \
    --validation_image_embeds='/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set_emb' \
    --validation_prompt="a sks texturemap" \
    --validation_seg="/workspace/wwang/ControlNetDreamBooth-main/UV_APP/all.npy" \
    --checkpointing_steps="51000" \

# CUDA_VISIBLE_DEVICES=7 python text2image_controlnet.py --guidance_scale 2 --inference_steps 75 \
#     --prompt "a sks texturemap of $P wearing a black top and white trousers " \
#     --model_path "simplitex-trained-model-controlnet" \
#     --output_path=$OUTPUT_PATH \
#     --validation_image \
#     --validation_image_embed= \
#     --validation_prompt="a sks texturemap" \
#     --validation_seg="/workspace/wwang/ControlNetDreamBooth-main/UV_APP/all.npy" \

# done