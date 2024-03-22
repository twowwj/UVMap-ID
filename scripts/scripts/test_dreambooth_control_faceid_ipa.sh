export OUTPUT_PATH="./3_20_test_faceid—nicu"
export PATH="/data/wjwang/anaconda3/envs/py3.9/bin:$PATH"
export CONTORL_MODEL="lllyasviel/control_v11p_sd15_seg"
export PROJECT_MODEL='./simplitex-trained-model-controlnet-faceid/'

CUDA_VISIBLE_DEVICES=5 python test_ckpt.py --guidance_scale 2 --inference_steps 75 \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --controlnet_model_name_or_path=$CONTORL_MODEL \
    --faceid_model_path=$PROJECT_MODEL \
    --instance_prompt "a professional photograph of a woman with red and very short hair" \
    --output_path=$OUTPUT_PATH \
    --validation_images='/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set' \
    --validation_image_embeds='/workspace/wwang/ControlNetDreamBooth-main/UV_APP/test_set_emb' \
