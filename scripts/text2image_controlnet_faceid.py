import argparse
import os.path

from diffusers import StableDiffusionPipeline
from diffusers import ControlNetModel
from diffusers.utils import load_image
import numpy as np
from PIL import Image

import torch
from diffusers.models import (
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms
from diffusers import DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer, PretrainedConfig
from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available

import torch
from safetensors import safe_open
from safetensors.torch import save_file



if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

def _encode_prompt_image_emb(prompt_image_emb, image_proj_model, image_proj_model_in_features, device, num_images_per_prompt, dtype, do_classifier_free_guidance):
    if isinstance(prompt_image_emb, torch.Tensor):
        prompt_image_emb = prompt_image_emb.clone().detach()
    else:
        prompt_image_emb = torch.tensor(prompt_image_emb)
    bs = prompt_image_emb.shape[0]
    prompt_image_emb = prompt_image_emb.reshape([bs, -1, image_proj_model_in_features])
    
    if do_classifier_free_guidance:
        prompt_image_emb = torch.cat([torch.zeros_like(prompt_image_emb), prompt_image_emb], dim=0)
    else:
        prompt_image_emb = torch.cat([prompt_image_emb], dim=0)
    
    prompt_image_emb = prompt_image_emb.to(device=image_proj_model.latents.device, 
                                            dtype=image_proj_model.latents.dtype)
    prompt_image_emb = image_proj_model(prompt_image_emb)

    bs_embed, seq_len, _ = prompt_image_emb.shape
    prompt_image_emb = prompt_image_emb.repeat(1, num_images_per_prompt, 1)
    prompt_image_emb = prompt_image_emb.view(bs_embed * num_images_per_prompt, seq_len, -1)
    
    return prompt_image_emb.to(device=device, dtype=dtype)

def set_ip_adapter(unet, num_tokens=16, scale=0.5):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
        else:
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                cross_attention_dim=cross_attention_dim, 
                                                scale=scale,
                                                num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)
    unet.set_attn_processor(attn_procs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", default="simplitex-trained-model-controlnet", type=str, help="Path to the model to use.")
    parser.add_argument("--controlnet_model_name_or_path", default="", type=str, help="Path to the pretrained model to use.")
    parser.add_argument("--faceid_model_path", default="", type=str, help="Path to the faceid model to use.")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Value of guidance step")
    parser.add_argument("--inference_steps", type=int, default=100, help="Numver of inference steps")
    parser.add_argument("--instance_prompt", type=str, default="a sks texturemap of an astronaut",
                        help="Prompt to use. Use sks texture map as part of your prompt for best results")
    parser.add_argument("--output_file", type=str, default="output.png", help="File onto which to save the results.")
    parser.add_argument("--output_path", type=str, default="target", help="Directory which to save the results")

    parser.add_argument("--validation_prompt", type=str, default="target", help="Directory which to save the results")
    parser.add_argument("--validation_seg", type=str, default="target", help="Directory which to save the results")
    parser.add_argument("--validation_image_embeds", type=str, default="target", help="Directory which to save the results")
    parser.add_argument("--validation_images", type=str, default="target", help="Directory which to save the results")
    parser.add_argument("--checkpointing_steps", type=str, default="target", help="Directory which to save the results")
    # parser.add_argument("--")
    device="cuda"
    args = parser.parse_args()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)
    set_ip_adapter(unet)
    tensors={}

    ### load unet model###
    unet_model_path_finetuned = args.faceid_model_path \
                                + f"/checkpoint-{args.checkpointing_steps}" \
                                + "/unet" \
                                + "/diffusion_pytorch_model.safetensors"
    with safe_open(unet_model_path_finetuned, 
                   framework="pt",
                   device="cpu"
                   ) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    unet.load_state_dict(tensors)
    #####################

    tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
            device=device
        )

    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path).to(device)

    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=16,
        embedding_dim=512,
        output_dim=768,
        ff_mult=4,
    )
    # text_encoder(batch["input_ids"])
    image_proj_model_path = args.faceid_model_path \
                            + f"checkpoint-{args.checkpointing_steps}" \
                            + "/image_proj.bin"
    image_proj_model.load_state_dict(torch.load(image_proj_model_path))

    example_instance_prompt_ids = tokenizer(
        args.instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        ).input_ids.to(device=device)
    text_embeds = text_encoder(example_instance_prompt_ids)


    validation_seg = args.validation_seg
    controlnet_image = np.load(validation_seg)
    transform = transforms.ToTensor()
    controlnet_image = transform(controlnet_image).to(device)

    celebrites_names = os.listdir(args.validation_images)
    for celebrity_name in celebrites_names:
        celebrite_validation_images = os.path.join(args.validation_images, celebrity_name)
        validation_images = os.listdir(celebrite_validation_images)
        for validation_file in validation_images:
            with torch.no_grad():
                validation_image = os.path.join(celebrite_validation_images, validation_file)
                # validation_image.jpg,
                file_name = validation_file.split(".")[0]
                validation_image_embed = os.path.join(args.validation_image_embeds, f"{celebrity_name}/{file_name}.npy")
                print(validation_image_embed)
                if not (validation_image.endswith(".jpg") or validation_image.endswith(".png") or validation_image.endswith(".jpeg")):
                    continue
                validation_image = Image.open(validation_image)
                
                output_dir = os.path.join(args.output_path, celebrity_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                null_token = tokenizer( 
                        [""]*1,
                        truncation=True,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(device=device)
                null_hidden_states = text_encoder(null_token)
                encoder_hidden_states = torch.cat([null_hidden_states[0], text_embeds[0][0:1]], dim=0)

                validation_image_embed = torch.from_numpy(np.load(validation_image_embed)).unsqueeze(0)

                shape = (1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size)
                latents = randn_tensor(shape).to(device=device)

                prompt_image_emb = _encode_prompt_image_emb(validation_image_embed, 
                                    image_proj_model, 
                                    512, 
                                    device, 
                                    1, 
                                    torch.float16, 
                                    True)
                prompt_embeds = torch.cat([encoder_hidden_states, prompt_image_emb], dim=1)

                ref_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                num_inference_steps = 50
                guidance_scale = 2.5
                ref_scheduler.set_timesteps(num_inference_steps)
                timesteps = ref_scheduler.timesteps

                for i, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond, noise_pred= noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                    latents = ref_scheduler.step(noise_pred, t, latents).prev_sample
            latents_ = latents.detach().clone()

            image = vae.decode(latents_ / 0.18215, return_dict=False)[0]
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
            generated_image = image_processor.postprocess(image.detach().cpu(), output_type="pil")
            generated_image[0].save(f'{output_dir}/pred_{validation_file}.png')
            validation_image.save(f'{output_dir}/target_{validation_file}.png')
            # train_image_ = image_processor.postprocess(train_image.detach().cpu(), output_type="pil")
            # train_image_[0].save(f'{args.output_dir}/train{step:04d}.png')
            # gt_image_ = image_processor.postprocess(gt_image.detach().cpu(), output_type="pil")
            # gt_image_[0].save(f'{args.output_dir}/gt{step:04d}.png')