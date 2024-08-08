import torch
from typing import List
from tqdm import tqdm
import numpy as np
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.models import (
    UNet2DConditionModel,
    AutoencoderKL,
)
import os
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
import argparse
from ip_adapter.ip_adapter_faceid import MLPProjModel
from safetensors import safe_open
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor_faceid import LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor, LoRAAttnProcessor2_0 as LoRAAttnProcessor
else:
    from ip_adapter.attention_processor_faceid import IPAttnProcessor, AttnProcessor

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(2021111)

def get_image_embeds(faceid_embeds):

    return image_prompt_embeds, uncond_image_prompt_embeds


def progress_bar(iterable=None, total=None):
        if iterable is not None:
            return tqdm(iterable)
        elif total is not None:
            return tqdm(total=total)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")
        
def encode_prompt(prompt, tokenizer, text_encoder, device, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = text_encoder(
            text_input_ids.to(device),
        )
        text_embeddings = text_embeddings[0]

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_embeddings = text_encoder(
                uncond_input.input_ids.to(device),
            )
            uncond_embeddings = uncond_embeddings[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            return text_embeddings, uncond_embeddings
            # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

def set_ip_adapter(unet, num_tokens=4, scale=0.5):
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
            attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=128,
            ).to(unet.device, dtype=unet.dtype)
        else:

            attn_procs[name] = LoRAIPAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0,rank=128,
                num_tokens=num_tokens,
            ).to(unet.device, dtype=unet.dtype)

    unet.set_attn_processor(attn_procs)

    tensors={}
    unet_model_path_finetuned = args.pretrained_model_name_or_path \
                                + f"/checkpoint-{args.resume_ckpt}" \
                                + "/unet" \
                                + "/diffusion_pytorch_model.safetensors"
    with safe_open(unet_model_path_finetuned, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    unet.load_state_dict(tensors)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", default="runwayml/stable-diffusion-v1-5", type=str, help="Path to the model to use.")
    parser.add_argument("--vae_model_name_or_path", default="runwayml/stable-diffusion-v1-5", type=str, help="Path to the model to use.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Value of guidance step")
    parser.add_argument("--instance_prompt", type=str, default="an Asian woman ",
                        help="Prompt to use. Use sks texture map as part of your prompt for best results")
    parser.add_argument("--negative_instance_prompt", type=str, default="monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
                        help="Prompt to use. Use sks texture map as part of your prompt for best results")
    parser.add_argument("--output_path", type=str, default="target", help="Directory which to save the results")
    parser.add_argument("--validation_image_embeds", type=str, default="target", help="Directory which to save the results")
    parser.add_argument("--validation_images", type=str, default="target", help="Directory which to save the results")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--resume_ckpt", type=int, default=1, help="Number of inference steps")
    parser.add_argument("--revision",type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    # parser.add_argument("--")
    device="cuda"
    args = parser.parse_args()
    print("current prompt is:", args.instance_prompt)
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path, revision=args.revision).to(device)
    #unet = UNet2DConditionModel.from_pretrained('/home/jichao.zhang/code/UVMap-ID/Realistic_Vision_V4.0_noVAE', subfolder="unet", revision=args.revision,).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,).to(device)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    clip_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    image_proj_model = MLPProjModel(
    cross_attention_dim=768,
    id_embeddings_dim=512,
    num_tokens=4,
    )
    image_proj_model = image_proj_model.to(device)
    image_proj_model_ckpt_path = os.path.join(args.pretrained_model_name_or_path, f"checkpoint-{args.resume_ckpt}",'image_proj_model.ckpt')
    image_proj_model.load_state_dict(torch.load(image_proj_model_ckpt_path))
    # print(image_proj_model.keys())
    # exit()
    set_ip_adapter(unet)

    faceid_embeds = torch.from_numpy(np.load(args.validation_image_embeds)).unsqueeze(0)

    do_classifier_free_guidance = True
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps

    # Prepare latent variables
    num_channels_latents = unet.config.in_channels  
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    batch_size = 1
    if do_classifier_free_guidance:
        shape = (batch_size, num_channels_latents, 512 // vae_scale_factor, 512 // vae_scale_factor)
    else:
        shape = (batch_size * 2, num_channels_latents, 512 // vae_scale_factor, 512 // vae_scale_factor)


    prompt = [args.instance_prompt] * batch_size
    n_prompt = [args.negative_instance_prompt] * batch_size

    text_embeddings, uncond_embeddings = encode_prompt(
        prompt, tokenizer, text_encoder, device, do_classifier_free_guidance, n_prompt
    )

    faceid_embeds = faceid_embeds.to(device, dtype=unet.dtype)
    image_prompt_embeds = image_proj_model(faceid_embeds)
    uncond_image_prompt_embeds = image_proj_model(torch.zeros_like(faceid_embeds))

    num_samples = 1
    bs_embed, seq_len, _ = image_prompt_embeds.shape
    image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
    image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
    prompt_embeds = torch.cat([text_embeddings, image_prompt_embeds], dim=1)

    negative_prompt_embeds = torch.cat([uncond_embeddings, uncond_image_prompt_embeds], dim=1)
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    num_warmup_steps = len(timesteps) - args.num_inference_steps * noise_scheduler.order
    with torch.no_grad():
        for number in range(100):
            latents = randn_tensor(shape, device=device)
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.to(dtype=unet.dtype)
            
                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_ = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_ - noise_pred_uncond)
                
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample 

            # Post-processing
            image_output = vae.decode(latents / 0.18215, return_dict=False)[0].detach().cpu()
            generation = clip_image_processor.postprocess(image_output, output_type="pil")[0]
            name = args.validation_images.split('/')[-2]
            output_dir = os.path.join(args.output_path, name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            text_prompt = args.instance_prompt.split('of')[-1]
            generation.save(os.path.join(output_dir,f'{name}_{number}_{text_prompt}.png'))