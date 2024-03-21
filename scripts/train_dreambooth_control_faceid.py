import argparse
import hashlib
import itertools
import math
import os
import warnings
from pathlib import Path
from typing import Optional
import PIL.Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, \
            UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler, DDIMScheduler
from transformers import CLIPImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import numpy as np
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.image_processor import VaeImageProcessor
import random
if is_wandb_available():
    import wandb

### 
from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)

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
    

    # state_dict = torch.load(model_ckpt, map_location="cpu")
    # ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
    # if 'ip_adapter' in state_dict:
    #     state_dict = state_dict['ip_adapter']
    # ip_layers.load_state_dict(state_dict)



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



def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel

    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def log_validationv2(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, 
                     step, scheduler, feature_extractor, image_proj_model, text_embeds, train_image, gt_image, flag_):
    with torch.no_grad():
        logger.info("Running validation... ")
        vae = accelerator.unwrap_model(vae)
        text_encoder = accelerator.unwrap_model(text_encoder)
        unet = accelerator.unwrap_model(unet)
        controlnet = accelerator.unwrap_model(controlnet)

        validation_image = args.validation_image[0]
        validation_seg = args.validation_seg[0]
        validation_image_embed = args.validation_image_embed[0]

        validation_image = Image.open(validation_image)

        controlnet_image = np.load(validation_seg)
        transform = transforms.ToTensor()
        controlnet_image = transform(controlnet_image).to(accelerator.device)

        null_token = tokenizer( 
                [""]*1,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device=accelerator.device)
        null_hidden_states = text_encoder(null_token)
        encoder_hidden_states = torch.cat([null_hidden_states[0], text_embeds[0][0:1]], dim=0)

        validation_image_embed = torch.from_numpy(np.load(validation_image_embed)).unsqueeze(0)

        shape = (1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size)
        latents = randn_tensor(shape).to(device=accelerator.device)

        prompt_image_emb = _encode_prompt_image_emb(validation_image_embed, 
                            image_proj_model, 
                            512, 
                            accelerator.device, 
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
        generated_image[0].save(f'{args.output_dir}/pred_step{step:04d}_{flag_}.png')
        validation_image.save(f'{args.output_dir}/target{step:04d}.png')
        train_image_ = image_processor.postprocess(train_image.detach().cpu(), output_type="pil")
        train_image_[0].save(f'{args.output_dir}/train{step:04d}.png')
        gt_image_ = image_processor.postprocess(gt_image.detach().cpu(), output_type="pil")
        gt_image_[0].save(f'{args.output_dir}/gt{step:04d}.png')

def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, scheduler, feature_extractor, image_proj_model, text_embeds):
    logger.info("Running validation... ")
    vae = accelerator.unwrap_model(vae)
    text_encoder = accelerator.unwrap_model(text_encoder)
    unet = accelerator.unwrap_model(unet)
    controlnet = accelerator.unwrap_model(controlnet)
    # controlnet = ControlNetModel.from_pretrained(
    #     args.controlnet_model_name_or_path, torch_dtype=torch.float16
    # ).to("cuda")

        # vae: AutoencoderKL,
        # text_encoder: CLIPTextModel,
        # tokenizer: CLIPTokenizer,
        # unet: UNet2DConditionModel,
        # controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        # scheduler: KarrasDiffusionSchedulers,
        # safety_checker: StableDiffusionSafetyChecker,
        # feature_extractor: CLIPImageProcessor,
        # requires_safety_checker: bool = True,
    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet = controlnet,
        scheduler = scheduler,
        safety_checker=None,
        feature_extractor=feature_extractor,
        requires_safety_checker=False,
    )



    # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # if len(args.validation_image) == len(args.validation_prompt):
    #     validation_images = args.validation_image
    #     validation_prompts = args.validation_prompt
    # elif len(args.validation_image) == 1:
    validation_image = args.validation_image[0]
    validation_prompt = args.validation_prompt[0]
    validation_seg = args.validation_seg[0]
    validation_image_embed = args.validation_image_embed[0]

    validation_image = Image.open(validation_image)
    controlnet_image = np.load(validation_seg)
    validation_image_embed = torch.from_numpy(np.load(validation_image_embed)).unsqueeze(0)

    prompt_image_emb = _encode_prompt_image_emb(validation_image_embed, 
                        image_proj_model, 
                        512, 
                        accelerator.device, 
                        1, 
                        torch.float16, 
                        False)

    prompt_embeds = torch.cat([text_embeds[0:1,:,:], prompt_image_emb], dim=1)
    # np to PIL
    controlnet_image = Image.fromarray(controlnet_image)

    with torch.autocast("cuda"):
        image = pipeline(
            prompt=None, 
            image=controlnet_image, 
            prompt_embeds=prompt_embeds,
            num_inference_steps=50, 
            generator=generator, 
            guidance_scale=2.5, 
            height=512, 
            width=512
        ).images[0]

    image.save(f'{args.output_dir}/pred_step{step:04d}.png')
    validation_image.save(f'{args.output_dir}/target{step:04d}.png')

def parse_args(input_args=None):

    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--instance_prompt2",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )

    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )

    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a sks texturemap",
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_image_embed",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_seg",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )

    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")

    parser.add_argument("--train_controlnet", action="store_true", help="Whether to train the text encoder")

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--images_steps",
        type=int,
        default=30,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )

    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")

    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        instance_prompt2,
        tokenizer,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.original_image_path = Path(instance_data_root + '/ori_img')
        self.uv_image_path = Path(instance_data_root + '/UVtexture')
        self.uv_images = list(self.uv_image_path.glob('*/*.png'))
        self.instance_prompt = instance_prompt
        self.instance_prompt2=instance_prompt2
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.np_transforms = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    def get_random_image(self, target_name):
        image_path = self.instance_data_root/ 'ori_img' / target_name
        embed_path = self.instance_data_root/ 'ori_img_aligned_emb' / target_name
        # get random image form folder
        files = os.listdir(embed_path)
        embed_files = [file for file in files if file.lower().endswith(('.npy'))]

        # 从图片列表中随机选择一个文件
        selected_embed = random.choice(embed_files) if embed_files else None
        # 同时选出对应的embed文件
        selected_image = selected_embed.replace('.npy', '.jpg')
        # 确保embed文件和selected_image 同时存在
        # if not os.path.exists(embed_path / selected_embed):
            # raise ValueError(f"Embed file {selected_embed} not found for image {selected_image} in {embed_path}")
        # open the image image_path / selected_image
        # Convert instance_image to RGB if it is not already
        instance_image = Image.open(image_path / selected_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        # Resize instance_image to (3, 512, 512)
        instance_image = instance_image.resize((512, 512))

        return instance_image, np.load(embed_path / selected_embed)


    def __len__(self):
        return len(self.uv_images)

    def __getitem__(self, index):

        example = {}
        image_path = self.uv_images[index]
        target_name = image_path.parent.name
        instance_image = Image.open(self.uv_images[index])
        app_image, app_image_face_embed = self.get_random_image(target_name)
        seg_image = np.load('../UV_APP/all.npy')
        image_name = image_path.name.split(',')[0]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        example['conditioning_pixel_values'] = self.np_transforms(seg_image)
        example["apparence_images"] = self.image_transforms(app_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["instance_prompt_ids2"] = self.tokenizer(
            self.instance_prompt2,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["apparence_images_face_embeds"] = app_image_face_embed

        return example

def collate_fn(examples):

    input_ids = [example["instance_prompt_ids"] for example in examples]
    input_ids2 = [example["instance_prompt_ids2"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    apparence_images = [example["apparence_images"] for example in examples]
    conditionl_pixel_values = [example["conditioning_pixel_values"] for example in examples]
    apparence_images_face_embeds = [example["apparence_images_face_embeds"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    apparence_images = torch.stack(apparence_images)
    apparence_images = apparence_images.to(memory_format=torch.contiguous_format).float()
    conditionl_pixel_values = torch.stack(conditionl_pixel_values)
    conditionl_pixel_values = conditionl_pixel_values.to(memory_format=torch.contiguous_format).float()


    apparence_images_face_embeds = np.stack(apparence_images_face_embeds)
    apparence_images_face_embeds = apparence_images_face_embeds.astype(np.float32)
    
    input_ids = torch.cat(input_ids, dim=0)
    input_ids2 = torch.cat(input_ids2, dim=0)

    batch = {
        "input_ids": input_ids,
        "input_ids2": input_ids2,
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditionl_pixel_values,
        'apparence_images': apparence_images,
        "apparence_images_face_embeds": apparence_images_face_embeds
    }

    return batch

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load models and create wrapper for stable diffusion
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    set_ip_adapter(unet)

    ### projection layer 
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
    image_proj_model = image_proj_model.to(accelerator.device)

    state_dict = torch.load(f'/workspace/wwang/InstantID/checkpoints/ip-adapter.bin', map_location="cpu")
    if 'image_proj' in state_dict:
        state_dict = state_dict["image_proj"]
    # image_proj_model.load_state_dict(state_dict)
    image_proj_model_in_features = 512
    do_classifier_free_guidance = False

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    vae.requires_grad_(False)

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    
    attn1_params = [params for name, params in unet.named_parameters() if 'attn1.processor' in name]
    attn2_params = [params for name, params in unet.named_parameters() if 'attn2.processor' in name]
    params_to_optimize = (
        itertools.chain(
            image_proj_model.parameters(),
            attn1_params+attn2_params,
                        )
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        instance_prompt2=args.instance_prompt2,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler, controlnet = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler, controlnet
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler, controlnet = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler, controlnet
        )


    
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    

    for epoch in range(first_epoch, args.num_train_epochs):

        # unet.train()
        image_proj_model.train()

        if args.train_text_encoder:
            text_encoder.train()

        if args.train_controlnet:
            controlnet.train()

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                # 将面部（120, 120, 120）置为1
                img_app = batch['apparence_images']
                img_app_face_embed = batch['apparence_images_face_embeds']
                # for i in range(img_app_face_embed.shape[0]):
                prompt_image_emb = _encode_prompt_image_emb(img_app_face_embed, 
                                        image_proj_model, 
                                        image_proj_model_in_features, 
                                        accelerator.device, 
                                        1, 
                                        torch.float16, 
                                        do_classifier_free_guidance)
                    # prompt_image_embs.append(prompt_image_emb[0])
                # prompt_image_emb = torch.stack(prompt_image_embs)
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # text_encoder(batch["input_ids"])[0] shape: 
                # print(text_encoder(batch["input_ids"])[0].shape, prompt_image_emb.shape)
                encoder_hidden_states = torch.cat((text_encoder(batch["input_ids"])[0], prompt_image_emb), dim=1)

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                #model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(
                            attn1_params+attn2_params,
                            image_proj_model.parameters(),
                                        )
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unet = accelerator.unwrap_model(unet)
                        if not os.path.exists(os.path.join(args.output_dir, f"checkpoint-{global_step}")):
                            os.makedirs(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                        if not os.path.exists(os.path.join(args.output_dir, f"checkpoint-{global_step}", "unet")):
                            os.makedirs(os.path.join(args.output_dir, f"checkpoint-{global_step}", "unet"))
                        unet.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{global_step}", "unet"))
                        
                        image_proj_model = accelerator.unwrap_model(image_proj_model)
                        torch.save(image_proj_model.state_dict(), os.path.join(args.output_dir, f"checkpoint-{global_step}", "image_proj.bin"))

                if global_step % args.images_steps == 0:
                    log_validationv2(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        controlnet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                        scheduler,
                        feature_extractor,
                        image_proj_model,
                        text_encoder(batch["input_ids"]),
                        batch["pixel_values"],
                        img_app,
                        "a sks  "
                    )

                    log_validationv2(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        controlnet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                        scheduler,
                        feature_extractor,
                        image_proj_model,
                        text_encoder(batch["input_ids2"]),
                        batch["pixel_values"],
                        img_app,
                        "a sks texturemap wearing red clothes"
                    )
                    # log_validation(
                    #     vae,
                    #     text_encoder,
                    #     tokenizer,
                    #     unet,
                    #     controlnet,
                    #     args,
                    #     accelerator,
                    #     weight_dtype,
                    #     global_step,
                    #     scheduler,
                    #     feature_extractor,
                    #     image_proj_model,
                    #     text_encoder(batch["input_ids"])[0]
                    # )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:

        controlnet = accelerator.unwrap_model(controlnet)
        # controlnet.save_pretrained(args.output_dir)

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            controlnet=controlnet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    # Load face encoder
    args = parse_args()
    main(args)