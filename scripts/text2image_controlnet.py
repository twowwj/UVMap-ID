import argparse
import os.path

from diffusers import StableDiffusionPipeline
from diffusers import ControlNetModel, AutoPipelineForText2Image, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
from PIL import Image

import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--controlmodel_path", default="simplitex-trained-model", type=str, help="Path to the model to use.")
    parser.add_argument("--model_path", default="simplitex-trained-model", type=str, help="Path to the model to use.")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Value of guidance step")
    parser.add_argument("--inference_steps", type=int, default=100, help="Numver of inference steps")
    parser.add_argument("--prompt", type=str, default="a sks texturemap of an astronaut",
                        help="Prompt to use. Use sks texture map as part of your prompt for best results")
    parser.add_argument("--output_file", type=str, default="output.png", help="File onto which to save the results.")
    parser.add_argument("--output_path", type=str, default="target", help="Directory which to save the results")

    args = parser.parse_args()

    assert args.guidance_scale >= 0., "Invalid guidance scale value"
    assert args.inference_steps > 0, "Invalid inference steps number"
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
        
    pose_image_ = np.load("./data_train/02339-1630119034-sks texturemap, ((Marco Reus)).jpg.npy")

    pose_image_ = Image.fromarray(pose_image_.astype(np.uint8))

    pose_image_.save(os.path.join(args.output_path, "pose.png"))

    # pipeline = AutoPipelineForText2Image.from_pretrained(
    #     args.model_path, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
    # ).to("cuda")

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
    ).to("cuda")

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()


    for i in range(40):

        image = pipeline(args.prompt, pose_image_, guidance_scale=args.guidance_scale, num_inference_steps=args.inference_steps).images[0]
        image.save(os.path.join(args.output_path, args.prompt + "{}.png".format(i)))

