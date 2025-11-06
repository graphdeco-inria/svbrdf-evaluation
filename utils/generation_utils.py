import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch
import numpy as np
import torchvision
from PIL import Image
from omegaconf import OmegaConf

from archs.estim_utils import output_inference_images
from archs.hyperfeat_extractor import HyperFeatureExtractor, HyperFeatureExtractorControlNet
from torchvision.transforms.functional import rgb_to_grayscale

def init_module(ckpt_folder, ckpt_name, conf_name, device, batch_size=1, seed=None, model_list=None):
    config_path = os.path.join(ckpt_folder, conf_name)
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config["ckpt_path"] = os.path.join(ckpt_folder, ckpt_name)
    config["batch_size"] = batch_size
    if seed is not None:
        config["seed"] = seed
    if os.path.exists(config["ckpt_path"]):
        # Generation
        if model_list is not None:
            config["model_list"] = model_list
            config["diffusion_mode"] = "generation"
            checkpoint = torch.load(config["ckpt_path"], map_location="cpu")
            pl_extractor = HyperFeatureExtractorControlNet(config=config, device=device)
            pl_extractor.load_state_dict(state_dict=checkpoint, strict=False)
            pl_extractor = pl_extractor.to(device)
            del checkpoint
        # Estimation
        else:
            checkpoint = torch.load(config["ckpt_path"], map_location="cpu")
            pl_extractor = HyperFeatureExtractor(config=config, device=device)
            pl_extractor.load_state_dict(state_dict=checkpoint, strict=False)
            pl_extractor = pl_extractor.to(device)
            del checkpoint
    else:
        print("could not find:", config["ckpt_path"])
        exit(0)
    return pl_extractor

def load_normalized_depth(filepath, blender_depth=False):
    depth = cv2.imread(filepath, -1)
    if depth is None:
        raise ValueError(f"ERROR: Open {filepath} failed!")
    if depth.ndim == 2:
        depth = depth[:,:,np.newaxis]
    depth = np.transpose(depth, [2, 0, 1])
    depth = torch.from_numpy(depth)
    depth = process_depth_normalize(depth)[None, :, :, :]
    if blender_depth:
        depth = depth[:, 0:1] #3/4 channels depth in Blender
    return depth

def load_lineart(filepath):
    image_lineart = cv2.imread(filepath, -1).astype(np.float32)[..., 3:4]
    image_lineart = torch.from_numpy(image_lineart)
    image_lineart = image_lineart.permute(2, 0, 1).repeat(3, 1, 1)[None, :, :, :] / 255.0
    return image_lineart

def process_depth_normalize(depth):
    depth[~torch.isfinite(depth)] = 0
    depth /= torch.clamp(torch.max(depth), min=1e-6) # Normalize depth
    return depth

def load_normals(filepath):
    normals = cv2.imread(filepath, -1)[..., ::-1].copy()
    if normals is None:
        raise ValueError(f"ERROR: Open {filepath} failed!")
    normals = torch.from_numpy(normals).permute(2, 0, 1)[None]
    return normals

def generate_image(
        pl_extractor,
        prompt, 
        negative_prompt, 
        guidance_scale,
        conditionning = None,
        control_image = None,
        seed = 0) -> Image:

    device = pl_extractor.device
    pl_extractor.diffusion_extractor.generator = torch.Generator(device=device).manual_seed(seed)

    if conditionning is None:
        pl_extractor.diffusion_extractor.prompt = prompt
        pl_extractor.diffusion_extractor.negative_prompt = negative_prompt
        pl_extractor.diffusion_extractor.init_extractor(pl_extractor.device)
    else:
        pl_extractor.diffusion_extractor.prompt = ""
        pl_extractor.diffusion_extractor.negative_prompt = ""
        pl_extractor.diffusion_extractor.init_extractor(pl_extractor.device)
        pl_extractor.diffusion_extractor.change_cond_emb(conditionning['conditioning'], "image", "cond")
        pl_extractor.diffusion_extractor.change_cond_emb(conditionning['unconditional_conditioning'], "", "uncond")

    with torch.autocast("cuda"):
        latents = torch.randn((
            pl_extractor.diffusion_extractor.batch_size, 
            pl_extractor.diffusion_extractor.unet.in_channels, 
            512 // 8, 512 // 8), 
                generator=pl_extractor.diffusion_extractor.generator, 
                device=device)

        with torch.inference_mode():
            _, outputs = pl_extractor.diffusion_extractor.forward(
                latents=latents, 
                guidance_scale=guidance_scale, 
                control_image=control_image.to(device))
                
        synthetic_images = outputs[-1] / 0.18215
        synthetic_images = pl_extractor.diffusion_extractor.vae.decode(
            synthetic_images.to(pl_extractor.diffusion_extractor.vae.dtype)).sample
        synthetic_images = (synthetic_images / 2 + 0.5).clamp(0, 1)
        output_image = torchvision.transforms.ToPILImage()(synthetic_images[0])

    return output_image
