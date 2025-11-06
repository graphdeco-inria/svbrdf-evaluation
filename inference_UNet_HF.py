import os
import torch
import random
import numpy as np
import argparse
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor

import utils.generation_utils as generation_utils
from utils.reproj_utils import extract_svbrdf

device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", 		        type=str, 	default="./data/bathroom") 
    parser.add_argument("--output_folder", 	        type=str,	default="./output/bathroom")
    
    parser.add_argument("--img_name", 		        type=str, 	default="img_seed1200_00.png") 

    parser.add_argument("--ckpt_folder", 	        type=str,	default="checkpoints/unet_hf/")
    parser.add_argument("--ckpt_name", 	            type=str,	default="UNet_HF_val_loss=0.435621.ckpt")
    parser.add_argument("--conf_name", 	            type=str,	default="config_UNet_HF_val_loss=0.435621.yaml")
    
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    pl_extractor = generation_utils.init_module(
        ckpt_folder = args.ckpt_folder, 
        ckpt_name = args.ckpt_name, 
        conf_name = args.conf_name,
        device = device,
        model_list=None).to(device)
  
    pl_extractor.diffusion_extractor.init_extractor(pl_extractor.device)

    img_input = Image.open(os.path.join(args.input_dir, args.img_name))
    img_input = ToTensor()(img_input)[None].to(device)[:, :3, ...]

    if pl_extractor.nogeom:
        depth_normalized = None
        normals = None
    else:
        depth_normalized = generation_utils.load_normalized_depth(
            os.path.join(args.input_dir, "depth_00.exr"), blender_depth=True)
        normals = generation_utils.load_normals(
            os.path.join(args.input_dir, "normal_00.exr"))

    extracted_svbrdf = extract_svbrdf(
        pl_extractor=pl_extractor,
        input_image=img_input,
        normals=normals,
        depth=depth_normalized,
        use_rgb_input=True
    )

    save_path = os.path.join(args.output_folder, f"{os.path.splitext(args.img_name)[0]}_svbrdf.png")
    ToPILImage()(make_grid([
            img_input[0],
            extracted_svbrdf[:, 0:3][0],
            extracted_svbrdf[:, 3:4][0].repeat(3, 1, 1),
            extracted_svbrdf[:, 4:5][0].repeat(3, 1, 1)
        ], nrow=4, padding=0)).save(save_path)
    print("saved:", save_path)
