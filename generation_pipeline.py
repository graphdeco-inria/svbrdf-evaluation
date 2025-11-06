import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch
import random
import numpy as np
import argparse
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

import utils.generation_utils as generation_utils
from thirdparty.ip_adapter.ip_adapter import IPAdapter
from utils.reproj_utils import (
    generate_and_extract, 
    inpaint_view, 
    get_camera_params,
    multistep_reprojection,
    erode_mask,
    make_inpaint_condition_custom
)

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

init_gen_model_list = [
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15_lineart",
    ]

inpaint_model_list = [
    "lllyasviel/control_v11p_sd15_inpaint",
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15_lineart"
    ]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", 		        type=str, 	default="./data/bathroom") 
    parser.add_argument("--output_folder", 	        type=str,	default="./output/bathroom")
    parser.add_argument("--sensor_count", 	        type=int,	default=5)
    parser.add_argument("--prompt", 	            type=str,	default="A cozy rustic bathroom with textured stone walls, a hammered copper freestanding bathtub. " \
    "The vanity is a thick slab of reclaimed wood with vessel sinks made of copper. Wrought iron fixtures and dim lantern lighting create a warm, inviting atmosphere.")
    # prompt = "A cozy rustic bathroom with textured stone walls, a hammered copper freestanding bathtub. The vanity is a thick slab of reclaimed wood with vessel sinks made of copper."
    # "Wrought iron fixtures and dim lantern lighting create a warm, inviting atmosphere."
    # prompt = "A kitchen corner with highly specular metallic jars and glasses, a book on a marble tabletop, highly detailed, photorealistic, specular and shiny"
    # prompt = "A kitchen corner with wooden and metallic elements on a marble tabletop"
    # prompt = "A baroque style bedroom, crisp and sharp details, with detailed golden objects and white sheets, with two wooden drawers"
    # prompt = "A modern style bedroom with golden moldings, a golden carpet, and two curtains surrounding the golden bed"
    # prompt = "A modern style livingroom with shiny wood and a detailed golden ornaments"
    # prompt = "A modern style livingroom with shiny objects and a detailed textures"
    # prompt = "A modern style livingroom with wood, detailed paintings, and shiny golden object"
    # prompt = "A modern style bathroom with wood, marbles, and shiny golden objects"
    # prompt = "A modern style kitchen with wooden planks, golden cups, on a marble tabletop"
    # prompt = "A modern kitchen with highly detailed patterns on a wooden tabletop with ceramic jars, shiny silver cups, cutting board, in front of a brick wall"
    # prompt = "A modern style livingroom with wood, detailed paintings, and a golden object"
    parser.add_argument("--ckpt_folder", 	        type=str,	default="checkpoints/unet_hf/")
    parser.add_argument("--ckpt_name", 	            type=str,	default="UNet_HF_val_loss=0.435621.ckpt")
    parser.add_argument("--conf_name", 	            type=str,	default="config_UNet_HF_val_loss=0.435621.yaml")
    parser.add_argument("--seed", 	                type=int,	default=1200)
    
    parser.add_argument("--output_gen_images",	    dest='output_gen_images', action='store_true', help="output all generated images")

    ### Do not change (default params)
    parser.add_argument("--init_guidance_scale",    type=float,	default=7.5)
    parser.add_argument("--gen_guidance_scale",     type=float,	default=2.5)
    parser.add_argument("--erode_size",             type=int,	default=10)
    parser.add_argument("--depth_mask_threshold",   type=float,	default=0.02)
    parser.add_argument("--negative_prompt",        type=str,	default="vignetting, text")

    parser.set_defaults(output_init_image = False)
    parser.set_defaults(output_gen_images = False)

    args = parser.parse_args()

    generator = torch.Generator(device=device).manual_seed(args.seed)

    os.makedirs(args.output_folder, exist_ok=True)

    pl_extractor = generation_utils.init_module(
        ckpt_folder = args.ckpt_folder, 
        ckpt_name = args.ckpt_name, 
        conf_name = args.conf_name,
        device = device,
        model_list=init_gen_model_list).to(device)
  
    use_blender_depth = True

    lineart  = generation_utils.load_lineart(os.path.join(args.input_dir, f"0.png"))
    depth_normalized = generation_utils.load_normalized_depth(os.path.join(args.input_dir, "depth_00.exr"), blender_depth=use_blender_depth)
    normals  = generation_utils.load_normals(os.path.join(args.input_dir, "normal_00.exr"))

    control_image_depth = 1 - depth_normalized.repeat(1,3,1,1) # SD-depth format: white to black
    control_images = [
        control_image_depth.to(device).half(), 
        lineart.to(device).half(),
        ]

    ##################
    # Step 1: Generate initial Image and SVBRDF
    ##################

    generated_rgb, generated_svbrdf = generate_and_extract(
        pl_extractor=pl_extractor,
        generator=generator,
        control_images=control_images,
        ctrlnet_cond_scale=[1.0, 0.8], # depth, lineart
        depth=depth_normalized,
        normals=normals,
        guidance_scale=args.init_guidance_scale, # default: 2.5
        prompt=args.prompt, 
        negative_prompt=args.negative_prompt, 
        resolution=depth_normalized.shape[-1],
    )

    ToPILImage()(make_grid([
            generated_rgb[0],
            generated_svbrdf[:, 0:3][0],
            generated_svbrdf[:, 3:4][0].repeat(3, 1, 1),
            generated_svbrdf[:, 4:5][0].repeat(3, 1, 1)
        ], nrow=4, padding=0)).save(
            os.path.join(args.output_folder, 
                         f"seed{args.seed}_{0:02d}_svbrdf.png"))
    print("saved:", os.path.join(args.output_folder, f"seed{args.seed}_{0:02d}_svbrdf.png"))
    
    ##################
    # Load Triple ControlNet and IP Adapter for the reprojection steps
    ##################

    print("Loading reprojection pipeline...")

    pl_extractor.diffusion_extractor.change_controlnets(inpaint_model_list, device)
    pl_extractor.diffusion_extractor.init_extractor(pl_extractor.device)

    ip_adapter = IPAdapter(
        pl_extractor.diffusion_extractor.pipe,
        "thirdparty/ip_adapter/models/ip-adapter-plus_sd15.bin", 
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
        device=device, 
        dtype=torch.float16)

    torch.cuda.empty_cache()

    ##################
    # Step 2: Reprojection and Inpainting
    ##################

    prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
        ToPILImage()(generated_rgb[0]),
        prompt=args.prompt,
        negative_prompt="vignetting",
    )

    pfd_cond = {
        'conditioning':prompt_embeds,
        'unconditional_conditioning':negative_prompt_embeds
    }

    initial_image = np.array(ToPILImage()(generated_rgb[0]))

    all_images_mask = []
    all_images_reprojected = []
    all_images_results = [Image.fromarray((initial_image).astype(np.uint8))]
    all_svbrdf_results = [generated_svbrdf.cpu()]

    depth_r_list    = []
    normals_list    = []
    depth_n_list    = []
    cam2world_list  = []
    lineart_list    = []
    idmap_list      = []
    for view_id in range(args.sensor_count):
        distmap_ = cv2.imread(os.path.join(args.input_dir, f"depth_{view_id:02d}.exr"), -1).astype(np.float32)
        depth_r_list.append(distmap_[..., 0] if use_blender_depth else distmap_)
        normals_list.append(generation_utils.load_normals(os.path.join(args.input_dir, f"normal_{view_id:02d}.exr")))
        depth_n_list.append(generation_utils.load_normalized_depth(os.path.join(args.input_dir, f"depth_{view_id:02d}.exr"), blender_depth=use_blender_depth))
        cam2world_list.append(np.load(os.path.join(args.input_dir,f'cam{view_id:02d}.npy')))
        lineart_list.append(cv2.imread(os.path.join(args.input_dir, f"{view_id}.png"), -1).astype(np.float32)[..., 3:4])

    fov_deg = 85
    K, rayxyz_vec, rayxyz_norm = get_camera_params(initial_image.shape, fov_deg)

    if not use_blender_depth:
        rayxyz_vec = rayxyz_vec / rayxyz_norm
        rayxyz_norm = rayxyz_norm / rayxyz_norm

    view_id = 0

    print("Starting reprojection pipeline")

    for view_id in range(1, args.sensor_count):
        
        img_target_masked, img_mask = multistep_reprojection(
            K,
            view_id, 
            depth_r_list, 
            cam2world_list, 
            all_images_results, 
            rayxyz_vec, 
            rayxyz_norm,
            threshold=args.depth_mask_threshold)

        all_images_reprojected.append(img_target_masked)
        all_images_mask.append(img_mask)

        normals_target = normals_list[view_id]
        target_lineart = lineart_list[view_id]
        target_depth_n = depth_n_list[view_id]

        image_target_masked = Image.fromarray((255*img_target_masked).astype(np.uint8))
        image_mask = Image.fromarray((255*img_mask).astype(np.uint8))
        
        eroded_image_mask = erode_mask(image_mask, args.erode_size)

        control_image_masked, image_mask_preproc = make_inpaint_condition_custom(image_target_masked, eroded_image_mask)
        
        control_image_depth = 1 - target_depth_n.repeat(1,3,1,1) # SD-depth format: white to black
        control_image_lineart = torch.from_numpy(target_lineart).permute(2, 0, 1).repeat(3, 1, 1)[None, :, :, :] / 255.0
        
        control_images = [
            control_image_masked.to(device).half(), 
            control_image_depth.to(device).half(), 
            control_image_lineart.to(device).half(),
            ]
        
        torch.cuda.empty_cache()

        inpainted_view, svbrdf_view = inpaint_view(
            pl_extractor=pl_extractor,
            generator=generator,
            control_image_masked=control_image_masked,
            image_mask_preproc=image_mask_preproc,
            control_images=control_images,
            depth = target_depth_n,
            normals = normals_target,
            controlnet_conditioning_scales=[1.0, 1.0, 1.0], # inpaint, depth, lineart
            guidance_scale=args.gen_guidance_scale,
            resolution=normals_target.shape[-1],
            pfd_conditionning=pfd_cond
            )
        
        all_images_results.append(ToPILImage()(inpainted_view[0]))
        all_svbrdf_results.append(svbrdf_view.cpu())
        
        ToPILImage()(make_grid([
            inpainted_view[0].squeeze(),
            svbrdf_view[:, 0:3][0],
            svbrdf_view[:, 3:4][0].repeat(3, 1, 1),
            svbrdf_view[:, 4:5][0].repeat(3, 1, 1)
        ], nrow=4, padding=0)).save(os.path.join(args.output_folder, f"seed{args.seed}_{view_id:02d}_svbrdf.png"))
        print("saved:", os.path.join(args.output_folder, f"seed{args.seed}_{view_id:02d}_svbrdf.png"))

        torch.cuda.empty_cache()

        if args.output_gen_images:
            for idx in range(len(all_images_results)):
                all_images_results[idx].save(os.path.join(args.output_folder, f"img_seed{args.seed}_{idx:02d}.png"))
        
