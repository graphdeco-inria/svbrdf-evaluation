
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import rgb_to_grayscale
from skimage.morphology import disk, binary_opening, binary_closing, erosion
from skimage.filters import gaussian
import time

def get_camera_params(img_shape, fov_deg=85):

    fy = img_shape[0] * 0.5 / (np.tan (fov_deg * 0.5 * np.pi/180))
    fx = img_shape[1] * 0.5 / (np.tan (fov_deg * 0.5 * np.pi/180))
    K = np.array([
        [fx, 0, (img_shape[1] - 1) / 2],
        [0, fy, (img_shape[0] - 1) / 2],
        [0, 0, 1],
    ])
    u, v = np.meshgrid(
        np.arange(img_shape[1], 0, -1, dtype=np.float32) - 1.0, 
        np.arange(img_shape[0], 0, -1, dtype=np.float32) - 1.0)
    rayxyz_vec = np.stack([
        (u - K[0, 2]) / K[0, 0], 
        (v - K[1, 2]) / K[1, 1], 
        np.ones_like(u)], axis=-1)
    rayxyz_norm = np.linalg.norm(rayxyz_vec, axis=-1, keepdims=True)
    # if blender_depth:
    #     rayxyz_normalized = rayxyz_vec                  # blender depth
    # else:
    #     rayxyz_normalized = rayxyz_vec / rayxyz_norm    # mitsuba distance

    return K, rayxyz_vec, rayxyz_norm

def get_xyz_target_camspace(
        rayxyz_vec,
        depthmap_target, #np.array shape (H,W)
        cam2world_source, 
        cam2world_target):

    # If distance (Mistuba), rayxyz_vec should be normalized
    xyz_source_cam_source = depthmap_target[..., None] * rayxyz_vec
    Rt_cam_source_2_target = np.linalg.inv(cam2world_target) @ cam2world_source
    R_cam_source_2_target = Rt_cam_source_2_target[:3, :3]
    t_cam_source_2_target = Rt_cam_source_2_target[:3, 3]
    xyz_source_cam_target = np.dot(xyz_source_cam_source, R_cam_source_2_target.T) + t_cam_source_2_target

    return xyz_source_cam_target

def get_torch_uv_target_cam_source(rayxyz, K, depthmap_target, cam2world_source, cam2world_target):
    xzy_target_cam_source = get_xyz_target_camspace(rayxyz, depthmap_target, cam2world_target, cam2world_source)
    uv_target_cam_source = xzy_target_cam_source[..., :2] / xzy_target_cam_source[..., 2:]
    uv_target_cam_source[..., 0] = uv_target_cam_source[..., 0] * K[0, 0] + K[0, 2]
    uv_target_cam_source[..., 1] = uv_target_cam_source[..., 1] * K[1, 1] + K[1, 2]
    torch_uv_target_cam_source = torch.tensor(
        uv_target_cam_source, dtype=torch.float32)[None] * (2 / (depthmap_target.shape[0] - 1)) - 1
    torch_uv_target_cam_source = - torch_uv_target_cam_source
    return torch_uv_target_cam_source

def get_img_source_cam_target(img_source: np.array, torch_uv_target_cam_source):
    torch_im_source = torch.tensor(img_source, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    img_source_cam_target = torch.nn.functional.grid_sample(
        input=torch_im_source**(2.2), 
        grid=torch_uv_target_cam_source, 
        align_corners=True)
    return img_source_cam_target**(1/2.2) #process in linear space, return in sRGB

def get_pc_source_cam_target(
    torch_uv_target_cam_source, 
    rayxyz_vec, 
    depthmap_source, 
    cam2world_source, 
    cam2world_target):
    xyz_source_cam_target = get_xyz_target_camspace(
        rayxyz_vec, depthmap_source, cam2world_source, cam2world_target)
    xyz_source_cam_target = torch.tensor(
        xyz_source_cam_target, dtype=torch.float32).permute(2,0,1)[None]
    pc_source_cam_target = torch.nn.functional.grid_sample(
        input=xyz_source_cam_target, 
        grid=torch_uv_target_cam_source, 
        align_corners=True)
    return pc_source_cam_target

def get_image_mask_from_depths(
        torch_uv_target_cam_source, 
        rayxyz_vec, 
        rayxyz_norm,
        depthmap_source, 
        depthmap_target, 
        cam2world_source, 
        cam2world_target, 
        mask_thresh=0.001):
    
    pc_source_cam_target = get_pc_source_cam_target(
        torch_uv_target_cam_source, rayxyz_vec, 
        depthmap_source, cam2world_source, cam2world_target)
    
    dist_source_cam_target = pc_source_cam_target[0].norm(dim=0).numpy()
    distmap_target = depthmap_target * rayxyz_norm[..., 0]
    
    diff_distance = np.abs(dist_source_cam_target - distmap_target)
    
    diff_distance_mask = diff_distance.copy()
    diff_distance_mask[diff_distance <  mask_thresh] = 1.0
    diff_distance_mask[diff_distance >= mask_thresh] = 0.0
    
    return diff_distance_mask

def get_reprojected_image_and_mask(
        img_source,   #np.array shape (H,W,C)
        distmap_source, #np.array shape (H,W)
        distmap_target, #np.array shape (H,W)
        cam2world_source,
        cam2world_target,
        fov_deg=85,
        mask_thresh=0.01,
        use_blender_depth=False):
    
    K, posCam_tmp_normalized = get_camera_params(
        img_source.shape, 
        fov_deg, 
        blender_depth=use_blender_depth)
    torch_uv_t_cam_source = get_torch_uv_target_cam_source(
        posCam_tmp_normalized, K,
        distmap_target, cam2world_source, cam2world_target)
    img_source_cam_target = get_img_source_cam_target(
        img_source, torch_uv_t_cam_source)
    mask_img_source_cam_target = get_image_mask_from_depths(
        torch_uv_t_cam_source, 
        posCam_tmp_normalized, 
        distmap_source, 
        distmap_target, 
        cam2world_source, 
        cam2world_target,
        mask_thresh=mask_thresh)
    
    img_source_cam_target_masked = mask_img_source_cam_target[..., None] * \
        img_source_cam_target[0].permute(1,2,0).numpy()

    return img_source_cam_target_masked, mask_img_source_cam_target

def reproj_tensor_from_depthsandcams(
    hf_tensor, 
    distmap_target,
    cam2world_source,
    cam2world_target,
    fov_deg=85, 
    exp_dir=None, 
    img_debug=None,
    write_debug_output=False,
    use_blender_depth=False):

    device = hf_tensor.device

    distmap_target = cv2.resize(distmap_target, 
        (hf_tensor.shape[-1],hf_tensor.shape[-2]), interpolation=cv2.INTER_NEAREST_EXACT)
    
    K, posCam_tmp_normalized = get_camera_params(distmap_target.shape, fov_deg, blender_depth=use_blender_depth)

    torch_uv_target_cam_source = get_torch_uv_target_cam_source(posCam_tmp_normalized, K, distmap_target, cam2world_source, cam2world_target)

    if write_debug_output:
        # img_debug = cv2.cvtColor(cv2.imread("experiments/reproj/img_debug.png"), cv2.COLOR_RGB2BGR)
        img_debug = cv2.resize(img_debug, (64,64), interpolation=cv2.INTER_LINEAR)
        torch_img0 = torch.tensor(img_debug, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
        img0_cam1 = torch.nn.functional.grid_sample(
            input=torch_img0.to(device), 
            grid=torch_uv_target_cam_source.to(device), 
            align_corners=True)
        cv2.imwrite(os.path.join(exp_dir,"img0_cam1.exr"), cv2.cvtColor(img0_cam1[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(exp_dir,"img0.exr"), cv2.cvtColor(torch_img0[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))

    HF_cam_target = torch.nn.functional.grid_sample(
        input=hf_tensor.float().to(device), 
        grid=torch_uv_target_cam_source.to(device),
        align_corners=True).cpu()
    
    return HF_cam_target.half()
    
def make_inpaint_condition_custom(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    image_mask = 1.0 - image_mask

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image, image_mask

def extract_svbrdf(
        pl_extractor,
        input_image, # tonemapped in [0, 1]
        normals, # in [-1, 1]
        depth, # in [0, 1]
        use_rgb_input=True,
        ):

    device = pl_extractor.device

    input_image_scaled = 2.0 * input_image - 1.0

    if normals is not None and depth is not None:
        depth_scaled = 2.0 * depth - 1.0
        inputs_list = [normals.to(device).float(), depth_scaled.to(device).float()]
    else:
        inputs_list = []
    if use_rgb_input:
        inputs_list = [rgb_to_grayscale(input_image_scaled.float())] + inputs_list

    torch.cuda.empty_cache()

    with torch.autocast("cuda"):
        with torch.inference_mode():
            activation_features, _ = pl_extractor.diffusion_extractor.forward(input_image_scaled.float())
            svbrdf_out = pl_extractor.extractor_wrapper(
                activation_features,
                inputs_list = inputs_list
            )

    return svbrdf_out
                
def generate_and_extract(
        pl_extractor,
        generator,
        control_images,
        ctrlnet_cond_scale,
        normals,
        depth,
        img2inpaint = None, # in [-1, 1] and masked
        mask = None,
        prompt = "", 
        negative_prompt = "", 
        guidance_scale = 7.5,
        resolution = 512,
        pfd_conditionning = None,
        sdedit_img=None,
        sdedit_strength=0.0
        ):

    device = pl_extractor.device
    pl_extractor.diffusion_extractor.generator = generator
    
    if pfd_conditionning is None:
        pl_extractor.diffusion_extractor.prompt = prompt
        pl_extractor.diffusion_extractor.negative_prompt = negative_prompt
        #TODO: use hf implem to prevent slight float mismatch
        pl_extractor.diffusion_extractor.init_extractor(pl_extractor.device)
    else:
        pl_extractor.diffusion_extractor.prompt = ""
        pl_extractor.diffusion_extractor.negative_prompt = ""
        pl_extractor.diffusion_extractor.init_extractor(pl_extractor.device)
        pl_extractor.diffusion_extractor.change_cond_emb(pfd_conditionning['conditioning'], "image", "cond")
        pl_extractor.diffusion_extractor.change_cond_emb(pfd_conditionning['unconditional_conditioning'], "", "uncond")

    with torch.autocast("cuda"):
        latents = torch.randn((
                pl_extractor.diffusion_extractor.batch_size, 
                pl_extractor.diffusion_extractor.unet.config.in_channels, 
                resolution // 8, resolution // 8), 
                    generator=pl_extractor.diffusion_extractor.generator, 
                    device=device)
        if sdedit_img is not None:
            sdedit_img_latent = pl_extractor.diffusion_extractor.vae.encode(
                sdedit_img.to(device)
            ).latent_dist.sample(generator=None) * 0.18215
        else:
            sdedit_img_latent = None
            
        if img2inpaint is not None:
            img2inpaint = pl_extractor.diffusion_extractor.vae.encode(
                img2inpaint.to(device)
            ).latent_dist.sample(generator=None) * 0.18215
            
        if mask is not None:
            mask = mask.to(device)

        torch.cuda.empty_cache()

        with torch.inference_mode():
            feats, outputs = pl_extractor.diffusion_extractor.forward(
                latents=latents, 
                guidance_scale=guidance_scale, 
                control_image=control_images, 
                cond_scale=ctrlnet_cond_scale, 
                inpaint_img=img2inpaint, 
                inpaint_mask=mask,
                sdedit_img_latent=sdedit_img_latent,
                pipeline_quant=1.0 - sdedit_strength)

            synthetic_images = outputs[-1] / 0.18215
            synthetic_images = pl_extractor.diffusion_extractor.vae.decode(
                synthetic_images.to(pl_extractor.diffusion_extractor.vae.dtype)).sample

            depth_scaled = 2.0 * depth - 1.0
            inputs_list = [normals.to(device).float(), depth_scaled.to(device).float()]
            inputs_list = [rgb_to_grayscale(synthetic_images.float())] + inputs_list

            torch.cuda.empty_cache()

            if feats is not None:
                start = time.time()
                svbrdfPred = pl_extractor.extractor_wrapper(
                    activation_features = feats,
                    inputs_list = inputs_list
                )
                total_time = time.time() - start
                print(f"svbrdf extraction: {total_time:.3f}s")
            else:
                svbrdfPred = torch.zeros(
                    (1, 5, synthetic_images.shape[-2], synthetic_images.shape[-1]),
                    device=synthetic_images.device)
            
            synthetic_images = (synthetic_images * 0.5 + 0.5).clamp(0, 1)

    return synthetic_images, svbrdfPred

def erode_mask(image_mask, erode_size=5):
    
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
            (2 * erode_size + 1, 2 * erode_size + 1), (erode_size, erode_size))
    eroded_image_mask = cv2.erode(np.array(image_mask), element)
    eroded_image_mask = Image.fromarray(np.array(eroded_image_mask))
    return eroded_image_mask
    
def inpaint_view(
        pl_extractor,
        generator,
        control_image_masked,
        image_mask_preproc,
        control_images,
        depth,
        normals,
        controlnet_conditioning_scales,
        prompt=None,
        negative_prompt=None,
        guidance_scale=7.5,
        resolution=512,
        pfd_conditionning=None
        ):
    
    control_image_masked = (2.0 * control_image_masked - 1.0) * (image_mask_preproc < 0.5)

    image_mask_preproc = torch.from_numpy(image_mask_preproc).unsqueeze(0).unsqueeze(0)
    image_mask_preproc = torch.nn.functional.interpolate(image_mask_preproc, scale_factor=1/8, mode='nearest-exact')

    image_inpainted, svbrdfPred = generate_and_extract(
        pl_extractor=pl_extractor,
        generator=generator,
        control_images=control_images,
        ctrlnet_cond_scale=controlnet_conditioning_scales,
        normals=normals,
        depth=depth,
        img2inpaint=control_image_masked, 
        mask=image_mask_preproc, 
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        guidance_scale=guidance_scale, 
        resolution=resolution,
        pfd_conditionning=pfd_conditionning
    )

    return image_inpainted, svbrdfPred

def fill_image_holes_with_mask(img, mask, del_small_bits):
    img = img * mask[..., np.newaxis]
    if not del_small_bits:
        processed_mask = binary_closing(mask, disk(1))
    else:
        processed_mask = binary_closing(binary_opening(mask, disk(1)))
    holes_mask = processed_mask * (1.0 - mask)
    blurred_img = gaussian(img, sigma=5, mode='reflect', channel_axis=-1)
    filled_img = img + holes_mask[..., np.newaxis] * blurred_img
    return filled_img, processed_mask

def compute_smoothed_masks(current_mask, new_mask, ks_smoothing=5, ks_preserve=10):
    curr_mask_only = current_mask * (1.0 - new_mask)
    blurred_overlap = new_mask * gaussian(
        erosion(current_mask, disk(ks_preserve)), sigma=ks_smoothing)
    blurred_new_mask_only = new_mask * (1.0 - blurred_overlap)
    return curr_mask_only, blurred_overlap, blurred_new_mask_only

def multistep_reprojection(
    K, 
    view_id, 
    depthmap_list, 
    cam2world_list, 
    images_list, 
    rayxyz_vec, 
    rayxyz_norm,
    threshold=0.02):

    depthmap_target = depthmap_list[view_id]
    cam2world_target = cam2world_list[view_id]
    current_mask = np.zeros_like(depthmap_target, dtype=np.float32)
    current_target = np.zeros_like(images_list[0], dtype=np.float32)

    for reproj_step in range(view_id):

        depthmap_source = depthmap_list[reproj_step]
        cam2world_source = cam2world_list[reproj_step]
        img_source = np.array(images_list[reproj_step])
        
        torch_uv_t_cam_source = get_torch_uv_target_cam_source(
            rayxyz_vec, # rayxyz_vec/rayxyz_norm for mitsuba
            K,
            depthmap_target, 
            cam2world_source, 
            cam2world_target)

        img_source_cam_target = get_img_source_cam_target(
            img_source, 
            torch_uv_t_cam_source)
        
        mask_img_source_cam_target = get_image_mask_from_depths(
            torch_uv_t_cam_source, 
            rayxyz_vec, 
            rayxyz_norm,
            depthmap_source, 
            depthmap_target, 
            cam2world_source, 
            cam2world_target, 
            mask_thresh=threshold)
        
        img_source_cam_target, mask_img_source_cam_target = fill_image_holes_with_mask(
            img=img_source_cam_target[0].permute(1,2,0).numpy(), 
            mask=mask_img_source_cam_target,
            del_small_bits = (reproj_step != 0))
        
        curr_mask_only, blurred_overlap, blurred_new_mask_only = compute_smoothed_masks(
            current_mask=current_mask,
            new_mask=mask_img_source_cam_target,
        )
        
        current_mask = curr_mask_only + blurred_overlap + blurred_new_mask_only

        current_target = \
            curr_mask_only[..., np.newaxis] * current_target \
            + blurred_overlap[..., np.newaxis] * current_target \
            + blurred_new_mask_only[..., np.newaxis] * img_source_cam_target
    
    return current_target, current_mask

def multistep_hypertensor_reprojection(
    exp_dir,
    K,
    view_id, 
    distmap_list, 
    cam2world_list, 
    images_list, 
    hf_list, 
    posCam_tmp_normalized):

    distmap_target = distmap_list[view_id]
    cam2world_target = cam2world_list[view_id]
    current_mask = np.zeros_like(distmap_target, dtype=np.float32)
    current_target = np.zeros_like(images_list[0], dtype=np.float32)

    for reproj_step in range(view_id):

        distmap_source = distmap_list[reproj_step]
        cam2world_source = cam2world_list[reproj_step]
        img_source = np.array(images_list[reproj_step])
        hf_source = hf_list[reproj_step]

        HF_source_cam_target = reproj_tensor_from_depthsandcams(
            hf_tensor=hf_source, 
            distmap_target=distmap_target, # in [0, +inf]
            cam2world_source=cam2world_source,
            cam2world_target=cam2world_target,
            exp_dir=exp_dir,
            )

        torch_uv_t_cam_source = get_torch_uv_target_cam_source(
            posCam_tmp_normalized, 
            K,
            distmap_target, 
            cam2world_source, 
            cam2world_target)
        
        img_source_cam_target = get_img_source_cam_target(
            img_source, 
            torch_uv_t_cam_source)

        mask_img_source_cam_target = get_image_mask_from_depths(
            torch_uv_t_cam_source, 
            posCam_tmp_normalized, 
            distmap_source, 
            distmap_target, 
            cam2world_source, 
            cam2world_target)
        
        mask_img_source_cam_target = (1.0 - current_mask) * mask_img_source_cam_target
        current_mask = current_mask + mask_img_source_cam_target

        img_source_cam_target_masked = mask_img_source_cam_target[..., None] * img_source_cam_target[0].permute(1,2,0).numpy()
        
        current_target += img_source_cam_target_masked
    
    return current_target, current_mask
