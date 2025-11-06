import torch
import numpy as np
import cv2
import os
from lightning.pytorch.utilities import rank_zero_only
from torch.nn.functional import interpolate
from torchvision.utils import make_grid

def output_inference_images(svbrdfPred, synthetic_images, output_dir, save_name, batch_size):

    for batch_idx in range(batch_size):

        output_grid = make_grid([
                synthetic_images[batch_idx],
                svbrdfPred[:, 0:3][batch_idx],
                svbrdfPred[:, 3:4][batch_idx].repeat(3, 1, 1),
                svbrdfPred[:, 4:5][batch_idx].repeat(3, 1, 1)
            ], nrow=4, padding=0)

        cv2.imwrite(os.path.join(output_dir, 
            f"{save_name}_{batch_idx}.png" if batch_size!=1 else f"{save_name}.png"), 
            cv2.cvtColor((255 * np.clip(output_grid.permute(
                1,2,0).detach().cpu().numpy(), 0.0, 1.0)).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
    return None

def loadHdr_Img(imName, interpolation=cv2.INTER_AREA):
    im = cv2.imread(imName, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"ERROR: Open {imName} failed!")
    im = im[:, 80:640-80, :]
    im = cv2.resize(im, (512,512), interpolation = interpolation)
    im = np.transpose(im, [2, 0, 1])
    im = im[::-1, :, :].copy()
    return torch.from_numpy(im)

def loadHdr_Depth(imName, interpolation=cv2.INTER_LINEAR):
    im = cv2.imread(imName, -1)
    if im is None:
        raise ValueError(f"ERROR: Open {imName} failed!")
    if im.shape[1] == 640:
        im = im[:, 80:640-80, :]
    im = cv2.resize(im, (512, 512), interpolation = interpolation)
    if im.ndim == 2:
        im = im[..., np.newaxis]
    im = np.transpose(im, [2, 0, 1])
    im = im[::-1, :, :].copy()
    im = torch.from_numpy(im)
    im[~torch.isfinite(im)] = 0
    im /= torch.clamp(torch.max(im), min=1e-6) # Normalize depth
    if im.shape[0] == 1:
        im = im.repeat(3,1,1)
    im = im[None, :, :, :]
    return im

@rank_zero_only
def save_output_img(
    forward_out,
    save_root, save_name, sceneName,
    batch_idx, idx_save, save_exr=False):

    input_image_hdr=forward_out["input_im_hdr"]
    input_image_tn=forward_out["input_im_tonemapped"]
    segmentation_mask=forward_out["seg"]
    predictions_batch=torch.cat([
        forward_out["albedoPred"],
        forward_out["matPred"]],
        dim=1)
    groundtruth_batch=torch.cat([
        forward_out["albedo"],   
        forward_out["mat"]],   
        dim=1)

    output_grid = make_grid([
        input_image_tn[batch_idx],
        groundtruth_batch[:, 0:3][batch_idx],
        groundtruth_batch[:, 3:4][batch_idx].repeat(3, 1, 1),
        groundtruth_batch[:, 4:5][batch_idx].repeat(3, 1, 1),
        torch.pow(input_image_hdr[batch_idx], 1.0 / 2.2).clip(0, 1),
        predictions_batch[:, 0:3][batch_idx],
        predictions_batch[:, 3:4][batch_idx].repeat(3, 1, 1),
        predictions_batch[:, 4:5][batch_idx].repeat(3, 1, 1),
    ], nrow=4, padding=0)

    if idx_save == -1:
        idx_save_str=""
    else:
        idx_save_str = f"_{idx_save:06d}"

    cv2.imwrite(save_root + f"/{save_name}{idx_save_str}_{sceneName[batch_idx]}.png", 
        cv2.cvtColor((255 * np.clip(output_grid.permute(
            1,2,0).detach().cpu().numpy(), 0.0, 1.0)).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if save_exr:
        cv2.imwrite(save_root + f"/{save_name}{idx_save_str}_{sceneName[batch_idx]}.exr", 
            cv2.cvtColor(output_grid.permute(1,2,0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR))