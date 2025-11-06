###
# From https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures
###

import os
import PIL
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import (
    DDIMScheduler, 
    StableDiffusionPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel
)
from archs.stable_diffusion.resnet import set_timestep, collect_feats

"""
Functions for running the generalized diffusion process 
(either inversion or generation) and other helpers 
related to latent diffusion models. Adapted from 
Shape-Guided Diffusion (Park et. al., 2022).
https://github.com/shape-guided-diffusion/shape-guided-diffusion/blob/main/utils.py
"""

def get_tokens_embedding(clip_tokenizer, clip, prompt, device):
  tokens = clip_tokenizer(
    prompt,
    padding="max_length",
    max_length=clip_tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
    return_overflowing_tokens=True,
  )
  # dirty hack to prevent HalfFloat error
  input_ids = tokens.input_ids.to(device)
  embedding = clip(input_ids).last_hidden_state
  return tokens, embedding

def latent_to_image(vae, latent):
  latent = latent / 0.18215
  image = vae.decode(latent.to(vae.dtype)).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  image = (image[0] * 255).round().astype("uint8")
  image = Image.fromarray(image)
  return image

def image_to_latent(vae, image, generator=None, mult=64, w=512, h=512):
  image = image.resize((w, h), resample=PIL.Image.LANCZOS)
  image = np.array(image).astype(np.float32)
  # remove alpha channel
  if len(image.shape) == 2:
    image = image[:, :, None]
  else:
    image = image[:, :, (0, 1, 2)]
  # (b, c, w, h)
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)
  image = image / 255.0
  image = 2. * image - 1.
  image = image.to(vae.device)
  image = image.to(vae.dtype)
  return vae.encode(image).latent_dist.sample(generator=generator) * 0.18215

def get_xt_next_inpainting(xt, et, at, at_next, eta, img, mask):
  """
  Uses the DDIM formulation for sampling xt_next
  Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
  """
  x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
  if eta == 0:
    c1 = 0
  else:
    c1 = (
      eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    )
  c2 = ((1 - at_next) - c1 ** 2).sqrt()
  xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(et) + c2 * et
  
  xt_next = (1.0 - mask) * img + mask * xt_next
  
  return x0_t, xt_next

def get_xt_next(xt, et, at, at_next, eta):
  """
  Uses the DDIM formulation for sampling xt_next
  Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
  """
  x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
  if eta == 0:
    c1 = 0
  else:
    c1 = (
      eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    )
  c2 = ((1 - at_next) - c1 ** 2).sqrt()
  xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(et) + c2 * et
  return x0_t, xt_next

def get_timesteps(scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order :]

    return timesteps, t_start #, num_inference_steps - t_start

def generalized_steps(x, model, scheduler, **kwargs):
  """
  Performs either the generation or inversion diffusion process.
  """
  
  inpaint_img = kwargs.get("inpaint_img", None)
  inpaint_mask = kwargs.get("inpaint_mask", None)
  inpaint_mode = False if inpaint_img is None or inpaint_mask is None else True
  
  seq = scheduler.timesteps
  t_start = 0
  seq = torch.flip(seq, dims=(0,))
  b = scheduler.betas
  b = b.to(x.device)

  with torch.no_grad():
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    if kwargs.get("run_inversion", False):
      seq_iter = seq_next
      seq_next_iter = seq
    else:
      seq_iter = reversed(seq)
      seq_next_iter = reversed(seq_next)

    xs = [x]
    prog_bar = tqdm(total=len(seq_iter))
    for i, (t, next_t) in enumerate(zip(seq_iter, seq_next_iter)):
      i += t_start
      t = (torch.ones(n) * t).to(x.device)
      next_t = (torch.ones(n) * next_t).to(x.device)
      if t.sum() == -t.shape[0]:
        at = torch.ones_like(t)
      else:
        at = (1 - b).cumprod(dim=0).index_select(0, t.long())
      if next_t.sum() == -next_t.shape[0]:
        at_next = torch.ones_like(t)
      else:
        at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
      
      # Expand to the correct dim
      at, at_next = at[:, None, None, None], at_next[:, None, None, None]

      if kwargs.get("run_inversion", False):
        set_timestep(model, len(seq_iter) - i - 1)
      else:
        set_timestep(model, i)

      xt = xs[-1].to(x.device)
      cond = kwargs["conditional"]
      guidance_scale = kwargs.get("guidance_scale", -1)
      do_classifier_free_guidance = guidance_scale > 1.0

      controlnet = kwargs.get("controlnet", None)
      if controlnet is not None:
        if do_classifier_free_guidance:
          prompt_embeds          = kwargs["conditional"]
          negative_prompt_embeds = kwargs["unconditional"]
          prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
          prompt_embeds          = kwargs["conditional"]
        control_image = kwargs.get("control_image", None)
        if control_image is None:
          raise FileNotFoundError
        if do_classifier_free_guidance: # and not guess_mode:
          control_image_ = []
          for ctrl_idx in range(len(control_image)):
            control_image_.append(
                torch.cat([control_image[ctrl_idx]] * 2)
              )
          control_image = control_image_
          xt_controlnet = torch.cat([xt] * 2)
        else:
          xt_controlnet = xt
        cond_scale = kwargs.get("cond_scale", 1.0)
        guess_mode = False
        down_block_res_samples, mid_block_res_sample = controlnet(
            xt_controlnet,
            t,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=control_image,
            conditioning_scale=cond_scale, #[0.5, 1.0, 1.0]
            guess_mode=guess_mode,
            return_dict=False,
        )
      else:
        mid_block_res_sample = None
        down_block_res_samples = None

      if guidance_scale == -1 or not do_classifier_free_guidance:
        et = model(xt, t, 
                   encoder_hidden_states=cond,
                   down_block_additional_residuals=down_block_res_samples if down_block_res_samples is not None else None,
                   mid_block_additional_residual=mid_block_res_sample if mid_block_res_sample is not None else None).sample
      else:
        # If using Classifier-Free Guidance, the saved feature maps
        # will be from the last call to the model, the conditional prediction
        uncond = kwargs["unconditional"]
        et_uncond = model(xt, t, encoder_hidden_states=uncond,
                          down_block_additional_residuals=[
                            down_block_res_sample[0:1] for down_block_res_sample in down_block_res_samples] if down_block_res_samples is not None else None,
                          mid_block_additional_residual=mid_block_res_sample[0:1] if mid_block_res_sample is not None else None).sample
        et_cond = model(xt, t, encoder_hidden_states=cond,
                        down_block_additional_residuals=[
                          down_block_res_sample[1:2] for down_block_res_sample in down_block_res_samples] if down_block_res_samples is not None else None,
                        mid_block_additional_residual=mid_block_res_sample[1:2] if mid_block_res_sample is not None else None).sample
        et = et_uncond + guidance_scale * (et_cond - et_uncond)
      
      eta = kwargs.get("eta", 0.0)
      
      if inpaint_mode:
        # x0_t, xt_next = get_xt_next_inpainting(xt, et, at, at_next, eta, inpaint_img, inpaint_mask)
        
        # compute the previous noisy sample x_t -> x_t-1
        xt_next = scheduler.step(et, int(t.item()), xt, eta=eta, return_dict=False)[0]

        if i < len(scheduler.timesteps) - 1:
          # noise_timestep = timesteps[i + 1]
          noise_timestep = int(next_t.item())
          init_latents_proper = inpaint_img
          init_latents_proper = scheduler.add_noise(
              init_latents_proper, x, torch.tensor([noise_timestep])
          )

        xt_next = (1 - inpaint_mask) * init_latents_proper + inpaint_mask * xt_next
      else:
        x0_t, xt_next = get_xt_next(xt, et, at, at_next, eta)
        
      xs.append(xt_next)
      prog_bar.update()

    prog_bar.close()
    return xs
    # return x0_preds

def freeze_weights(weights):
  for param in weights.parameters():
    param.requires_grad = False

def init_models(
    model_id="runwayml/stable-diffusion-v1-5",
    freeze=True
  ):
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
  )
  unet = pipe.unet
  vae = pipe.vae
  clip = pipe.text_encoder
  clip_tokenizer = pipe.tokenizer
  if freeze:
    freeze_weights(unet)
    freeze_weights(vae)
    freeze_weights(clip)
  return unet, vae, clip, clip_tokenizer

def init_models_controlnet(model_list):
  controlnets = []
  for model_link in model_list:
    controlnets.append(
      ControlNetModel.from_pretrained(
        model_link, torch_dtype=torch.float16)
      )
  pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnets,
    torch_dtype=torch.float16
  )
  scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  unet = pipe.unet
  vae = pipe.vae
  clip = pipe.text_encoder
  clip_tokenizer = pipe.tokenizer
  controlnet = pipe.controlnet
  freeze_weights(unet)
  freeze_weights(vae)
  freeze_weights(clip)
  freeze_weights(controlnet)
  
  return unet, vae, clip, clip_tokenizer, controlnet, scheduler, pipe

def collect_and_resize_feats(unet, idxs, timestep, resolution=-1):
  latent_feats = collect_feats(unet, idxs=idxs)
  latent_feats = [feat[timestep] for feat in latent_feats]
  if resolution > 0:
      latent_feats = [torch.nn.functional.interpolate(latent_feat, size=resolution, mode="bilinear") for latent_feat in latent_feats]
  latent_feats = torch.cat(latent_feats, dim=1)
  return latent_feats