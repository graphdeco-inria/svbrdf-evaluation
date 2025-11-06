
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import torch

from archs.diffusion_extractor import DiffusionExtractor
from archs.diffusion_extractor_controlnet import DiffusionExtractorControlNet
from archs.aggregation_network import AggregationNetwork
from archs.svbrdf_estimator import SvbrdfEstimator
from archs.stable_diffusion.resnet import collect_dims


class HyperFeatureExtractorWrapper(torch.nn.Module):
    def __init__(self, aggregation_network, svbrdf_estimator, batch_size):
        super().__init__()
        self.b = batch_size
        self.svbrdf_estimator = svbrdf_estimator
        self.aggregation_network = aggregation_network

    def forward(self, activation_features, inputs_list):
        hf_tensor = self.aggregation_network( # B, ProjDim * Fsize, W, H
            activation_features.float().view(
                (self.b, -1, activation_features.shape[-2], activation_features.shape[-1]))) 
        svbrdf_out = self.svbrdf_estimator(
            hf_tensor = hf_tensor.float(), 
            inputs_list = inputs_list)
        return svbrdf_out

class HyperFeatureExtractor(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.diffusion_extractor = DiffusionExtractor(config)
        self.diffusion_extractor_is_init = False
        self.dims = collect_dims(self.diffusion_extractor.unet, idxs=self.diffusion_extractor.idxs)
        self.extractor_wrapper = HyperFeatureExtractorWrapper(
            aggregation_network=AggregationNetwork(
                projection_dim=config["projection_dim"],
                num_norm_groups=config.get("num_norm_groups", 32),
                feature_dims=self.dims,
                save_timestep=config["save_timestep"],
                num_timesteps=config["num_timesteps"]
            ), 
            svbrdf_estimator=SvbrdfEstimator(
                proj_dim=config["projection_dim"],
                use_rgb_input=config["use_rgb_input"],
                num_norm_groups=config.get("num_norm_groups", 32),
                nogeom=config.get("nogeom", False)
            ), 
            batch_size=config["batch_size"]
            )
        self.nogeom = config.get("nogeom", False)

class HyperFeatureExtractorControlNet(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.diffusion_extractor = DiffusionExtractorControlNet(config)
        self.diffusion_extractor_is_init = False
        self.dims = collect_dims(self.diffusion_extractor.unet, idxs=self.diffusion_extractor.idxs)
        self.extractor_wrapper = HyperFeatureExtractorWrapper(
            aggregation_network=AggregationNetwork(
                projection_dim=config["projection_dim"],
                num_norm_groups=config.get("num_norm_groups", 32),
                feature_dims=self.dims,
                save_timestep=config["save_timestep"],
                num_timesteps=config["num_timesteps"]
            ), 
            svbrdf_estimator=SvbrdfEstimator(
                proj_dim=config["projection_dim"],
                use_rgb_input=config["use_rgb_input"],
                num_norm_groups=config.get("num_norm_groups", 32)
            ), 
            batch_size=config["batch_size"]
            )