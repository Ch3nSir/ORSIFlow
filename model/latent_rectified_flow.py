"""
Latent Rectified Flow: Combines VAE and Rectified Flow for latent space diffusion.

This module wraps the VAE encoder/decoder around the flow matching process,
enabling diffusion in a compressed latent space for improved efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from functools import partial

from tqdm import tqdm
from einops import repeat

from denoising_diffusion_pytorch.simple_diffusion import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    right_pad_dims_to,
    default,
    exists,
)
from model.vae import VAE, PretrainedVAEWrapper
from model.rectified_flow import CondRectifiedFlow, CondRectifiedFlowWithLogSNR


class LatentRectifiedFlow(nn.Module):
    """
    Latent Space Rectified Flow combining VAE and Flow Matching.
    
    Pipeline:
    - Training: x -> VAE.encode -> z -> flow.p_losses(z, v_target)
    - Sampling: noise -> flow.sample -> z_clean -> VAE.decode -> x
    
    Args:
        model: Velocity prediction network (UViT-like, operates on latents)
        vae: VAE model for encoding/decoding
        image_size: Original image size (before VAE encoding)
        channels: Number of image channels (before VAE)
        latent_channels: Number of latent channels (VAE output)
        cond_channels: Condition image channels
        extra_channels: Extra conditioning channels
        num_sample_steps: Number of ODE steps for sampling
        loss_type: Loss function type
        use_vae_loss: Whether to include VAE reconstruction loss
        kl_weight: Weight for KL divergence loss
        freeze_vae: Whether to freeze VAE weights
    """
    
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        *,
        image_size: int,
        channels: int = 1,
        latent_channels: int = 4,
        cond_channels: int = 3,
        extra_channels: int = 0,
        num_sample_steps: int = 10,
        loss_type: str = 'l2',
        use_vae_loss: bool = False,
        kl_weight: float = 0.0,
        freeze_vae: bool = True,
        clip_sample_denoised: bool = True,
    ):
        super().__init__()
        
        self.model = model
        self.vae = vae
        
        self.image_size = image_size
        self.channels = channels
        self.latent_channels = latent_channels
        self.cond_channels = cond_channels
        self.extra_channels = extra_channels
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        
        # Calculate latent spatial size
        self.downsample_factor = getattr(vae, 'downsample_factor', 4)
        self.latent_size = image_size // self.downsample_factor
        
        self.use_vae_loss = use_vae_loss
        self.kl_weight = kl_weight
        
        # Parse loss type
        if loss_type not in ['l2', 'l1', 'l1+l2', 'mean(l1, l2)']:
            try:
                from utils.import_utils import get_obj_from_str
                loss_type = get_obj_from_str(loss_type)
            except:
                raise NotImplementedError(f"Unknown loss type: {loss_type}")
        self.loss_type = loss_type
        
        # Freeze VAE if requested
        if freeze_vae:
            self._freeze_vae()
        
        # History for time ensemble
        self.history = []
    
    def _freeze_vae(self):
        """Freeze VAE parameters."""
        for param in self.vae.parameters():
            param.requires_grad = False
        print("VAE parameters frozen")
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space using VAE.
        
        Args:
            x: Image tensor (B, C, H, W), values in [0, 1] or [-1, 1]
        
        Returns:
            Latent tensor (B, latent_channels, H/f, W/f)
        """
        with torch.no_grad() if not self.vae.training else torch.enable_grad():
            z = self.vae.encode(x, deterministic=True)
        return z
    
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image space using VAE.
        
        Args:
            z: Latent tensor (B, latent_channels, H/f, W/f)
        
        Returns:
            Image tensor (B, C, H, W)
        """
        with torch.no_grad() if not self.vae.training else torch.enable_grad():
            x = self.vae.decode(z)
        return x
    
    def q_sample(
        self,
        z_start: torch.Tensor,
        z_noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear interpolation in latent space for flow matching.
        
        z_t = (1 - t) * z_0 + t * z_1
        """
        t = right_pad_dims_to(z_start, times)
        return (1 - t) * z_start + t * z_noise
    
    def get_velocity_target(
        self,
        z_start: torch.Tensor,
        z_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Get target velocity: v = z_noise - z_start"""
        return z_noise - z_start
    
    def p_losses(
        self,
        z_start: torch.Tensor,
        times: torch.Tensor,
        cond_img: torch.Tensor,
        z_seg: Optional[torch.Tensor] = None,
        extra_cond: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss in latent space.
        """
        noise = default(noise, lambda: torch.randn_like(z_start))
        
        # Use segmentation latent if provided
        z_target = z_seg if z_seg is not None else z_start
        
        # Interpolate
        z_t = self.q_sample(z_target, noise, times)
        
        # Target velocity
        v_target = self.get_velocity_target(z_target, noise)
        
        # Prepare model input
        if extra_cond is not None:
            model_input = torch.cat([z_t, extra_cond], dim=1)
        else:
            model_input = z_t
        
        # Predict velocity
        v_pred = self.model(model_input, times, cond_img)
        if v_pred.shape != v_target.shape:
            raise ValueError(f"Model output shape {tuple(v_pred.shape)} must match velocity target {tuple(v_target.shape)}")
        
        # Compute loss
        if self.loss_type == 'l2':
            return F.mse_loss(v_pred, v_target)
        elif self.loss_type == 'l1':
            return F.l1_loss(v_pred, v_target)
        elif self.loss_type == 'l1+l2':
            return F.mse_loss(v_pred, v_target) + F.l1_loss(v_pred, v_target)
        elif self.loss_type == 'mean(l1, l2)':
            return (F.mse_loss(v_pred, v_target) + F.l1_loss(v_pred, v_target)) / 2
        else:
            return self.loss_type(v_pred, v_target)
    
    def forward(
        self,
        img: torch.Tensor,
        cond_img: torch.Tensor,
        seg: Optional[torch.Tensor] = None,
        extra_cond: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Training forward pass.
        
        1. Encode image to latent space
        2. Compute flow matching loss in latent space
        
        Args:
            img: Ground truth image (B, C, H, W) in [0, 1]
            cond_img: Conditioning image (B, cond_C, H, W)
            seg: Optional segmentation mask
            extra_cond: Extra conditioning
        
        Returns:
            Loss tensor
        """
        b, channels, h, w = img.shape
        
        assert h == w == self.image_size, f"Expected size {self.image_size}, got {h}x{w}"
        
        # Normalize image to [-1, 1] for VAE encoding
        img_normalized = normalize_to_neg_one_to_one(img)
        
        # Encode to latent
        z = self.encode_to_latent(img_normalized)
        
        # Encode segmentation if provided
        z_seg = None
        if seg is not None:
            seg_normalized = normalize_to_neg_one_to_one(seg)
            z_seg = self.encode_to_latent(seg_normalized)
        
        # Prepare extra conditioning in latent space
        if extra_cond is not None:
            # Downsample extra_cond to latent resolution
            extra_cond_latent = F.interpolate(
                extra_cond, 
                size=(self.latent_size, self.latent_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            extra_cond_latent = torch.zeros(
                (b, self.extra_channels, self.latent_size, self.latent_size),
                device=self.device
            )
        
        # Random timesteps
        times = torch.zeros((b,), device=self.device).float().uniform_(0, 1)
        
        # Flow matching loss
        loss = self.p_losses(z, times, cond_img, z_seg, extra_cond_latent)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        cond_img: torch.Tensor,
        extra_cond: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples.
        
        1. Sample from flow in latent space
        2. Decode to image space
        
        Returns:
            Generated images (B, C, H, W) in [0, 1]
        """
        b, c, h, w = cond_img.shape
        
        # Prepare extra conditioning in latent space
        if extra_cond is not None:
            extra_cond_latent = F.interpolate(
                extra_cond,
                size=(self.latent_size, self.latent_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            extra_cond_latent = torch.zeros(
                (b, self.extra_channels, self.latent_size, self.latent_size),
                device=self.device
            )
        
        # Sample in latent space
        z = self.p_sample_loop(
            (b, self.latent_channels, self.latent_size, self.latent_size),
            cond_img,
            extra_cond_latent,
            verbose=verbose,
        )
        
        # Decode the sampled latent directly. The VAE decoder expects the same
        # scaled latent distribution produced by encode_to_latent.
        img = self.decode_from_latent(z)
        
        # Normalize output to [0, 1]
        img = unnormalize_to_zero_to_one(img.clamp(-1, 1))
        
        return img
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: tuple,
        cond_img: torch.Tensor,
        extra_cond: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        ODE sampling loop in latent space.
        """
        self.history = []
        
        # Start from noise
        z = torch.randn(shape, device=self.device)
        
        # Extract conditioning features
        conditioning_features = self.model.extract_features(cond_img)
        
        # Time steps from 1 to 0
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)
        
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step',
                      total=self.num_sample_steps, disable=not verbose):
            t = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t
            
            # Prepare input
            if extra_cond is not None:
                model_input = torch.cat([z, extra_cond], dim=1)
            else:
                model_input = z
            
            # Time embedding
            batch_t = repeat(t, ' -> b', b=shape[0])
            
            # Predict velocity
            v = self.model.sample_unet(model_input, batch_t, conditioning_features)
            
            # Euler step
            z = z + dt * v
            
            # Estimate clean latent for history
            z_start_estimate = z - t_next * v
            if self.clip_sample_denoised:
                z_start_estimate = z_start_estimate.clamp(-1., 1.)
            self.history.append(z_start_estimate)
        
        # Clamp only to keep the latent trajectory bounded; do not convert the
        # latent to image-space [0, 1] before passing it to the VAE decoder.
        if self.clip_sample_denoised:
            z = z.clamp(-1., 1.)
        
        return z


class LatentRectifiedFlowWithLogSNR(LatentRectifiedFlow):
    """
    Latent Rectified Flow with log_snr-based time conditioning.
    
    Maintains compatibility with existing UViT models that expect log_snr.
    """
    
    def __init__(self, *args, noise_schedule=None, noise_d=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Setup log_snr schedule
        from denoising_diffusion_pytorch.simple_diffusion import (
            logsnr_schedule_cosine,
            logsnr_schedule_shifted,
        )
        
        self.log_snr = noise_schedule if noise_schedule else logsnr_schedule_cosine
        
        if noise_d is not None:
            self.log_snr = logsnr_schedule_shifted(self.log_snr, self.latent_size, noise_d)
    
    def p_losses(self, z_start, times, cond_img, z_seg=None, extra_cond=None, noise=None):
        """Override to use log_snr for model conditioning."""
        noise = default(noise, lambda: torch.randn_like(z_start))
        
        z_target = z_seg if z_seg is not None else z_start
        z_t = self.q_sample(z_target, noise, times)
        v_target = self.get_velocity_target(z_target, noise)
        
        # Convert time to log_snr
        log_snr = self.log_snr(times)
        
        if extra_cond is not None:
            model_input = torch.cat([z_t, extra_cond], dim=1)
        else:
            model_input = z_t
        
        v_pred = self.model(model_input, log_snr, cond_img)
        if v_pred.shape != v_target.shape:
            raise ValueError(f"Model output shape {tuple(v_pred.shape)} must match velocity target {tuple(v_target.shape)}")
        
        if self.loss_type == 'l2':
            return F.mse_loss(v_pred, v_target)
        elif self.loss_type == 'l1':
            return F.l1_loss(v_pred, v_target)
        elif self.loss_type == 'l1+l2':
            return F.mse_loss(v_pred, v_target) + F.l1_loss(v_pred, v_target)
        elif self.loss_type == 'mean(l1, l2)':
            return (F.mse_loss(v_pred, v_target) + F.l1_loss(v_pred, v_target)) / 2
        else:
            return self.loss_type(v_pred, v_target)
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond_img, extra_cond, verbose=True):
        """Override to use log_snr for model conditioning."""
        self.history = []
        
        z = torch.randn(shape, device=self.device)
        conditioning_features = self.model.extract_features(cond_img)
        
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)
        
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step',
                      total=self.num_sample_steps, disable=not verbose):
            t = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t
            
            if extra_cond is not None:
                model_input = torch.cat([z, extra_cond], dim=1)
            else:
                model_input = z
            
            # Convert to log_snr
            log_snr = self.log_snr(t)
            batch_log_snr = repeat(log_snr, ' -> b', b=shape[0])
            
            v = self.model.sample_unet(model_input, batch_log_snr, conditioning_features)
            
            z = z + dt * v
            
            z_start_estimate = z - t_next * v
            if self.clip_sample_denoised:
                z_start_estimate = z_start_estimate.clamp(-1., 1.)
            self.history.append(z_start_estimate)
        
        if self.clip_sample_denoised:
            z = z.clamp(-1., 1.)
        
        return z


class ORSIFlow(LatentRectifiedFlowWithLogSNR):
    """Named alias matching the ORSIFlow paper configuration."""

    pass
