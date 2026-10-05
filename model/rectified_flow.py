"""
Rectified Flow (Flow Matching) diffusion module.

Implements the Rectified Flow formulation where:
- Training: predict velocity v = x_1 - x_0 (noise - clean)
- Sampling: ODE integration from noise to clean data

Reference: Flow Matching for Generative Modeling (Lipman et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sqrt
from torch.special import expm1
from typing import Optional, Callable, Tuple, Union
from functools import partial

from tqdm import tqdm
from einops import rearrange, repeat

from denoising_diffusion_pytorch.simple_diffusion import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    right_pad_dims_to,
    default,
    exists,
)


class CondRectifiedFlow(nn.Module):
    """
    Conditional Rectified Flow for latent space diffusion.
    
    Uses flow matching objective to predict velocity field v(x_t, t).
    Training: v_target = x_1 - x_0 where x_t = (1-t)*x_0 + t*x_1
    Sampling: ODE integration from noise x_1 to clean data x_0
    
    Args:
        model: Velocity prediction network (UViT-like)
        image_size: Spatial size of input (can be latent size for latent diffusion)
        channels: Number of channels
        extra_channels: Extra conditioning channels concatenated with input
        cond_channels: Condition image channels
        num_sample_steps: Number of ODE integration steps for sampling
        loss_type: 'l2' or 'l1' loss for training
    """
    
    def __init__(
        self,
        model: nn.Module,
        *,
        image_size: int,
        channels: int = 4,
        extra_channels: int = 0,
        cond_channels: int = 3,
        num_sample_steps: int = 10,
        loss_type: str = 'l2',
        clip_sample_denoised: bool = True,
    ):
        super().__init__()
        
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.extra_channels = extra_channels
        self.cond_channels = cond_channels
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        
        if loss_type not in ['l2', 'l1', 'l1+l2', 'mean(l1, l2)']:
            try:
                from utils.import_utils import get_obj_from_str
                loss_type = get_obj_from_str(loss_type)
            except:
                raise NotImplementedError(f"Unknown loss type: {loss_type}")
        
        self.loss_type = loss_type
        
        # History for time ensemble during sampling
        self.history = []
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        x_noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear interpolation between clean data and noise.
        
        x_t = (1 - t) * x_0 + t * x_1
        
        Args:
            x_start: Clean data x_0
            x_noise: Noise x_1
            times: Time t in [0, 1], where t=0 is clean and t=1 is noise
        
        Returns:
            Interpolated sample x_t
        """
        # Expand time dimensions to match spatial
        t = right_pad_dims_to(x_start, times)
        return (1 - t) * x_start + t * x_noise
    
    def get_velocity_target(
        self,
        x_start: torch.Tensor,
        x_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the target velocity for flow matching.
        
        v_target = x_1 - x_0 = noise - clean
        
        This is the optimal transport velocity field that transforms
        the data distribution to the noise distribution.
        """
        return x_noise - x_start
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        times: torch.Tensor,
        cond_img: torch.Tensor,
        seg: Optional[torch.Tensor] = None,
        extra_cond: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            x_start: Clean data (ground truth latent)
            times: Random timesteps in [0, 1]
            cond_img: Conditioning image
            seg: Optional segmentation for modification task
            extra_cond: Extra conditioning channels
            noise: Optional pre-generated noise
        
        Returns:
            Loss tensor
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # If seg is provided, use it as the target (modification task)
        if seg is not None:
            x_target = seg
        else:
            x_target = x_start
        
        # Interpolate to get x_t
        x_t = self.q_sample(x_target, noise, times)
        
        # Get target velocity
        v_target = self.get_velocity_target(x_target, noise)
        
        # Prepare input with extra conditioning
        if extra_cond is not None:
            model_input = torch.cat([x_t, extra_cond], dim=1)
        else:
            model_input = x_t
        
        # Model predicts velocity
        # Note: we use times directly instead of log_snr for flow matching
        v_pred = self.model(model_input, times, cond_img)
        
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
            # Custom loss function
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
        
        Args:
            img: Input image/latent (ground truth)
            cond_img: Conditioning image
            seg: Optional segmentation mask
            extra_cond: Extra conditioning
        
        Returns:
            Loss tensor
        """
        b, channels, h, w = img.shape
        cond_channels = cond_img.shape[1]
        
        assert channels == self.channels, f"Expected {self.channels} channels, got {channels}"
        assert cond_channels == self.cond_channels, f"Expected {self.cond_channels} cond channels, got {cond_channels}"
        
        # Normalize to [-1, 1] for consistency
        img = normalize_to_neg_one_to_one(img)
        seg = normalize_to_neg_one_to_one(seg) if seg is not None else None
        
        # Default extra conditioning
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        
        # Random timesteps uniformly in [0, 1]
        times = torch.zeros((b,), device=self.device).float().uniform_(0, 1)
        
        return self.p_losses(img, times, cond_img, seg, extra_cond, *args, **kwargs)
    
    @torch.no_grad()
    def sample(
        self,
        cond_img: torch.Tensor,
        extra_cond: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using ODE integration.
        
        Args:
            cond_img: Conditioning image
            extra_cond: Extra conditioning
            verbose: Show progress bar
        
        Returns:
            Generated samples
        """
        b, c, h, w = cond_img.shape
        
        # Calculate latent size (assuming conditioning is at full resolution)
        latent_size = self.image_size
        
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, latent_size, latent_size), device=self.device))
        
        return self.p_sample_loop(
            (b, self.channels, latent_size, latent_size),
            cond_img,
            extra_cond=extra_cond,
            verbose=verbose,
        )
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        cond_img: torch.Tensor,
        extra_cond: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        ODE sampling loop using Euler method.
        
        Integrates from t=1 (noise) to t=0 (clean data).
        
        dx/dt = -v(x_t, t)  (negative because we go from noise to data)
        x_{t-dt} = x_t - dt * v(x_t, t)
        """
        self.history = []
        
        # Start from pure noise at t=1
        x = torch.randn(shape, device=self.device)
        
        # Extract conditioning features once
        conditioning_features = self.model.extract_features(cond_img)
        
        # Time steps from 1 to 0
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)
        
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', 
                      total=self.num_sample_steps, disable=not verbose):
            t = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t  # Negative step size (going from 1 to 0)
            
            # Prepare input
            if extra_cond is not None:
                model_input = torch.cat([x, extra_cond], dim=1)
            else:
                model_input = x
            
            # Get time embedding (repeat for batch)
            batch_t = repeat(t, ' -> b', b=shape[0])
            
            # Predict velocity
            v = self.model.sample_unet(model_input, batch_t, conditioning_features)
            
            # Euler step: x_{t+dt} = x_t + dt * v
            # Since dt is negative (going from 1 to 0), this moves toward clean data
            x = x + dt * v
            
            # Estimate x_0 for history (for time ensemble)
            # At time t, x_0 ≈ x_t - t * v (linear extrapolation to t=0)
            x_start_estimate = x - t_next * v
            if self.clip_sample_denoised:
                x_start_estimate = x_start_estimate.clamp(-1., 1.)
            self.history.append(x_start_estimate)
        
        # Final output
        if self.clip_sample_denoised:
            x = x.clamp(-1., 1.)
        
        x = unnormalize_to_zero_to_one(x)
        return x


class CondRectifiedFlowWithLogSNR(CondRectifiedFlow):
    """
    Rectified Flow variant that maintains compatibility with log_snr-based time conditioning.
    
    This version converts the linear time t to log_snr for the model,
    maintaining compatibility with existing UViT architectures trained with DDPM.
    """
    
    def __init__(self, *args, time_to_logsnr: Optional[Callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Default: use cosine schedule for log_snr conversion
        if time_to_logsnr is None:
            from denoising_diffusion_pytorch.simple_diffusion import logsnr_schedule_cosine
            self.time_to_logsnr = logsnr_schedule_cosine
        else:
            self.time_to_logsnr = time_to_logsnr
    
    def p_losses(self, x_start, times, cond_img, seg=None, extra_cond=None, noise=None, *args, **kwargs):
        """Override to use log_snr for model conditioning."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        if seg is not None:
            x_target = seg
        else:
            x_target = x_start
        
        # Linear interpolation for flow matching
        x_t = self.q_sample(x_target, noise, times)
        
        # Target velocity
        v_target = self.get_velocity_target(x_target, noise)
        
        # Convert time to log_snr for model conditioning
        log_snr = self.time_to_logsnr(times)
        
        if extra_cond is not None:
            model_input = torch.cat([x_t, extra_cond], dim=1)
        else:
            model_input = x_t
        
        # Model predicts velocity, conditioned on log_snr
        v_pred = self.model(model_input, log_snr, cond_img)
        
        # Loss
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
        """Override to use log_snr for model conditioning during sampling."""
        self.history = []
        
        x = torch.randn(shape, device=self.device)
        conditioning_features = self.model.extract_features(cond_img)
        
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)
        
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', 
                      total=self.num_sample_steps, disable=not verbose):
            t = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t
            
            if extra_cond is not None:
                model_input = torch.cat([x, extra_cond], dim=1)
            else:
                model_input = x
            
            # Convert to log_snr for model
            log_snr = self.time_to_logsnr(t)
            batch_log_snr = repeat(log_snr, ' -> b', b=shape[0])
            
            v = self.model.sample_unet(model_input, batch_log_snr, conditioning_features)
            
            x = x + dt * v
            
            x_start_estimate = x - t_next * v
            if self.clip_sample_denoised:
                x_start_estimate = x_start_estimate.clamp(-1., 1.)
            self.history.append(x_start_estimate)
        
        if self.clip_sample_denoised:
            x = x.clamp(-1., 1.)
        
        x = unnormalize_to_zero_to_one(x)
        return x
