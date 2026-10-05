"""
VAE (Variational Autoencoder) module for latent space diffusion.
Provides a wrapper for pretrained VAE models (e.g., from Stable Diffusion).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiagonalGaussianDistribution:
    """Gaussian distribution with diagonal covariance for VAE reparameterization."""
    
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        
    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.mean)
    
    def kl(self, other: Optional['DiagonalGaussianDistribution'] = None) -> torch.Tensor:
        """Compute KL divergence against standard normal or another distribution."""
        if other is None:
            # KL divergence against N(0, 1)
            return 0.5 * torch.sum(
                self.mean ** 2 + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3]
            )
        else:
            return 0.5 * torch.sum(
                (self.mean - other.mean) ** 2 / other.var
                + self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3]
            )
    
    def mode(self) -> torch.Tensor:
        return self.mean


class ResBlock(nn.Module):
    """Residual block for VAE encoder/decoder."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    """Downsample with strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample with interpolation + convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class VAEEncoder(nn.Module):
    """
    VAE Encoder: maps input images to latent distribution parameters.
    
    Args:
        in_channels: Number of input image channels
        latent_channels: Number of latent space channels (output will be 2x for mean+logvar)
        hidden_channels: Base hidden channel dimension
        num_res_blocks: Number of residual blocks per resolution
        downsample_factor: Total spatial downsampling factor (e.g., 4 means H/4, W/4)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        hidden_channels: int = 64,
        num_res_blocks: int = 2,
        downsample_factor: int = 4,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        
        # Calculate number of downsampling steps
        import math
        num_downsamples = int(math.log2(downsample_factor))
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = hidden_channels
        for i in range(num_downsamples):
            out_ch = ch * 2 if i < num_downsamples - 1 else ch
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResBlock(ch, out_ch))
                ch = out_ch
            block.append(Downsample(ch))
            self.down_blocks.append(block)
        
        # Middle blocks
        self.mid_block1 = ResBlock(ch, ch)
        self.mid_block2 = ResBlock(ch, ch)
        
        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, latent_channels * 2, 3, padding=1)  # mean + logvar
    
    def forward(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.conv_in(x)
        
        for blocks in self.down_blocks:
            for block in blocks:
                h = block(h)
        
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        
        return DiagonalGaussianDistribution(h)


class VAEDecoder(nn.Module):
    """
    VAE Decoder: maps latent vectors back to image space.
    
    Args:
        out_channels: Number of output image channels
        latent_channels: Number of latent space channels
        hidden_channels: Base hidden channel dimension
        num_res_blocks: Number of residual blocks per resolution
        upsample_factor: Total spatial upsampling factor (e.g., 4 means 4x upscale)
    """
    
    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 4,
        hidden_channels: int = 64,
        num_res_blocks: int = 2,
        upsample_factor: int = 4,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        
        # Calculate number of upsampling steps
        import math
        num_upsamples = int(math.log2(upsample_factor))
        
        # Start with higher channel count (matching encoder's middle)
        ch = hidden_channels * (2 ** (num_upsamples - 1)) if num_upsamples > 1 else hidden_channels
        
        # Input from latent
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)
        
        # Middle blocks
        self.mid_block1 = ResBlock(ch, ch)
        self.mid_block2 = ResBlock(ch, ch)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(num_upsamples):
            out_ch = ch // 2 if i < num_upsamples - 1 else hidden_channels
            block = nn.ModuleList()
            block.append(Upsample(ch))
            for _ in range(num_res_blocks):
                block.append(ResBlock(ch, out_ch))
                ch = out_ch
            self.up_blocks.append(block)
        
        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        
        for blocks in self.up_blocks:
            for block in blocks:
                h = block(h)
        
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        
        return h


class VAE(nn.Module):
    """
    Complete VAE model combining encoder and decoder.
    
    Args:
        in_channels: Number of input/output image channels
        latent_channels: Number of latent space channels
        hidden_channels: Base hidden channel dimension
        downsample_factor: Spatial compression factor
        pretrained_path: Optional path to pretrained weights
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        hidden_channels: int = 64,
        downsample_factor: int = 4,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        
        self.encoder = VAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            downsample_factor=downsample_factor,
        )
        
        self.decoder = VAEDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            upsample_factor=downsample_factor,
        )
        
        # Scaling factor for latent space (like SD's 0.18215)
        self.scale_factor = 0.18215
        
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
    
    def load_pretrained(self, path: str):
        """Load pretrained weights."""
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded VAE weights from {path}")
    
    def encode(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            x: Input image tensor (B, C, H, W), expected in [0, 1] or [-1, 1]
            deterministic: If True, use mean instead of sampling
        
        Returns:
            Latent tensor (B, latent_channels, H/f, W/f)
        """
        posterior = self.encoder(x)
        if deterministic:
            z = posterior.mode()
        else:
            z = posterior.sample()
        return z * self.scale_factor
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image space.
        
        Args:
            z: Latent tensor (B, latent_channels, H/f, W/f)
        
        Returns:
            Reconstructed image (B, C, H, W)
        """
        z = z / self.scale_factor
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """
        Full forward pass (encode + decode).
        
        Returns:
            Tuple of (reconstruction, posterior distribution)
        """
        posterior = self.encoder(x)
        z = posterior.sample()
        recon = self.decoder(z)
        return recon, posterior


class PretrainedVAEWrapper(nn.Module):
    """
    Wrapper for using Stable Diffusion's pretrained VAE.
    
    This allows using the powerful SD VAE while adapting input/output channels
    if needed (e.g., for single-channel segmentation masks).
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.downsample_factor = 8  # SD VAE uses 8x compression
        
        # Channel adapters (if input is not 3-channel RGB)
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, 1)
            self.output_adapter = nn.Conv2d(3, in_channels, 1)
        else:
            self.input_adapter = nn.Identity()
            self.output_adapter = nn.Identity()
        
        # We'll use the simple VAE implementation for now
        # In practice, you'd load the actual SD VAE here
        self.vae = VAE(
            in_channels=3,
            latent_channels=latent_channels,
            hidden_channels=128,
            downsample_factor=8,
        )
        
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        
        if freeze:
            self.freeze_vae()
    
    def load_pretrained(self, path: str):
        """Load pretrained SD VAE weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            # Handle different checkpoint formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.vae.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained VAE from {path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained VAE: {e}")
    
    def freeze_vae(self):
        """Freeze VAE parameters (only train adapters)."""
        for param in self.vae.parameters():
            param.requires_grad = False
        # Keep adapters trainable
        for param in self.input_adapter.parameters():
            param.requires_grad = True
        for param in self.output_adapter.parameters():
            param.requires_grad = True
    
    def encode(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Encode with channel adaptation."""
        x = self.input_adapter(x)
        return self.vae.encode(x, deterministic=deterministic)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode with channel adaptation."""
        x = self.vae.decode(z)
        return self.output_adapter(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode pass."""
        z = self.encode(x)
        return self.decode(z)
