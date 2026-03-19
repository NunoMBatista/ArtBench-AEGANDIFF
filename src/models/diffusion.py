import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
	"""Create sinusoidal embeddings for diffusion timesteps."""
	half_dim = dim // 2
	if half_dim == 0:
		return timesteps.float().unsqueeze(-1)

	exponent = -math.log(10000.0) / max(1, half_dim - 1)
	freqs = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
	args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
	emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
	if dim % 2 == 1:
		emb = F.pad(emb, (0, 1))
	return emb


class ResidualBlock(nn.Module):
	"""A minimal residual block conditioned by time and class embeddings."""

	def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.norm1 = nn.GroupNorm(8, in_channels)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		self.norm2 = nn.GroupNorm(8, out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

		self.cond_proj = nn.Linear(cond_dim, out_channels)
		self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		h = self.conv1(F.silu(self.norm1(x)))

		# Broadcast conditioning to all spatial locations.
		cond_bias = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
		h = h + cond_bias

		h = self.conv2(F.silu(self.norm2(h)))
		return h + self.skip(x)


class TinyUNet(nn.Module):
	"""Small UNet designed for 32x32 images used in ArtBench-10."""

	def __init__(self, in_channels: int, base_channels: int, cond_dim: int, use_attention: bool = False):
		super().__init__()
		c = base_channels
		self.use_attention = use_attention

		self.in_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

		self.down1 = ResidualBlock(c, c, cond_dim)
		self.downsample1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)

		self.down2 = ResidualBlock(c * 2, c * 2, cond_dim)
		self.downsample2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)

		self.mid = ResidualBlock(c * 4, c * 4, cond_dim)
		self.mid_attn = SelfAttention2d(c * 4) if use_attention else nn.Identity()

		self.upsample1 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
		self.up1 = ResidualBlock(c * 4, c * 2, cond_dim)

		self.upsample2 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
		self.up2 = ResidualBlock(c * 2, c, cond_dim)

		self.out_norm = nn.GroupNorm(8, c)
		self.out_conv = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		x0 = self.in_conv(x)

		d1 = self.down1(x0, cond)
		d2_in = self.downsample1(d1)

		d2 = self.down2(d2_in, cond)
		mid_in = self.downsample2(d2)

		mid = self.mid(mid_in, cond)
		mid = self.mid_attn(mid)

		u1_in = self.upsample1(mid)
		u1 = self.up1(torch.cat([u1_in, d2], dim=1), cond)

		u2_in = self.upsample2(u1)
		u2 = self.up2(torch.cat([u2_in, d1], dim=1), cond)

		return self.out_conv(F.silu(self.out_norm(u2)))


class SelfAttention2d(nn.Module):
	"""Minimal self-attention block over spatial tokens."""

	def __init__(self, channels: int, num_heads: int = 4):
		super().__init__()
		if channels % num_heads != 0:
			raise ValueError("channels must be divisible by num_heads")
		self.channels = channels
		self.num_heads = num_heads
		self.head_dim = channels // num_heads

		self.norm = nn.GroupNorm(8, channels)
		self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
		self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b, c, h, w = x.shape
		h_in = x
		x = self.norm(x)
		qkv = self.qkv(x)
		q, k, v = torch.chunk(qkv, 3, dim=1)

		# Flatten spatial dims into tokens and apply multi-head scaled dot-product attention.
		q = q.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
		k = k.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
		v = v.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

		attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
		out = torch.matmul(attn, v)

		out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
		out = self.proj_out(out)
		return h_in + out


class DiffusionModel(nn.Module):
	"""
	Minimal class-conditional DDPM with classifier-free guidance and DDIM sampling.

	The model predicts noise epsilon and uses MSE training objective.
	"""

	def __init__(
		self,
		latent_dim: int = 128,
		base_channels: int = 128,
		img_channels: int = 3,
		image_size: int = 32,
		num_classes: int = 10,
		num_diffusion_steps: int = 1000,
		cfg_dropout: float = 0.1,
		sample_steps: int = 100,
		guidance_scale: float = 2.0,
		class_conditional: bool = True,
		use_attention: bool = False,
		**_unused,
	):
		super().__init__()
		del latent_dim  # Kept in signature to stay constructor-compatible with existing factories.

		self.image_size = image_size
		self.img_channels = img_channels
		self.num_classes = num_classes
		self.num_diffusion_steps = int(num_diffusion_steps)
		self.cfg_dropout = float(cfg_dropout)
		self.sample_steps = int(sample_steps)
		self.guidance_scale = float(guidance_scale)
		self.class_conditional = bool(class_conditional)
		self.use_attention = bool(use_attention)

		cond_dim = base_channels * 4
		self.time_mlp = nn.Sequential(
			nn.Linear(cond_dim, cond_dim),
			nn.SiLU(),
			nn.Linear(cond_dim, cond_dim),
		)

		# Extra class index is the null token used for classifier-free guidance.
		self.null_class_idx = num_classes
		self.class_emb = nn.Embedding(num_classes + 1, cond_dim)
		self.unet = TinyUNet(
			in_channels=img_channels,
			base_channels=base_channels,
			cond_dim=cond_dim,
			use_attention=self.use_attention,
		)

		betas = torch.linspace(1e-4, 2e-2, self.num_diffusion_steps, dtype=torch.float32)
		alphas = 1.0 - betas
		alphas_cumprod = torch.cumprod(alphas, dim=0)
		alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

		# Precompute diffusion coefficients once and keep them on the same device as the model.
		self.register_buffer("betas", betas)
		self.register_buffer("alphas", alphas)
		self.register_buffer("alphas_cumprod", alphas_cumprod)
		self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
		self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
		self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
		self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

		posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
		self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))

	def _gather(self, arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
		# Gather per-timestep scalars and reshape for broadcast over (C, H, W).
		out = arr.gather(0, t)
		while out.ndim < len(x_shape):
			out = out.unsqueeze(-1)
		return out

	def _build_cond(self, t: torch.Tensor, y: Optional[torch.Tensor], class_drop_prob: float = 0.0) -> torch.Tensor:
		time_emb = _sinusoidal_timestep_embedding(t, self.class_emb.embedding_dim)
		time_emb = self.time_mlp(time_emb)

		# In unconditional mode, always use the null class embedding.
		if not self.class_conditional:
			y = None

		if y is None:
			y_idx = torch.full_like(t, self.null_class_idx)
		else:
			y_idx = y.clone()
			if class_drop_prob > 0.0:
				# Randomly drop class conditioning for classifier-free guidance training.
				drop_mask = torch.rand_like(y_idx.float()) < class_drop_prob
				y_idx[drop_mask] = self.null_class_idx

		class_emb = self.class_emb(y_idx)
		return time_emb + class_emb

	def predict_noise(
		self,
		x_t: torch.Tensor,
		t: torch.Tensor,
		y: Optional[torch.Tensor] = None,
		class_drop_prob: float = 0.0,
	) -> torch.Tensor:
		cond = self._build_cond(t, y, class_drop_prob=class_drop_prob)
		return self.unet(x_t, cond)

	def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
		if noise is None:
			noise = torch.randn_like(x_start)
		sqrt_alpha_bar = self._gather(self.sqrt_alphas_cumprod, t, x_start.shape)
		sqrt_one_minus_alpha_bar = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
		# Forward process: progressively corrupt x_0 into x_t.
		return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

	def p_losses(self, x_start: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
		batch_size = x_start.size(0)
		# Sample random timesteps so each batch trains denoising at different noise levels.
		t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=x_start.device)
		noise = torch.randn_like(x_start)
		x_t = self.q_sample(x_start, t, noise)

		pred_noise = self.predict_noise(x_t, t, y=y, class_drop_prob=self.cfg_dropout)
		return F.mse_loss(pred_noise, noise)

	def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
		return self.p_losses(x, y)

	def _predict_eps_with_cfg(self, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor], guidance_scale: float) -> torch.Tensor:
		if not self.class_conditional:
			# CFG only makes sense with conditional and unconditional branches.
			return self.predict_noise(x_t, t, y=None, class_drop_prob=0.0)

		if guidance_scale <= 1.0:
			return self.predict_noise(x_t, t, y=y, class_drop_prob=0.0)

		# Classifier-free guidance: combine conditional and unconditional predictions.
		eps_cond = self.predict_noise(x_t, t, y=y, class_drop_prob=0.0)
		eps_uncond = self.predict_noise(x_t, t, y=None, class_drop_prob=0.0)
		return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

	@torch.no_grad()
	def sample(
		self,
		num_samples: int,
		device: torch.device,
		labels: Optional[torch.Tensor] = None,
		guidance_scale: Optional[float] = None,
		num_steps: Optional[int] = None,
		use_ddim: bool = True,
	) -> torch.Tensor:
		guidance = self.guidance_scale if guidance_scale is None else float(guidance_scale)
		sampling_steps = self.sample_steps if num_steps is None else int(num_steps)
		sampling_steps = max(1, min(sampling_steps, self.num_diffusion_steps))

		x = torch.randn(num_samples, self.img_channels, self.image_size, self.image_size, device=device)
		if not self.class_conditional:
			labels = None
		elif labels is None:
			labels = torch.randint(0, self.num_classes, (num_samples,), device=device)

		if use_ddim:
			# Uniformly subsample timesteps for faster deterministic DDIM inference.
			step_indices = torch.linspace(
				self.num_diffusion_steps - 1,
				0,
				sampling_steps,
				device=device,
			).long()
		else:
			step_indices = torch.arange(self.num_diffusion_steps - 1, -1, -1, device=device)

		for i, t_scalar in enumerate(step_indices):
			t = torch.full((num_samples,), int(t_scalar.item()), device=device, dtype=torch.long)
			eps = self._predict_eps_with_cfg(x, t, labels, guidance)

			# Reconstruct x_0 estimate from current x_t and predicted noise.
			alpha_bar_t = self._gather(self.alphas_cumprod, t, x.shape)
			sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
			sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
			x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t.clamp(min=1e-8)

			if use_ddim:
				if i == len(step_indices) - 1:
					alpha_bar_next = torch.ones_like(alpha_bar_t)
				else:
					next_t = torch.full((num_samples,), int(step_indices[i + 1].item()), device=device, dtype=torch.long)
					alpha_bar_next = self._gather(self.alphas_cumprod, next_t, x.shape)

				# Deterministic DDIM update (eta=0) for fast and stable sampling.
				x = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * eps
			else:
				# Ancestral DDPM step adds stochasticity through posterior variance noise.
				alpha_t = self._gather(self.alphas, t, x.shape)
				beta_t = self._gather(self.betas, t, x.shape)
				sqrt_recip_alpha_t = self._gather(self.sqrt_recip_alphas, t, x.shape)
				mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps)

				if int(t_scalar.item()) > 0:
					noise = torch.randn_like(x)
					var = self._gather(self.posterior_variance, t, x.shape)
					x = mean + torch.sqrt(var) * noise
				else:
					x = mean

		return x.clamp(-1.0, 1.0)