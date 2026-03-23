from typing import Optional, cast

import torch
from torch import nn

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class GoogleDDPMFineTuner(nn.Module):
	"""Fine-tuning wrapper for `google/ddpm-cifar10-32`.

	This class is intentionally small and keeps compatibility with the repo's
	current training loop: `forward(images, labels=None)` returns a scalar loss,
	and `sample(num_samples, device=...)` returns images in [-1, 1].
	"""

	def __init__(
		self,
		latent_dim: int = 128,
		base_channels: int = 128,
		pretrained_model_id: str = "google/ddpm-cifar10-32",
		num_diffusion_steps: int = 1000,
		sample_steps: int = 100,
		**_unused,
	):
		super().__init__()
		del latent_dim
		del base_channels

		self.pretrained_model_id = pretrained_model_id
		self.sample_steps = int(sample_steps)

		# Some Hub repos store components at root, others under subfolders.
		# Try both layouts so this wrapper works across diffusers packaging styles.
		self.unet = self._load_unet(pretrained_model_id)
		self.scheduler = self._load_scheduler(pretrained_model_id)

		# Strict scheduler locking: training must use the scheduler's own timestep
		# configuration and beta schedule from the checkpoint.
		scheduler_steps = int(self.scheduler.config.get("num_train_timesteps", 1000))
		beta_schedule = str(self.scheduler.config.get("beta_schedule", "linear"))
		if beta_schedule != "linear":
			raise ValueError(
				f"Expected linear beta schedule, got '{beta_schedule}'. "
				"Use the original pretrained scheduler config for fine-tuning."
			)
		if int(num_diffusion_steps) != scheduler_steps:
			raise ValueError(
				f"num_diffusion_steps={num_diffusion_steps} does not match pretrained "
				f"scheduler timesteps={scheduler_steps}. Keep them identical for fine-tuning."
			)
		self.num_diffusion_steps = scheduler_steps

		# Partial fine-tuning: freeze encoder + bottleneck; adapt only decoder.
		self._freeze_all_then_enable_up_blocks()

	def _freeze_all_then_enable_up_blocks(self) -> None:
		for p in self.unet.parameters():
			p.requires_grad = False

		for p in self.unet.up_blocks.parameters():
			p.requires_grad = True

	@staticmethod
	def _load_unet(model_id: str) -> UNet2DModel:
		last_error: Optional[Exception] = None
		for subfolder in ("unet", None):
			try:
				if subfolder is None:
					return UNet2DModel.from_pretrained(model_id)
				return UNet2DModel.from_pretrained(model_id, subfolder=subfolder)
			except Exception as exc:
				last_error = exc
		raise OSError(
			f"Could not load UNet weights from '{model_id}' (tried subfolder='unet' and root)."
		) from last_error

	@staticmethod
	def _load_scheduler(model_id: str) -> DDPMScheduler:
		last_error: Optional[Exception] = None
		for subfolder in ("scheduler", None):
			try:
				if subfolder is None:
					return DDPMScheduler.from_pretrained(model_id)
				return DDPMScheduler.from_pretrained(model_id, subfolder=subfolder)
			except Exception as exc:
				last_error = exc
		raise OSError(
			f"Could not load scheduler config from '{model_id}' (tried subfolder='scheduler' and root)."
		) from last_error

	def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
		# This pretrained model is unconditional by design.
		del y

		batch_size = x.shape[0]
		device = x.device

		noise = torch.randn_like(x)
		t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device, dtype=torch.long)

		# Use scheduler's forward corruption to stay consistent with pretrained math.
		t_int = cast(torch.IntTensor, t)
		x_t = self.scheduler.add_noise(original_samples=x, noise=noise, timesteps=t_int)
		pred_noise = self.unet(x_t, t).sample
		return torch.nn.functional.mse_loss(pred_noise, noise)

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
		# Keep compatibility with the repo-wide sample API.
		del labels
		del guidance_scale
		del use_ddim

		self.unet.eval()

		height = int(self.unet.config.get("sample_size", 32))
		width = int(self.unet.config.get("sample_size", 32))
		in_channels = int(self.unet.config.get("in_channels", 3))

		x = torch.randn(num_samples, in_channels, height, width, device=device)

		steps = self.sample_steps if num_steps is None else int(num_steps)
		steps = max(1, min(steps, self.num_diffusion_steps))
		self.scheduler.set_timesteps(steps, device=device)

		for t in self.scheduler.timesteps:
			model_out = self.unet(x, t).sample
			step_out = self.scheduler.step(model_out, int(t.item()), x)
			x = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

		return x.clamp(-1.0, 1.0)
