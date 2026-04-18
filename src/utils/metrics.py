from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

def _to_torch_images(images: np.ndarray) -> torch.Tensor:
    # This function converts a numpy array of images to a torch tensor, ensure (C x H x W) format and normalized to [0, 1].

    # If incorrect shape
    if images.ndim != 4:
        raise ValueError(f"Not a 4D image tensor! It's {images.shape}")

    # If already channel-first, just convert to tensor. If channel-last, permute to channel-first. Otherwise, raise an error.
    if images.shape[1] == 3:
        tensor = torch.from_numpy(images)
    elif images.shape[-1] == 3:
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Expected channel-first or channel-last RGB!! it's {images.shape}")

    # Ensure the tensor is float and normalized to [0, 1]
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    # Many training pipelines store images in [-1, 1]; convert back to [0, 1] for torchmetrics.
    if tensor.min() < 0.0:
        tensor = tensor.add(1.0).div(2.0)
    # Handle raw uint8-like arrays if present.
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def compute_fid_kid(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    device=None,
    batch_size: int = 64,
) -> Tuple[float, float, float]:

    if real_images.shape != fake_images.shape:
        raise ValueError(
            f"DIFFERENT REAL/FAKE SHAPES :/ {real_images.shape} vs {fake_images.shape}"
        )

    # Metrics run on selected device, while arrays stay on CPU until batch transfer.
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert real and fake images to torch tensors (keep on CPU, move per batch)
    real = _to_torch_images(real_images)
    fake = _to_torch_images(fake_images)

    # Initialize FID and KID metrics
    fid_metric = FrechetInceptionDistance(
        feature=2048,
        normalize=True
    ).to(dev)

    kid_metric = KernelInceptionDistance(
        subset_size=100,
        subsets=50,
        normalize=True,
    ).to(dev)


    # Update metrics in batches to avoid GPU OOM.
    def _update_in_batches(tensor: torch.Tensor, is_real: bool, label: str):
        with tqdm(total=tensor.shape[0], desc=label, unit="img") as pbar:
            for start in range(0, tensor.shape[0], batch_size):
                batch = tensor[start : start + batch_size].to(dev)
                fid_metric.update(batch, real=is_real)
                kid_metric.update(batch, real=is_real)
                pbar.update(batch.shape[0])

    _update_in_batches(real, True, "fid/kid real")
    _update_in_batches(fake, False, "fid/kid fake")

    # Compute FID and KID metrics and return them as floats.
    fid = float(fid_metric.compute().item())
    kid_mean, kid_std = kid_metric.compute()
    return fid, float(kid_mean.item()), float(kid_std.item())


def compute_inception_score(
    fake_images: np.ndarray,
    device=None,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """Compute Inception Score (mean, std) on fake images only."""
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake = _to_torch_images(fake_images)

    is_metric = InceptionScore(normalize=True).to(dev)

    with tqdm(total=fake.shape[0], desc="IS fake", unit="img") as pbar:
        for start in range(0, fake.shape[0], batch_size):
            batch = fake[start:start + batch_size].to(dev)
            is_metric.update(batch)
            pbar.update(batch.shape[0])

    is_mean, is_std = is_metric.compute()
    return float(is_mean.item()), float(is_std.item())
