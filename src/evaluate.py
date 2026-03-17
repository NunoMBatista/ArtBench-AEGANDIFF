import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from globals import ensure_repo_root

ensure_repo_root()

from src.models.VAE import VAE
from src.models.DCGAN import DCGAN
from src.utils.data_loader import get_dataloaders
from src.utils.metrics import compute_fid_kid
from src.utils.seed_setter import set_global_seed


SamplerFn = Callable[[int, torch.device], torch.Tensor]


@dataclass
class EvalConfig:
    seed: int = 11
    batch_size: int = 64
    num_workers: int = 2
    kaggle_root: str = "data"
    use_subset: bool = False
    subset_mode: str = "csv"
    subset_csv_path: str = "provided/student_start_pack/training_20_percent.csv"
    subset_seed: int = 11
    num_samples: int = 5000
    metrics_batch_size: int = 32
    checkpoint_path: str = ""
    latent_dim: int = 128
    base_channels: int = 64
    use_spectral_norm: bool = False
    run_prefix: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def update(self, path):
        config_path = os.path.join(os.path.dirname(path), "config.yml")
        with open(config_path, "r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f) or {}
        self.latent_dim = train_cfg.get("latent_dim", self.latent_dim)
        self.base_channels = train_cfg.get("base_channels", self.base_channels)
        self.use_spectral_norm = train_cfg.get("use_spectral_norm", self.use_spectral_norm)
        self.run_prefix = train_cfg.get("run_prefix", self.run_prefix)
        self.use_subset = train_cfg.get("use_subset", self.use_subset)
        self.subset_mode = train_cfg.get("subset_mode", self.subset_mode)
        self.subset_csv_path = train_cfg.get("subset_csv_path", self.subset_csv_path)
        self.subset_seed = train_cfg.get("subset_seed", self.subset_seed)
        self.kaggle_root = train_cfg.get("kaggle_root", self.kaggle_root)
        self.batch_size = train_cfg.get("batch_size", self.batch_size)
        self.num_workers = train_cfg.get("num_workers", self.num_workers)
        self.num_samples = train_cfg.get("num_samples", self.num_samples)
        self.metrics_batch_size = train_cfg.get("metrics_batch_size", self.metrics_batch_size)
        self.checkpoint_path = train_cfg.get("checkpoint_path", self.checkpoint_path)
        self.device = train_cfg.get("device", self.device)


def sample_real_images(config: EvalConfig) -> np.ndarray:
    # This function samples real images from the training set of ArtBench-10 according to the provided configuration.
    train_loader, _, _ = get_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_subset=config.use_subset,
        subset_mode=config.subset_mode,
        subset_csv_path=config.subset_csv_path,
        subset_seed=config.subset_seed,
        kaggle_root=config.kaggle_root,
        shuffle_train=True,
    )

    # Iterate through training data loader and collect images until we have enough samples.
    images = []
    total = 0
    with tqdm(total=config.num_samples, desc="real images", unit="img") as pbar:
        for batch, _ in train_loader:
            batch = batch.cpu().numpy()
            images.append(batch)
            total += batch.shape[0]
            pbar.update(batch.shape[0])
            if total >= config.num_samples:
                break

    # Concatenate collected batches and trim to the desired number of samples.
    images = np.concatenate(images, axis=0)[: config.num_samples]
    return images


def sample_fake_images(config: EvalConfig, sampler_fn: SamplerFn) -> np.ndarray:
    device = torch.device(config.device)
    samples = []

    # Sample fake images using the provided sampler function until we have enough samples.
    remaining = config.num_samples
    with tqdm(total=config.num_samples, desc="fake images", unit="img") as pbar:
        while remaining > 0:
            cur = min(config.batch_size, remaining)

            # Sample a batch of fake images using the sampler function. The sampler function takes the number of samples to generate and the device to use.
            with torch.no_grad():
                batch = sampler_fn(cur, device)

            samples.append(batch.cpu().numpy())
            remaining -= cur
            pbar.update(cur)
    return np.concatenate(samples, axis=0)


def _find_latest_checkpoint(model_type: str) -> str:
    """Search outputs/ for the most recent checkpoint matching model_type.
    
    Looks for:
      - outputs/run_<model_type>_*/model.pt
    Returns the newest checkpoint found.
    """
    outputs_dir = "outputs"
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError("outputs/ directory not found")

    prefix = f"run_{model_type}_"
    candidates = []
    for name in os.listdir(outputs_dir):
        if not name.startswith(prefix):
            continue
        ckpt_path = os.path.join(outputs_dir, name, "model.pt")
        if os.path.isfile(ckpt_path):
            candidates.append(ckpt_path)    

    if not candidates:
        raise FileNotFoundError(
            f"No {model_type} checkpoints found under outputs/{prefix}*"
        )

    return max(candidates, key=os.path.getmtime)


# Registry mapping model_type -> constructor
_MODEL_REGISTRY: Dict[str, type] = {
    "vae": VAE,
    "dcgan": DCGAN,
}


def _load_model(
    model_type: str,
    checkpoint_path: str,
    latent_dim: int,
    base_channels: int,
    device: torch.device,
    **extra_kwargs,
) -> torch.nn.Module:
    """Instantiate model_type, load checkpoint weights, set to eval mode.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    cls = _MODEL_REGISTRY[model_type]
    model = cls(latent_dim=latent_dim, base_channels=base_channels, **extra_kwargs).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def evaluate(config: EvalConfig, sampler_fn: SamplerFn) -> Tuple[float, float, float]:
    set_global_seed(config.seed)

    # get real and fake images
    real_images = sample_real_images(config)
    fake_images = sample_fake_images(config, sampler_fn)

    # compute metrics
    fid, kid_mean, kid_std = compute_fid_kid(
        real_images,
        fake_images,
        device=config.device,
        batch_size=config.metrics_batch_size,
    )
    return fid, kid_mean, kid_std


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a generative model (FID / KID).")
    parser.add_argument(
        "model_type",
        type=str,
        choices=list(_MODEL_REGISTRY.keys()),
        help="Model architecture to evaluate (e.g. vae, dcgan).",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="",
        help="Path to a specific checkpoint. If omitted, the latest one is used.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    model_type = args.model_type.lower()

    config = EvalConfig()
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint

    ckpt_path = config.checkpoint_path or _find_latest_checkpoint(model_type)
    config.update(ckpt_path)
    device = torch.device(config.device)

    extra_kwargs = {}
    if model_type == "dcgan":
        extra_kwargs["use_spectral_norm"] = config.use_spectral_norm

    model = _load_model(
        model_type,
        ckpt_path,
        config.latent_dim,
        config.base_channels,
        device,
        **extra_kwargs,
    )

    # All models implement .sample(num_samples, device)
    def sampler(num_samples: int, device: torch.device) -> torch.Tensor:
        return model.sample(num_samples, device)

    fid, kid_mean, kid_std = evaluate(config, sampler)
    print({"fid": fid, "kid_mean": kid_mean, "kid_std": kid_std})


if __name__ == "__main__":
    main()
