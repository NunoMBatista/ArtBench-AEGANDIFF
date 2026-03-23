import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from globals import ensure_repo_root

ensure_repo_root()
load_dotenv()

from src.models.VAE import VAE
from src.models.DCGAN import DCGAN
from src.models.cGAN import cGAN
from src.models.diffusion import DiffusionModel
from src.models.google_DDPM import GoogleDDPMFineTuner
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
    image_size: int = 32
    img_channels: int = 3
    num_classes: int = 10
    num_diffusion_steps: int = 1000
    cfg_dropout: float = 0.1
    sample_steps: int = 100
    guidance_scale: float = 2.0
    class_conditional: bool = True
    use_attention: bool = False
    pretrained_model_id: str = "google/ddpm-cifar10-32"
    disable_attention_on_cpu: bool = True
    wandb: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    run_prefix: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def update(self, path):
        # Evaluation reuses training config from the run folder to avoid mismatch.
        config_path = os.path.join(os.path.dirname(path), "config.yml")
        with open(config_path, "r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f) or {}
        self.latent_dim = train_cfg.get("latent_dim", self.latent_dim)
        self.base_channels = train_cfg.get("base_channels", self.base_channels)
        self.use_spectral_norm = train_cfg.get("use_spectral_norm", self.use_spectral_norm)
        self.image_size = train_cfg.get("image_size", self.image_size)
        self.img_channels = train_cfg.get("img_channels", self.img_channels)
        self.num_classes = train_cfg.get("num_classes", self.num_classes)
        self.num_diffusion_steps = train_cfg.get("num_diffusion_steps", self.num_diffusion_steps)
        self.cfg_dropout = train_cfg.get("cfg_dropout", self.cfg_dropout)
        self.sample_steps = train_cfg.get("sample_steps", self.sample_steps)
        self.guidance_scale = train_cfg.get("guidance_scale", self.guidance_scale)
        self.class_conditional = train_cfg.get("class_conditional", self.class_conditional)
        self.use_attention = train_cfg.get("use_attention", self.use_attention)
        self.pretrained_model_id = train_cfg.get("pretrained_model_id", self.pretrained_model_id)
        self.disable_attention_on_cpu = train_cfg.get("disable_attention_on_cpu", self.disable_attention_on_cpu)
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
        self.wandb = train_cfg.get("wandb", self.wandb)


def _init_wandb_eval(config: EvalConfig, model_type: str, checkpoint_path: str):
    # Evaluation logging mirrors training behavior but stays optional.
    wandb_cfg = config.wandb or {}
    if not wandb_cfg.get("enabled", False):
        return None
    if wandb is None:
        print("WARNING: wandb is enabled in config but the package is not installed. Continuing without wandb.")
        return None

    entity = wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
    project = wandb_cfg.get("project") or os.getenv("WANDB_PROJECT") or "ArtBench-AEGANDIFF"
    run_name = wandb_cfg.get("eval_run_name") or f"eval-{model_type}-{os.path.basename(os.path.dirname(checkpoint_path))}"

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config={
                "model_type": model_type,
                "checkpoint_path": checkpoint_path,
                "num_samples": config.num_samples,
                "metrics_batch_size": config.metrics_batch_size,
                "device": config.device,
            },
            tags=wandb_cfg.get("tags", ["evaluation", model_type]),
            notes=wandb_cfg.get("notes", ""),
        )
        return run
    except Exception as e:
        print(f"WARNING: failed to initialize wandb for evaluation ({e}). Continuing without wandb.")
        return None


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

    # Pick newest checkpoint so "evaluate <model_type>" works without manual paths.
    return max(candidates, key=os.path.getmtime)


# Registry mapping model_type -> constructor
_MODEL_REGISTRY: Dict[str, type] = {
    "vae": VAE,
    "dcgan": DCGAN,
    "cgan": cGAN,
    "diffusion": DiffusionModel,
    "google_ddpm": GoogleDDPMFineTuner,
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

    # Metrics are computed on matched real/fake sample counts.
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

    # Prefer explicit checkpoint, otherwise fall back to latest run artifact.
    ckpt_path = config.checkpoint_path or _find_latest_checkpoint(model_type)
    config.update(ckpt_path)
    device = torch.device(config.device)
    wandb_run = _init_wandb_eval(config, model_type, ckpt_path)

    extra_kwargs = {}
    if model_type == "dcgan":
        extra_kwargs["use_spectral_norm"] = config.use_spectral_norm
    elif model_type == "cgan":
        extra_kwargs["num_classes"] = config.num_classes
        extra_kwargs["use_spectral_norm"] = config.use_spectral_norm
    elif model_type == "diffusion":
        extra_kwargs.update(
            {
                "image_size": config.image_size,
                "img_channels": config.img_channels,
                "num_classes": config.num_classes,
                "num_diffusion_steps": config.num_diffusion_steps,
                "cfg_dropout": config.cfg_dropout,
                "sample_steps": config.sample_steps,
                "guidance_scale": config.guidance_scale,
                "class_conditional": config.class_conditional,
                "use_attention": config.use_attention,
            }
        )
    elif model_type == "google_ddpm":
        extra_kwargs.update(
            {
                "pretrained_model_id": config.pretrained_model_id,
                "num_diffusion_steps": config.num_diffusion_steps,
                "sample_steps": config.sample_steps,
                "disable_attention_on_cpu": config.disable_attention_on_cpu,
            }
        )

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
    result = {"fid": fid, "kid_mean": kid_mean, "kid_std": kid_std}
    print(result)

    if wandb_run is not None:
        try:
            wandb_run.log(result)
        finally:
            wandb_run.finish()


if __name__ == "__main__":
    main()
