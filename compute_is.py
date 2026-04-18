"""
Compute Inception Score for all five model families using best checkpoints.
Generates 5000 samples per model, prints IS mean ± std.

Usage:
    python compute_is.py
"""
import os, sys
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.evaluate import _load_model
from src.utils.metrics import compute_inception_score
from src.utils.seed_setter import set_global_seed

NUM_SAMPLES = 5000
BATCH_SIZE  = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "VAE": {
        "type": "vae",
        "ckpt": "outputs/run_vae_colorful_20260418_161402/model.pt",
        "kwargs": {"latent_dim": 256, "base_channels": 128},
    },
    "DCGAN": {
        "type": "dcgan",
        "ckpt": "outputs/run_dcgan_20260418_001836/model.pt",
        "kwargs": {"latent_dim": 64, "base_channels": 128, "use_spectral_norm": False},
    },
    "cGAN": {
        "type": "cgan",
        "ckpt": "outputs/run_cgan_20260418_201238/model.pt",
        "kwargs": {"latent_dim": 128, "base_channels": 128, "num_classes": 10,
                   "use_spectral_norm": False},
    },
    "Diffusion": {
        "type": "diffusion",
        "ckpt": "outputs/run_diffusion_20260418_115813/model.pt",
        "kwargs": {
            "latent_dim": 128, "base_channels": 128, "img_channels": 3,
            "image_size": 32, "num_classes": 10, "num_diffusion_steps": 1000,
            "cfg_dropout": 0.1, "sample_steps": 100, "guidance_scale": 2.0,
            "class_conditional": False, "use_attention": True,
        },
    },
    "Google DDPM": {
        "type": "google_ddpm",
        "ckpt": "outputs/final_eval_full_20260410_180238/google_ddpm.pt",
        "kwargs": {
            "latent_dim": 128, "base_channels": 128,
            "pretrained_model_id": "google/ddpm-cifar10-32",
            "num_diffusion_steps": 1000, "sample_steps": 100,
            "disable_attention_on_cpu": False,
        },
    },
}


def generate_samples(model, num_samples, device, batch_size=64):
    model.eval()
    batches = []
    remaining = num_samples
    with torch.no_grad():
        with tqdm(total=num_samples, desc="sampling", unit="img") as pbar:
            while remaining > 0:
                cur = min(batch_size, remaining)
                batches.append(model.sample(cur, device=device).cpu().numpy())
                remaining -= cur
                pbar.update(cur)
    return np.concatenate(batches, axis=0)


def main():
    set_global_seed(42)
    results = {}

    for name, cfg in MODELS.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        try:
            extra = {k: v for k, v in cfg["kwargs"].items()
                     if k not in ("latent_dim", "base_channels")}
            model = _load_model(
                cfg["type"], cfg["ckpt"],
                cfg["kwargs"]["latent_dim"],
                cfg["kwargs"]["base_channels"],
                DEVICE,
                **extra,
            )
            fake = generate_samples(model, NUM_SAMPLES, DEVICE, BATCH_SIZE)
            is_mean, is_std = compute_inception_score(fake, device=DEVICE, batch_size=BATCH_SIZE)
            results[name] = (is_mean, is_std)
            print(f"  IS = {is_mean:.3f} ± {is_std:.3f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = (None, None)

    print("\n\n" + "="*50)
    print("  INCEPTION SCORE SUMMARY")
    print("="*50)
    for name, (m, s) in results.items():
        if m is not None:
            print(f"  {name:<15} IS = {m:.3f} ± {s:.3f}")
        else:
            print(f"  {name:<15} FAILED")


if __name__ == "__main__":
    main()
