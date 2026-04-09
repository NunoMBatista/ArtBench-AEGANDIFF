"""Train each model 10 times (seeds 1-10, 100% ArtBench) and plot mean ± std
training-loss curves per model.

Run from the repo root:
    python src/loss_curves.py
"""

import json
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

from globals import ensure_repo_root
ensure_repo_root()
load_dotenv()

from src.train import (
    get_model,
    get_optimizer,
    get_step_fn,
    train_loop,
)
from src.utils.data_loader import get_dataloaders
from src.utils.seed_setter import set_global_seed


# ── Best hyperparameters (docs/best_params.md) ───────────────────────────────

TRAIN_SEEDS = list(range(1, 11))
#TRAIN_SEEDS = list(range(1, 2))

_COMMON = dict(
    kaggle_root="data",
    num_workers=0,           # Windows-safe (avoids spawn/pickle worker crashes)
    use_subset=False,        # 100 % of the dataset
    subset_mode="csv",
    subset_csv_path="provided/training_20_percent.csv",
    subset_seed=67,
    subset_fraction=1.0,
    device="cuda",
)

BEST_CONFIGS: Dict[str, dict] = {
    "vae": {
        **_COMMON,
        "model_type": "vae",
        "base_channels": 128,
        "latent_dim": 256,
        "beta": 0.5,
        "lr": 0.002,
        "batch_size": 256,
        "epochs": 20,
        "optimizer": {"name": "adam", "betas": [0.9, 0.999], "weight_decay": 0.0},
    },
    "dcgan": {
        **_COMMON,
        "model_type": "dcgan",
        "base_channels": 128,
        "latent_dim": 64,
        "lr": 0.0002,
        "d_updates_per_g": 1,
        "use_spectral_norm": False,
        "batch_size": 128,
        "epochs": 20,
        "optimizer": {"name": "adam", "betas": [0.0, 0.999], "weight_decay": 0.0},
    },
    "cgan": {
        **_COMMON,
        "model_type": "cgan",
        "base_channels": 128,
        "latent_dim": 128,
        "embed_dim": 64,
        "num_classes": 10,
        "lr": 5e-5,
        "d_updates_per_g": 5,
        "use_spectral_norm": False,
        "batch_size": 128,
        "epochs": 20,
        "optimizer": {"name": "adam", "betas": [0.0, 0.999], "weight_decay": 0.0},
    },
    "diffusion": {
        **_COMMON,
        "model_type": "diffusion",
        "base_channels": 128,
        "latent_dim": 128,
        "img_channels": 3,
        "image_size": 32,
        "num_classes": 10,
        "num_diffusion_steps": 1000,
        "sample_steps": 100,
        "guidance_scale": 2.0,
        "cfg_dropout": 0.1,
        "class_conditional": False,
        "use_attention": False,
        "lr": 0.0002,
        "batch_size": 32,
        "epochs": 20,
        "optimizer": {"name": "adamw", "betas": [0.9, 0.999], "weight_decay": 1e-4},
    },
    "google_ddpm": {
        **_COMMON,
        "model_type": "google_ddpm",
        "latent_dim": 128,
        "base_channels": 128,
        "pretrained_model_id": "google/ddpm-cifar10-32",
        "num_diffusion_steps": 1000,
        "sample_steps": 100,
        "disable_attention_on_cpu": False,
        "lr": 0.0005,
        "batch_size": 32,
        "epochs": 20,
        "optimizer": {"name": "adamw", "betas": [0.9, 0.999], "weight_decay": 1e-4},
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_device(config: dict) -> torch.device:
    requested = config.get("device", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def train_one_seed(model_name: str, config: dict, seed: int) -> List[Dict]:
    """Train model for one seed and return the epoch history."""
    set_global_seed(seed)
    device = _resolve_device(config)

    train_loader, _, _ = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_subset=config["use_subset"],
        subset_mode=config["subset_mode"],
        subset_csv_path=config["subset_csv_path"],
        subset_seed=config["subset_seed"],
        subset_fraction=config.get("subset_fraction", 1.0),
        kaggle_root=config["kaggle_root"],
    )

    model = get_model(config, device)
    optimizer = get_optimizer(model, config)
    step_fn = get_step_fn(config)

    history = train_loop(
        model,
        optimizer,
        step_fn,
        train_loader,
        epochs=config["epochs"],
        device=device,
        d_updates_per_g=config.get("d_updates_per_g", 1),
    )
    return history


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_loss_curves(
    model_name: str,
    histories: List[List[Dict]],
    out_dir: str,
) -> str:
    """Plot mean ± std loss curves for all keys logged during training.

    histories: list of per-seed histories (each is a list of epoch dicts).
    """
    # Collect all logged keys (exclude 'epoch')
    loss_keys = [k for k in histories[0][0] if k != "epoch"]
    epochs = [h["epoch"] for h in histories[0]]

    n_keys = len(loss_keys)
    fig, axes = plt.subplots(1, n_keys, figsize=(5 * n_keys, 4), squeeze=False)

    for ax, key in zip(axes[0], loss_keys):
        # shape: (n_seeds, n_epochs)
        values = np.array([[h[key] for h in hist] for hist in histories])
        mean = values.mean(axis=0)
        std = values.std(axis=0)

        ax.plot(epochs, mean, color="steelblue", linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.25, color="steelblue")
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(linestyle="--", alpha=0.35)

    fig.suptitle(
        f"{model_name} — mean ± std over {len(histories)} seeds",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"loss_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_overview(all_histories: Dict[str, List[List[Dict]]], out_dir: str) -> str:
    """Combined overview: one subplot per model, train_loss mean ± std."""
    model_names = list(all_histories.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        histories = all_histories[model_name]
        epochs = [h["epoch"] for h in histories[0]]
        values = np.array([[h["train_loss"] for h in hist] for hist in histories])
        mean = values.mean(axis=0)
        std = values.std(axis=0)

        ax.plot(epochs, mean, color="steelblue", linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.25, color="steelblue")
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("train_loss")
        ax.grid(linestyle="--", alpha=0.35)

    fig.suptitle(
        f"Training loss — mean ± std over {len(TRAIN_SEEDS)} seeds",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curves_overview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"loss_curves_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # all_histories[model_name] = list of per-seed histories
    all_histories: Dict[str, List[List[Dict]]] = {}

    for model_name, config in BEST_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()}  ({len(TRAIN_SEEDS)} training seeds)")
        print(f"{'='*60}")

        seed_histories = []
        for seed in TRAIN_SEEDS:
            print(f"  seed={seed} ...", flush=True)
            history = train_one_seed(model_name, config, seed)
            seed_histories.append(history)

        all_histories[model_name] = seed_histories

        # Per-model per-run data file
        model_data = {
            f"seed_{seed}": history
            for seed, history in zip(TRAIN_SEEDS, seed_histories)
        }
        model_path = os.path.join(out_dir, f"loss_{model_name}_runs.json")
        with open(model_path, "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"  -> runs data: {model_path}")

        curve_path = plot_loss_curves(model_name, seed_histories, out_dir)
        print(f"  -> {curve_path}")

    # Persist raw histories
    histories_path = os.path.join(out_dir, "loss_histories.json")
    with open(histories_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    print(f"\nLoss histories JSON -> {histories_path}")

    overview_path = plot_overview(all_histories, out_dir)
    print(f"Overview -> {overview_path}")


if __name__ == "__main__":
    main()
