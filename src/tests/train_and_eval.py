"""Train all 5 models once (seed=67, 100% ArtBench by default), save 5 .pt
files, then evaluate each trained model with 10 different seeds (1-10) and
produce a summary histogram (mean ± std).

Run from the repo root:
    python src/tests/train_and_eval.py            # full 50k dataset
    python src/tests/train_and_eval.py --subset   # 20% subset (~10k images)
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from globals import ensure_repo_root
ensure_repo_root()
load_dotenv()

from src.train import (
    _compute_fid_kid_metrics,
    get_model,
    get_optimizer,
    get_step_fn,
    train_loop,
)
from src.utils.data_loader import get_dataloaders
from src.utils.seed_setter import set_global_seed


# ── Best hyperparameters (docs/best_params.md) ───────────────────────────────

TRAIN_SEED = 67

_COMMON = dict(
    kaggle_root="data",
    num_workers=0,           # Windows-safe (avoids spawn/pickle worker crashes)
    use_subset=False,        # 100 % of the dataset
    subset_mode="csv",
    subset_csv_path="provided/training_20_percent.csv",
    subset_seed=67,
    subset_fraction=1.0,
    device="cuda",
    eval_num_samples=5000,
    eval_metrics_batch_size=32,
    eval_gen_batch_size=32,
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

EVAL_SEEDS = list(range(1, 11))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_device(config: dict) -> torch.device:
    requested = config.get("device", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def train_model(model_name: str, config: dict, out_dir: str) -> tuple:
    """Train model once with TRAIN_SEED. Returns (model, train_loader, device, pt_path)."""
    print(f"\n{'='*60}")
    print(f"  TRAINING  {model_name.upper()}  (seed={TRAIN_SEED})")
    print(f"{'='*60}")

    set_global_seed(TRAIN_SEED)
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

    pt_path = os.path.join(out_dir, f"{model_name}.pt")
    torch.save(
        {"seed": TRAIN_SEED, "model_type": model_name, "model_state": model.state_dict()},
        pt_path,
    )
    print(f"  Saved -> {pt_path}")

    return model, train_loader, device, pt_path, history


def eval_model(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    config: dict,
    eval_seeds: List[int],
) -> List[Dict]:
    """Evaluate a trained model with multiple seeds. Returns list of metric dicts."""
    results = []
    for seed in eval_seeds:
        print(f"  eval seed={seed} ... ", end="", flush=True)
        set_global_seed(seed)
        metrics = _compute_fid_kid_metrics(
            model=model,
            train_loader=train_loader,
            device=device,
            num_samples=config["eval_num_samples"],
            metrics_batch_size=config["eval_metrics_batch_size"],
            gen_batch_size=config["eval_gen_batch_size"],
        )
        metrics["seed"] = seed
        results.append(metrics)
        print(f"FID={metrics['fid']:.2f}  KID={metrics['kid_mean']:.4f}")
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_results: Dict[str, List[Dict]], out_dir: str, dataset_label: str = "100% ArtBench") -> str:
    model_names = list(all_results.keys())
    fid_by_model = {m: [r["fid"] for r in all_results[m]] for m in model_names}
    kid_by_model = {m: [r["kid_mean"] for r in all_results[m]] for m in model_names}

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    x = np.arange(len(model_names))
    width = 0.5
    rng = np.random.default_rng(0)

    for ax, scores_by_model, ylabel, ann_fmt in zip(
        axes,
        [fid_by_model, kid_by_model],
        ["FID ↓", "KID ↓"],
        [".1f", ".3f"],
    ):
        means = np.array([np.mean(scores_by_model[m]) for m in model_names])
        stds = np.array([np.std(scores_by_model[m]) for m in model_names])

        ax.bar(
            x, means, width,
            yerr=stds, capsize=6,
            alpha=0.72, color="steelblue", ecolor="black", linewidth=0,
        )

        # Scatter individual seed scores with slight jitter for visibility
        for i, m in enumerate(model_names):
            jitter = rng.uniform(-0.10, 0.10, len(scores_by_model[m]))
            ax.scatter(
                i + jitter, scores_by_model[m],
                color="black", s=20, zorder=3, alpha=0.55,
            )

        # Annotate mean ± std above each bar
        y_pad = (means + stds).max() * 0.03
        for i, (mu, sigma) in enumerate(zip(means, stds)):
            ax.text(
                i, mu + sigma + y_pad,
                f"{mu:{ann_fmt}}±{sigma:{ann_fmt}}",
                ha="center", va="bottom", fontsize=8.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} — 10 eval seeds, {dataset_label} (trained seed={TRAIN_SEED})", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_xlim(-0.6, len(model_names) - 0.4)

    plt.tight_layout(pad=2.0)
    hist_path = os.path.join(out_dir, "histogram_fid_kid.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    return hist_path


def plot_loss_curves(all_histories: Dict[str, List[Dict]], out_dir: str) -> str:
    """Save a per-model loss-curve PNG and a combined overview PNG."""
    # ── per-model plots ───────────────────────────────────────────────────────
    for model_name, history in all_histories.items():
        epochs = [h["epoch"] for h in history]
        loss_keys = [k for k in history[0] if k != "epoch"]

        fig, ax = plt.subplots(figsize=(8, 4))
        for key in loss_keys:
            values = [h[key] for h in history]
            ax.plot(epochs, values, marker="o", markersize=3, label=key)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss / metric")
        ax.set_title(f"{model_name} — training loss (seed={TRAIN_SEED})")
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.35)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"loss_{model_name}.png"), dpi=150)
        plt.close(fig)

    # ── combined overview: one subplot per model, train_loss only ─────────────
    n = len(all_histories)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (model_name, history) in zip(axes, all_histories.items()):
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        ax.plot(epochs, train_loss, color="steelblue", marker="o", markersize=3)
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("train_loss")
        ax.grid(linestyle="--", alpha=0.35)

    fig.suptitle(f"Training loss curves — seed={TRAIN_SEED}", fontsize=12)
    plt.tight_layout()
    combined_path = os.path.join(out_dir, "loss_curves_overview.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    return combined_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train all models and evaluate with 10 seeds.")
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Train on the 20%% subset (~10k images) instead of the full 50k dataset.",
    )
    args = parser.parse_args()

    if args.subset:
        for cfg in BEST_CONFIGS.values():
            cfg["use_subset"] = True
        dataset_label = "20% subset"
        run_tag = "subset"
    else:
        dataset_label = "100% ArtBench"
        run_tag = "full"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"final_eval_{run_tag}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    all_results: Dict[str, List[Dict]] = {}
    all_histories: Dict[str, List[Dict]] = {}

    for model_name, config in BEST_CONFIGS.items():
        # 1. Train once
        model, train_loader, device, pt_path, history = train_model(model_name, config, out_dir)
        all_histories[model_name] = history

        # 2. Evaluate with 10 seeds
        print(f"\n  EVALUATING  {model_name.upper()}  ({len(EVAL_SEEDS)} seeds)")
        results = eval_model(model, train_loader, device, config, EVAL_SEEDS)
        all_results[model_name] = results

        fids = [r["fid"] for r in results]
        kids = [r["kid_mean"] for r in results]
        print(
            f"\n  [{model_name}]  "
            f"FID: {np.mean(fids):.2f} ± {np.std(fids):.2f}  |  "
            f"KID: {np.mean(kids):.4f} ± {np.std(kids):.4f}"
        )

    # Persist raw results
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults JSON -> {results_path}")

    histories_path = os.path.join(out_dir, "loss_histories.json")
    with open(histories_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    print(f"Loss histories JSON -> {histories_path}")

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Model':<15} {'FID mean':>10} {'FID std':>10} {'KID mean':>10} {'KID std':>10}")
    print("=" * 72)
    for model_name, results in all_results.items():
        fids = [r["fid"] for r in results]
        kids = [r["kid_mean"] for r in results]
        print(
            f"{model_name:<15} {np.mean(fids):>10.2f} {np.std(fids):>10.2f}"
            f" {np.mean(kids):>10.4f} {np.std(kids):>10.4f}"
        )

    hist_path = plot_results(all_results, out_dir, dataset_label)
    print(f"\nHistogram -> {hist_path}")

    loss_overview_path = plot_loss_curves(all_histories, out_dir)
    print(f"Loss curves -> {loss_overview_path}")


if __name__ == "__main__":
    main()
