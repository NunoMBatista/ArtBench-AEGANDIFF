import os
import sys
from datetime import datetime
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from globals import ensure_repo_root
ensure_repo_root()

from src.models.VAE import VAE, vae_loss
from src.utils.data_loader import get_dataloaders
from src.utils.metrics import compute_fid_kid
from src.utils.seed_setter import set_global_seed


Batch = Tuple[torch.Tensor, torch.Tensor]
StepFn = Callable[[torch.nn.Module, Batch, torch.device, bool], Tuple[torch.Tensor, Dict[str, float]]]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {path}")
    return config


def _move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    # This takes a batch of (images, labels) and moves them to the specified device (CPU or GPU).
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    step_fn: StepFn,
    device: torch.device,
    train: bool,
):

    # Set the model to training or evaluation mode
    model.train() if train else model.eval()

    # Initialize accumulators
    total_loss = 0.0 # loss
    total_count = 0 # samples processed
    metric_sums: Dict[str, float] = {} # sum of each metric across all samples

    # enable gradients if training, disable if evaluating
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        # For each batch
        for batch in tqdm(loader, desc="train" if train else "eval", unit="batch"):
            batch = _move_batch_to_device(batch, device)

            # Compute loss and metrics for the batch using the provided step function
            # A step function takes the model, a batch of data and the device
            loss, metrics = step_fn(model, batch, device, train)

            # Perform backpropagation and optimization step
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = batch[0].shape[0]

            # Accumulate loss and metrics, weighted by the number of samples in the batch
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_size

    # Compute average loss and metrics over the entire epoch
    avg_loss = total_loss / max(1, total_count)
    avg_metrics = {k: v / max(1, total_count) for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step_fn: StepFn,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader = None,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
):
    history = []

    # For each epoch, run training and validation (if provided)
    for epoch in range(1, epochs + 1):
        # Run training
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, step_fn, device, train=True
        )

        # Log epoch index, training loss, training metrics and validation loss/metrics if val_loader is provided
        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
        }

        # If a validation loader is provided, run validation and log those metrics as well
        if val_loader is not None:
            val_loss, val_metrics = run_epoch(
                model, val_loader, optimizer, step_fn, device, train=False
            )
            log.update(
                {
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

        history.append(log)
        print(log)
    return history


def _make_run_dir(prefix: str = "run_vae") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_history_plot(history, run_dir: str):
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", 0.0) for h in history]
    val_loss = [h.get("val_loss", 0.0) for h in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train")
    if any(val_loss):
        plt.plot(epochs, val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()


def _save_sample_grid(model: VAE, device: torch.device, run_dir: str, num_samples: int = 64):
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device=device).cpu()
    grid = make_grid(samples, nrow=8)
    save_image(grid, os.path.join(run_dir, "samples.png"))


def main():
    if len(sys.argv) != 2:
        print(f"gruda assim: {sys.argv[0]} <config.yml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    set_global_seed(config["seed"])

    run_dir = _make_run_dir(config.get("run_prefix", "run_vae"))

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_subset=config["use_subset"],
        subset_mode=config["subset_mode"],
        subset_csv_path=config["subset_csv_path"],
        subset_seed=config["subset_seed"],
        subset_fraction=config.get("subset_fraction", 0.2),
        kaggle_root=config["kaggle_root"],
    )

    device = torch.device(config["device"])
    model_type = str(config.get("model_type", "vae")).lower()

    if model_type == "vae":
        model = VAE(
            latent_dim=config["latent_dim"],
            base_channels=config.get("base_channels", 64),
        ).to(device)
    elif model_type == "dcgan":
        raise NotImplementedError("NÃO HÁ DCGAN CHIÇA")
    elif model_type == "cgan":
        raise NotImplementedError("NÃO HÁ CGAN CHIÇA")
    elif model_type == "diffusion":
        raise NotImplementedError("NÃO HÁ DIFFUSION CHIÇA")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optim_cfg = config.get("optimizer", {"name": "adam"})
    if optim_cfg["name"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
    else:
        # Add other optimizers later if needed, idk
        raise ValueError(f"Unsupported optimizer: {optim_cfg['name']}")

    def step_fn(model, batch, device, train):
        images, _ = batch
        recon, mu, logvar = model(images)
        loss, metrics = vae_loss(recon, images, mu, logvar, beta=config["beta"])
        return loss, metrics

    history = train_loop(
        model,
        optimizer,
        step_fn,
        train_loader,
        val_loader=None,
        epochs=config["epochs"],
        device=device,
    )

    _save_history_plot(history, run_dir)
    _save_sample_grid(model, device, run_dir)

    real_batches = []
    total = 0
    for batch, _ in train_loader:
        real_batches.append(batch.cpu().numpy())
        total += batch.shape[0]
        if total >= config["eval_num_samples"]:
            break
    real_images = np.concatenate(real_batches, axis=0)[: config["eval_num_samples"]]

    with torch.no_grad():
        fake_images = model.sample(config["eval_num_samples"], device=device).cpu().numpy()

    fid, kid_mean, kid_std = compute_fid_kid(
        real_images,
        fake_images,
        device=config["device"],
        batch_size=config["eval_metrics_batch_size"],
    )

    metrics = {
        "fid": fid,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
    }

    with open(os.path.join(run_dir, "metrics.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    with open(os.path.join(run_dir, "history.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f, sort_keys=False)

    with open(os.path.join(run_dir, "config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    torch.save({"model_state": model.state_dict()}, os.path.join(run_dir, "vae.pt"))


if __name__ == "__main__":
    main()
