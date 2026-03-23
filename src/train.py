import os
import sys
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

try:
    import wandb
except ImportError:
    wandb = None

from globals import ensure_repo_root
ensure_repo_root()
load_dotenv()

from src.models.VAE import VAE, vae_loss
from src.models.DCGAN import DCGAN, dcgan_loss
from src.models.cGAN import cGAN, cgan_loss
from src.models.diffusion import DiffusionModel
from src.models.google_DDPM import GoogleDDPMFineTuner
from src.utils.data_loader import get_dataloaders
from src.utils.metrics import compute_fid_kid
from src.utils.seed_setter import set_global_seed


Batch = Tuple[torch.Tensor, torch.Tensor]
StepFn = Callable[[torch.nn.Module, Batch, torch.device, bool], Tuple[torch.Tensor, Dict[str, float]]]


def load_config(path: str) -> Dict:
    # Centralized YAML loading keeps all experiment controls in config files.
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
    optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    step_fn: Callable,
    device: torch.device,
    train: bool,
    d_updates_per_g: int = 1,
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

            if train:
                if isinstance(optimizer, dict):
                    # GAN mode: Handle multiple optimizers (D and G)
                    # We pass the optimizers to the step function which handles the internal update logic
                    loss, metrics = step_fn(model, batch, optimizer, device, train, d_updates_per_g)
                else:
                    # Standard mode (VAE)
                    loss, metrics = step_fn(model, batch, device, train)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            else:
                # Evaluation mode
                if isinstance(optimizer, dict):
                    loss, metrics = step_fn(model, batch, optimizer, device, train, d_updates_per_g)
                else:
                    loss, metrics = step_fn(model, batch, device, train)

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
    d_updates_per_g: int = 1,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every_epochs: int = 0,
    epoch_logger: Optional[Callable[[Dict], None]] = None,
    epoch_eval_fn: Optional[Callable[[int], Dict[str, float]]] = None,
):
    history = []

    # For each epoch, run training and validation (if provided)
    for epoch in range(1, epochs + 1):
        # Run training
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, step_fn, device, train=True, d_updates_per_g=d_updates_per_g
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
                model, val_loader, optimizer, step_fn, device, train=False, d_updates_per_g=d_updates_per_g
            )
            log.update(
                {
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

        if epoch_eval_fn is not None:
            extra_metrics = epoch_eval_fn(epoch)
            if extra_metrics:
                log.update(extra_metrics)

        history.append(log)
        print(log)

        if epoch_logger is not None:
            epoch_logger(log)

        # Optionally save intermediate checkpoints every N epochs.
        if checkpoint_dir and checkpoint_every_epochs > 0 and (epoch % checkpoint_every_epochs == 0):
            ckpt_name = f"model_epoch_{epoch:03d}.pt"
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(checkpoint_dir, ckpt_name),
            )
    return history


def _init_wandb(config: Dict, model_type: str, run_dir: str):
    # W&B integration is optional and should never block local training.
    wandb_cfg = config.get("wandb", {}) or {}
    if not wandb_cfg.get("enabled", False):
        return None
    if wandb is None:
        print("WARNING: wandb is enabled in config but the package is not installed. Continuing without wandb.")
        return None

    try:
        entity = wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
        project = wandb_cfg.get("project") or os.getenv("WANDB_PROJECT") or "ArtBench-AEGANDIFF"
        run = wandb.init(
            project=project,
            entity=entity,
            name=wandb_cfg.get("run_name", os.path.basename(run_dir)),
            dir=run_dir,
            config=config,
            tags=wandb_cfg.get("tags", [model_type]),
            notes=wandb_cfg.get("notes", ""),
        )
        # When running inside a sweep, wandb.config contains the agent's
        # chosen hyperparameters.  Merge them back into the local config
        # dict so every downstream consumer (model, optimizer, etc.) picks
        # up the sweep values automatically.
        for key, value in dict(wandb.config).items():
            if key in config:
                config[key] = value
            elif "." in key:
                # Support nested keys like "optimizer.betas"
                parts = key.split(".")
                d = config
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = value
        return run
    except Exception as e:
        print(f"WARNING: failed to initialize wandb ({e}). Continuing without wandb.")
        return None


def _make_run_dir(prefix: str = "run_model") -> str:
    # Every run writes to a timestamped folder for reproducibility and easy comparison.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_history_plot(history, run_dir: str):
    # Plot only train/val loss curves because these are available for all model families.
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


def _save_sample_grid(model, device: torch.device, run_dir: str, num_samples: int = 64):
    # Save a quick qualitative snapshot in image space [0, 1].
    model.eval()
    with torch.no_grad():
        if hasattr(model, "sample"):
            samples = model.sample(num_samples, device=device).cpu()
        else:
            # Fallback for generic models
            z = torch.randn(num_samples, model.latent_dim, 1, 1, device=device)
            samples = model.generator(z).cpu()
    samples = samples.add(1.0).div(2.0).clamp(0.0, 1.0)
    grid = make_grid(samples, nrow=8)
    save_image(grid, os.path.join(run_dir, "samples.png"))


def _compute_fid_kid_metrics(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int,
    metrics_batch_size: int,
    gen_batch_size: int,
) -> Dict[str, float]:
    # This helper keeps FID/KID computation consistent for periodic and final evaluation.
    was_training = model.training
    model.eval()

    real_batches = []
    total = 0
    for batch, _ in train_loader:
        real_batches.append(batch.cpu().numpy())
        total += batch.shape[0]
        if total >= num_samples:
            break
    real_images = np.concatenate(real_batches, axis=0)
    target_samples = min(num_samples, real_images.shape[0])
    real_images = real_images[:target_samples]

    fake_batches = []
    remaining = target_samples
    gen_batch_size = max(1, int(gen_batch_size))
    with torch.no_grad():
        while remaining > 0:
            cur = min(gen_batch_size, remaining)
            fake_batches.append(model.sample(cur, device=device).cpu().numpy())
            remaining -= cur
    fake_images = np.concatenate(fake_batches, axis=0)

    fid, kid_mean, kid_std = compute_fid_kid(
        real_images,
        fake_images,
        device=device,
        batch_size=metrics_batch_size,
    )

    if was_training:
        model.train()

    return {
        "fid": float(fid),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
    }


def get_model(config: Dict, device: torch.device) -> torch.nn.Module:
    # Single factory entrypoint so new model families plug into the same train loop.
    model_type = config["model_type"].lower()
    latent_dim = config.get("latent_dim", 128)
    base_channels = config.get("base_channels", 64)

    if model_type == "vae":
        return VAE(latent_dim=latent_dim, base_channels=base_channels).to(device)
    elif model_type == "dcgan":
        return DCGAN(
            latent_dim=latent_dim,
            base_channels=base_channels,
            use_spectral_norm=config.get("use_spectral_norm", False)
        ).to(device)
    elif model_type == "cgan":
        return cGAN(
            latent_dim=latent_dim,
            num_classes=config.get("num_classes", 10),
            embed_dim=config.get("embed_dim", 64),
            base_channels=base_channels,
            use_spectral_norm=config.get("use_spectral_norm", False),
        ).to(device)
    elif model_type == "diffusion":
        return DiffusionModel(
            latent_dim=latent_dim,
            base_channels=base_channels,
            img_channels=config.get("img_channels", 3),
            image_size=config.get("image_size", 32),
            num_classes=config.get("num_classes", 10),
            num_diffusion_steps=config.get("num_diffusion_steps", 1000),
            cfg_dropout=config.get("cfg_dropout", 0.1),
            sample_steps=config.get("sample_steps", 100),
            guidance_scale=config.get("guidance_scale", 2.0),
            class_conditional=config.get("class_conditional", True),
            use_attention=config.get("use_attention", False),
        ).to(device)
    elif model_type == "google_ddpm":
        return GoogleDDPMFineTuner(
            latent_dim=latent_dim,
            base_channels=base_channels,
            pretrained_model_id=config.get("pretrained_model_id", "google/ddpm-cifar10-32"),
            num_diffusion_steps=config.get("num_diffusion_steps", 1000),
            sample_steps=config.get("sample_steps", 100),
            disable_attention_on_cpu=config.get("disable_attention_on_cpu", device.type == "cpu"),
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_optimizer(model: torch.nn.Module, config: Dict) -> Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]:
    # Optimizer selection is model-aware because GANs need separate G/D optimizers.
    model_type = config["model_type"].lower()
    optim_cfg = config.get("optimizer", {"name": "adam"})
    lr = config["lr"]
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    betas = tuple(optim_cfg.get("betas", [0.9, 0.999]))

    if model_type == "vae":
        if optim_cfg["name"].lower() == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer for VAE: {optim_cfg['name']}")

    elif model_type in ("dcgan", "cgan"):
        # GANs usually need specific betas for stability
        g_betas = tuple(optim_cfg.get("betas", [0.0, 0.999]))
        d_betas = tuple(optim_cfg.get("betas", [0.0, 0.999]))

        opt_g = torch.optim.Adam(
            model.generator.parameters(),
            lr=lr,
            betas=g_betas,
            weight_decay=weight_decay,
        )
        opt_d = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=lr,
            betas=d_betas,
            weight_decay=weight_decay,
        )
        return {"opt_g": opt_g, "opt_d": opt_d}

    elif model_type == "diffusion":
        opt_name = optim_cfg.get("name", "adamw").lower()
        if opt_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        if opt_name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer for diffusion: {optim_cfg['name']}")

    elif model_type == "google_ddpm":
        # Fine-tune only explicitly trainable layers from the HF UNet wrapper.
        params = [p for p in model.parameters() if p.requires_grad]
        opt_name = optim_cfg.get("name", "adamw").lower()
        if opt_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        if opt_name == "adam":
            return torch.optim.Adam(
                params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer for google_ddpm: {optim_cfg['name']}")
    
    raise ValueError(f"Unsupported model_type for optimization: {model_type}")


def get_step_fn(config: Dict) -> Callable:
    # A step function encapsulates per-model forward/loss behavior behind one interface.
    model_type = config["model_type"].lower()
    
    if model_type == "vae":
        beta = config.get("beta", 1.0)
        def step_fn(model, batch, device, train):
            images, _ = batch
            recon, mu, logvar = model(images)
            loss, metrics = vae_loss(recon, images, mu, logvar, beta=beta)
            return loss, metrics
        return step_fn

    elif model_type == "dcgan":
        def step_fn(model, batch, optimizers, device, train, d_updates_per_g=1):
            images, _ = batch
            batch_size = images.size(0)
            real_label = 1.0
            fake_label = 0.0

            opt_g = optimizers["opt_g"]
            opt_d = optimizers["opt_d"]

            if train:
                total_d_loss = 0
                for _ in range(d_updates_per_g):
                    opt_d.zero_grad()
                    label_real = torch.full((batch_size,), real_label, device=device)
                    output_real = model.discriminator(images)
                    errD_real = dcgan_loss(output_real, label_real)
                    
                    noise = torch.randn(batch_size, model.latent_dim, device=device)
                    fake = model.generator(noise)
                    label_fake = torch.full((batch_size,), fake_label, device=device)
                    output_fake = model.discriminator(fake.detach())
                    errD_fake = dcgan_loss(output_fake, label_fake)
                    
                    errD = errD_real + errD_fake
                    errD.backward()
                    opt_d.step()
                    total_d_loss += errD.item()

                opt_g.zero_grad()
                label_g = torch.full((batch_size,), real_label, device=device)
                output_g = model.discriminator(fake)
                errG = dcgan_loss(output_g, label_g)
                errG.backward()
                opt_g.step()

                metrics = {
                    "errD": total_d_loss / d_updates_per_g,
                    "errG": errG.item(),
                    "D_x": output_g.mean().item(),
                }
                return errG, metrics
            else:
                with torch.no_grad():
                    noise = torch.randn(batch_size, model.latent_dim, device=device)
                    fake = model.generator(noise)
                    output_fake = model.discriminator(fake)
                    output_real = model.discriminator(images)
                    errG = dcgan_loss(output_fake, torch.full((batch_size,), real_label, device=device))
                    errD = dcgan_loss(output_real, torch.full((batch_size,), real_label, device=device)) + \
                           dcgan_loss(output_fake, torch.full((batch_size,), fake_label, device=device))
                return errG, {"errD": errD.item(), "errG": errG.item()}
        return step_fn

    elif model_type == "cgan":
        def step_fn(model, batch, optimizers, device, train, d_updates_per_g=1):
            images, labels = batch
            batch_size = images.size(0)
            real_label = 1.0
            fake_label = 0.0

            opt_g = optimizers["opt_g"]
            opt_d = optimizers["opt_d"]

            if train:
                total_d_loss = 0
                for _ in range(d_updates_per_g):
                    opt_d.zero_grad()
                    label_real = torch.full((batch_size,), real_label, device=device)
                    output_real = model.discriminator(images, labels)
                    errD_real = cgan_loss(output_real, label_real)

                    noise = torch.randn(batch_size, model.latent_dim, device=device)
                    fake = model.generator(noise, labels)
                    label_fake = torch.full((batch_size,), fake_label, device=device)
                    output_fake = model.discriminator(fake.detach(), labels)
                    errD_fake = cgan_loss(output_fake, label_fake)

                    errD = errD_real + errD_fake
                    errD.backward()
                    opt_d.step()
                    total_d_loss += errD.item()

                opt_g.zero_grad()
                label_g = torch.full((batch_size,), real_label, device=device)
                output_g = model.discriminator(fake, labels)
                errG = cgan_loss(output_g, label_g)
                errG.backward()
                opt_g.step()

                metrics = {
                    "errD": total_d_loss / d_updates_per_g,
                    "errG": errG.item(),
                    "D_x": output_g.mean().item(),
                }
                return errG, metrics
            else:
                with torch.no_grad():
                    noise = torch.randn(batch_size, model.latent_dim, device=device)
                    fake = model.generator(noise, labels)
                    output_fake = model.discriminator(fake, labels)
                    output_real = model.discriminator(images, labels)
                    errG = cgan_loss(output_fake, torch.full((batch_size,), real_label, device=device))
                    errD = cgan_loss(output_real, torch.full((batch_size,), real_label, device=device)) + \
                           cgan_loss(output_fake, torch.full((batch_size,), fake_label, device=device))
                return errG, {"errD": errD.item(), "errG": errG.item()}
        return step_fn

    elif model_type == "diffusion":
        class_conditional = bool(config.get("class_conditional", True))

        def step_fn(model, batch, device, train):
            images, labels = batch

            # Diffusion loss is MSE between true and predicted noise at random timesteps.
            loss = model(images, labels if class_conditional else None)
            return loss, {"noise_mse": float(loss.item())}
        return step_fn

    elif model_type == "google_ddpm":
        def step_fn(model, batch, device, train):
            images, _labels = batch
            # Unconditional fine-tuning by design for google/ddpm-cifar10-32.
            loss = model(images, y=None)
            return loss, {"noise_mse": float(loss.item())}
        return step_fn
    
    raise ValueError(f"No step_fn for model_type: {model_type}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    set_global_seed(config["seed"])

    model_type = config["model_type"].lower()
    run_dir = _make_run_dir(config.get("run_prefix", f"run_{model_type}"))
    wandb_run = _init_wandb(config, model_type, run_dir)

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

    requested_device = config.get("device", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("WARNING: 'cuda' device requested but torch.cuda.is_available() is False. Falling back to 'cpu'.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    model = get_model(config, device)
    optimizer = get_optimizer(model, config)
    step_fn = get_step_fn(config)

    def _epoch_logger(log: Dict):
        if wandb_run is not None:
            wandb_run.log(log, step=log["epoch"])

    # Optional periodic evaluation during training for trend tracking.
    fid_kid_every_epochs = int(config.get("fid_kid_every_epochs", 0))
    fid_kid_num_samples = int(config.get("fid_kid_num_samples", config.get("eval_num_samples", 5000)))
    fid_kid_metrics_batch_size = int(
        config.get("fid_kid_metrics_batch_size", config.get("eval_metrics_batch_size", 32))
    )
    fid_kid_gen_batch_size = int(
        config.get(
            "fid_kid_gen_batch_size",
            config.get("eval_gen_batch_size", config.get("batch_size", 32)),
        )
    )

    def _epoch_eval_fn(epoch: int) -> Dict[str, float]:
        if fid_kid_every_epochs <= 0 or (epoch % fid_kid_every_epochs != 0):
            return {}
        # Reuse the same batched evaluator used at end-of-run to avoid OOM drift.
        return _compute_fid_kid_metrics(
            model=model,
            train_loader=train_loader,
            device=device,
            num_samples=fid_kid_num_samples,
            metrics_batch_size=fid_kid_metrics_batch_size,
            gen_batch_size=fid_kid_gen_batch_size,
        )

    history = train_loop(
        model,
        optimizer,
        step_fn,
        train_loader,
        val_loader=val_loader if config.get("use_val", False) else None,
        epochs=config["epochs"],
        device=device,
        d_updates_per_g=config.get("d_updates_per_g", 1),
        checkpoint_dir=run_dir,
        checkpoint_every_epochs=int(config.get("checkpoint_every_epochs", 0)),
        epoch_logger=_epoch_logger,
        epoch_eval_fn=_epoch_eval_fn,
    )

    _save_history_plot(history, run_dir)
    _save_sample_grid(model, device, run_dir)

    # Simple evaluation at the end
    metrics = _compute_fid_kid_metrics(
        model=model,
        train_loader=train_loader,
        device=device,
        num_samples=int(config["eval_num_samples"]),
        metrics_batch_size=int(config["eval_metrics_batch_size"]),
        gen_batch_size=int(config.get("eval_gen_batch_size", config["batch_size"])),
    )

    # Persist run artifacts for offline analysis and reproducibility.
    with open(os.path.join(run_dir, "metrics.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    with open(os.path.join(run_dir, "history.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f, sort_keys=False)

    with open(os.path.join(run_dir, "config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    torch.save({"model_state": model.state_dict()}, os.path.join(run_dir, "model.pt"))

    if wandb_run is not None and wandb is not None:
        try:
            wandb_run.log(metrics, step=len(history))
            samples_path = os.path.join(run_dir, "samples.png")
            if os.path.isfile(samples_path):
                wandb_run.log({"samples": wandb.Image(samples_path)}, step=len(history))

            artifact = wandb.Artifact(name=os.path.basename(run_dir), type="training-run")
            artifact.add_dir(run_dir)
            wandb_run.log_artifact(artifact)
        finally:
            wandb_run.finish()


if __name__ == "__main__":
    main()
