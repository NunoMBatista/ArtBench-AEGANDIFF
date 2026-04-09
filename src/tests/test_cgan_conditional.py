"""
Test cGAN's conditional generation: generate N images per class and save a grid
so you can visually inspect whether the class conditioning is working.

Usage:
    python src/test_cgan_conditional.py
    python src/test_cgan_conditional.py --checkpoint outputs/cgan.pt --n_per_class 8
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cGAN import cGAN

ARTBENCH_CLASSES = [
    "art_nouveau",
    "baroque",
    "expressionism",
    "impressionism",
    "post_impressionism",
    "realism",
    "renaissance",
    "romanticism",
    "surrealism",
    "ukiyo_e",
]


def load_generator(checkpoint_path: str, device: torch.device) -> cGAN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support both raw state_dict and wrapped checkpoint dicts
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        cfg = {}
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        cfg = checkpoint.get("config", {})
    elif isinstance(checkpoint, dict) and any(k.startswith("generator.") for k in checkpoint):
        state_dict = checkpoint
        cfg = {}
    else:
        state_dict = checkpoint
        cfg = {}

    # Best-config hyperparameters (from docs/best_params.md)
    model = cGAN(
        latent_dim=cfg.get("latent_dim", 128),
        num_classes=cfg.get("num_classes", 10),
        embed_dim=cfg.get("embed_dim", 64),
        img_channels=3,
        img_size=32,
        base_channels=cfg.get("base_channels", 128),
        use_spectral_norm=cfg.get("use_spectral_norm", False),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_per_class(model: cGAN, n_per_class: int, device: torch.device) -> torch.Tensor:
    """Returns tensor of shape (num_classes * n_per_class, 3, 32, 32) in [-1, 1]."""
    all_images = []
    for class_idx in range(model.num_classes):
        z = torch.randn(n_per_class, model.latent_dim, device=device)
        labels = torch.full((n_per_class,), class_idx, dtype=torch.long, device=device)
        imgs = model.generator(z, labels)  # (n, 3, 32, 32)
        all_images.append(imgs.cpu())
    return torch.cat(all_images, dim=0)  # (num_classes * n_per_class, 3, 32, 32)


def save_grid(images: torch.Tensor, n_per_class: int, class_names: list, out_path: Path):
    """Save images as a grid with one row per class."""
    # images: (num_classes * n_per_class, 3, 32, 32) in [-1, 1]
    num_classes = len(class_names)
    assert images.shape[0] == num_classes * n_per_class

    # Denormalise from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = images.clamp(0, 1)

    # nrow = n_per_class so each row = one class
    grid = vutils.make_grid(images, nrow=n_per_class, padding=2, normalize=False)

    # Convert to PIL and add class labels on the left
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = Image.fromarray(grid_np)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved grid to: {out_path}")


def print_class_summary(images: torch.Tensor, n_per_class: int, class_names: list):
    """Print per-class pixel stats to spot degenerate outputs."""
    num_classes = len(class_names)
    print(f"\n{'Class':<22}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        chunk = images[i * n_per_class : (i + 1) * n_per_class]
        print(
            f"{name:<22}  {chunk.mean():>8.4f}  {chunk.std():>8.4f}"
            f"  {chunk.min():>8.4f}  {chunk.max():>8.4f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Test cGAN conditional generation")
    parser.add_argument("--checkpoint", default="outputs/cgan.pt", help="Path to cgan.pt")
    parser.add_argument("--n_per_class", type=int, default=8, help="Images to generate per class")
    parser.add_argument("--out", default="outputs/cgan_conditional_test.png", help="Output grid path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_generator(args.checkpoint, device)
    print(f"Model loaded — latent_dim={model.latent_dim}, num_classes={model.num_classes}")

    images = generate_per_class(model, args.n_per_class, device)
    print(f"Generated {images.shape[0]} images ({args.n_per_class} per class)")

    print_class_summary(images, args.n_per_class, ARTBENCH_CLASSES)

    save_grid(images, args.n_per_class, ARTBENCH_CLASSES, Path(args.out))

    # Also save individual class grids for closer inspection
    out_dir = Path(args.out).parent / "cgan_per_class"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(ARTBENCH_CLASSES):
        chunk = images[i * args.n_per_class : (i + 1) * args.n_per_class]
        chunk = (chunk + 1) / 2
        row = vutils.make_grid(chunk, nrow=args.n_per_class, padding=2, normalize=False)
        row_np = (row.permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(row_np).save(out_dir / f"{i:02d}_{name}.png")
    print(f"Per-class rows saved to: {out_dir}/")


if __name__ == "__main__":
    main()