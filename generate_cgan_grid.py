"""
Class-wise cGAN sample grid: each row = one art style, each column = a different sample.

Usage:
    python generate_cgan_grid.py [--ckpt PATH] [--cols 8]
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torchvision.utils import make_grid

from src.evaluate import EvalConfig, _load_model, _find_latest_checkpoint
from src.utils.seed_setter import set_global_seed

ARTBENCH_CLASSES = [
    "Art Nouveau", "Baroque", "Expressionism", "Impressionism",
    "Post-Impressionism", "Realism", "Renaissance", "Romanticism",
    "Surrealism", "Ukiyo-e",
]


def tensor_to_img(t):
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = (img + 1.0) / 2.0
    return np.clip(img, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Path to cGAN checkpoint")
    parser.add_argument("--cols", type=int, default=8, help="Number of samples per class (columns)")
    parser.add_argument("--out", type=str, default="docs/report/images/cgan_class_grid.png", help="Output file path")
    parser.add_argument("--title", type=str, default="cGAN — Generated Samples by Class")
    parser.add_argument("--run_prefix", type=str, default="cgan", help="Run prefix for auto checkpoint discovery")
    parser.add_argument("--no_labels", action="store_true", help="Output a tight unlabeled grid (like make_grid)")
    parser.add_argument("--class_offset", type=int, default=0, help="First class index to include")
    parser.add_argument("--n_classes", type=int, default=None, help="Number of classes to include (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(42)

    config = EvalConfig()
    ckpt_path = args.ckpt or _find_latest_checkpoint(args.run_prefix)
    print(f"Using checkpoint: {ckpt_path}")
    config.update(ckpt_path)

    model = _load_model(
        "cgan", ckpt_path,
        config.latent_dim, config.base_channels, device,
        num_classes=config.num_classes,
        use_spectral_norm=config.use_spectral_norm,
    )

    n_cols = args.cols
    class_offset = args.class_offset
    n_classes = args.n_classes if args.n_classes is not None else config.num_classes
    class_indices = list(range(class_offset, class_offset + n_classes))
    class_names = [ARTBENCH_CLASSES[c] for c in class_indices]

    print(f"Generating {n_cols} samples for classes {class_indices}...")
    model.eval()
    all_images = []
    with torch.no_grad():
        for c in class_indices:
            z = torch.randn(n_cols, model.latent_dim, device=device)
            lbls = torch.full((n_cols,), c, dtype=torch.long, device=device)
            imgs = model.generator(z, lbls)
            all_images.append(imgs)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    if args.no_labels:
        # Tight unlabeled grid: rows = classes, cols = samples (like make_grid training output)
        flat = torch.cat(all_images, dim=0)          # (n_classes * n_cols, C, H, W)
        flat = flat.add(1.0).div(2.0).clamp(0, 1)   # [-1,1] -> [0,1]
        grid = make_grid(flat, nrow=n_cols, padding=2, pad_value=0)
        img = grid.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(n_cols * 1.0, n_classes * 1.0))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        fig = plt.figure(figsize=(n_cols * 1.5, n_classes * 1.5 + 0.5))
        gs = gridspec.GridSpec(
            n_classes, n_cols + 1,
            width_ratios=[1.8] + [1] * n_cols,
            hspace=0.05, wspace=0.05,
        )
        for row, (class_imgs, class_name) in enumerate(zip(all_images, class_names)):
            ax_label = fig.add_subplot(gs[row, 0])
            ax_label.axis("off")
            ax_label.text(
                0.95, 0.5, class_name,
                ha="right", va="center", fontsize=9, fontweight="bold",
                transform=ax_label.transAxes,
            )
            for col in range(n_cols):
                ax = fig.add_subplot(gs[row, col + 1])
                ax.imshow(tensor_to_img(class_imgs[col]))
                ax.axis("off")
        fig.suptitle(args.title, fontsize=13, y=1.01)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
