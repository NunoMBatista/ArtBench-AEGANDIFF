"""
UMAP of VAE reconstructions in Inception V3 feature space.

For each real image (with known class label):
  1. Encode through VAE -> mu
  2. Decode mu -> reconstruction
  3. Extract 2048-dim Inception V3 pool features
  4. UMAP + KDE soft class borders

Usage:
    python generate_vae_inception_umap.py [--ckpt PATH] [--out PATH] [--num_samples 5000]
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import Resize
from scipy.stats import gaussian_kde
from tqdm import tqdm

try:
    import umap
except ImportError:
    print("umap-learn not installed. Run: pip install umap-learn")
    sys.exit(1)

from src.evaluate import EvalConfig, _load_model, _find_latest_checkpoint
from src.utils.data_loader import get_dataloaders
from src.utils.seed_setter import set_global_seed

ARTBENCH_CLASSES = [
    "Art Nouveau", "Baroque", "Expressionism", "Impressionism",
    "Post-Impressionism", "Realism", "Renaissance", "Romanticism",
    "Surrealism", "Ukiyo-e",
]


def build_inception_extractor(device):
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.aux_logits = False
    model = model.to(device).eval()
    return model, Resize((299, 299), antialias=True)


def extract_inception_features(images, inception, resize, device, batch_size=64):
    """images: (N, C, H, W) tensor in [-1, 1]. Returns (N, 2048) numpy array."""
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, images.size(0), batch_size), desc="Inception features"):
            batch = images[i:i + batch_size].to(device)
            batch = (batch + 1.0) / 2.0          # [-1,1] -> [0,1]
            batch = resize(batch)                  # -> 299x299
            feats.append(inception(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def reconstruct_through_vae(vae, images, device, batch_size=128):
    """Encode then decode each image; returns tensor same shape as input."""
    vae.eval()
    recons = []
    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i + batch_size].to(device)
            mu, _ = vae.encode(batch)
            recon = vae.decode(mu)               # use mu directly (no noise)
            recons.append(recon.cpu())
    return torch.cat(recons, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--run_prefix", type=str, default="vae")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--out", type=str, default="docs/report/images/umap_vae_inception.png")
    parser.add_argument("--title", type=str, default="VAE Reconstructions — Inception Features UMAP by Class")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(42)

    config = EvalConfig()
    config.use_subset = True
    config.subset_mode = "csv"
    config.subset_csv_path = "provided/training_20_percent.csv"
    config.subset_seed = 67
    config.kaggle_root = "data"
    config.num_workers = 2

    ckpt_path = args.ckpt or _find_latest_checkpoint(args.run_prefix)
    print(f"Using checkpoint: {ckpt_path}")
    config.update(ckpt_path)

    vae = _load_model("vae", ckpt_path, config.latent_dim, config.base_channels, device)

    # Load real images with labels
    print("Loading real images...")
    train_loader, _, class_names = get_dataloaders(
        batch_size=128,
        num_workers=config.num_workers,
        use_subset=config.use_subset,
        subset_mode=config.subset_mode,
        subset_csv_path=config.subset_csv_path,
        subset_seed=config.subset_seed,
        kaggle_root=config.kaggle_root,
        shuffle_train=True,
    )
    if class_names is None:
        class_names = ARTBENCH_CLASSES

    images, labels = [], []
    for imgs, lbls in train_loader:
        images.append(imgs)
        labels.append(lbls)
        if sum(x.size(0) for x in images) >= args.num_samples:
            break
    images = torch.cat(images, dim=0)[:args.num_samples]
    labels_np = torch.cat(labels, dim=0)[:args.num_samples].numpy()

    # Reconstruct through VAE
    print("Reconstructing images through VAE...")
    reconstructions = reconstruct_through_vae(vae, images, device)

    # Extract Inception features from reconstructions
    print("Loading Inception V3...")
    inception, resize = build_inception_extractor(device)
    features = extract_inception_features(reconstructions, inception, resize, device)

    # Filter NaN/Inf rows
    valid = np.isfinite(features).all(axis=1)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} samples with NaN/Inf in Inception features.")
    features = features[valid]
    labels_np = labels_np[valid]

    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(features)

    # Plot
    num_classes = len(class_names)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 9))

    for c in range(num_classes):
        mask = labels_np == c
        pts = embedding[mask]
        color = cmap(c / max(num_classes - 1, 1))

        ax.scatter(pts[:, 0], pts[:, 1],
                   color=color, s=12, alpha=0.65, edgecolors="none", label=class_names[c])

        if len(pts) >= 10:
            try:
                kde = gaussian_kde(pts.T, bw_method=0.35)
                xlo, xhi = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
                ylo, yhi = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(xlo, xhi, 250),
                                     np.linspace(ylo, yhi, 250))
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                threshold = np.percentile(kde(pts.T), 15)
                ax.contour(xx, yy, zz, levels=[threshold],
                           colors=[color], linewidths=1.8, alpha=0.95, zorder=3)
                ax.contourf(xx, yy, zz, levels=[threshold, zz.max()],
                            colors=["#888888"], alpha=0.13, zorder=2)
            except Exception:
                pass

    ax.legend(title="Class", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9,
              title_fontsize=10, markerscale=2)
    ax.set_title(args.title, fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
