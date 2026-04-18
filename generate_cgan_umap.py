"""
UMAP visualization for cGAN: real images as circles, fake images as squares,
both colored by class label. Uses 5000 samples of each.

Usage:
    python generate_cgan_umap.py [--ckpt PATH] [--num_samples 5000]
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
import matplotlib.patches as mpatches
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


def extract_disc_features(model, images, labels, device, batch_size=64):
    feature_extractor = nn.Sequential(*list(model.discriminator.main.children())[:-1])
    feature_extractor.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            img_b = images[i:i + batch_size].to(device)
            lbl_b = labels[i:i + batch_size].to(device)
            emb = model.discriminator.label_emb(lbl_b)
            emb = emb.view(emb.size(0), 1, model.discriminator.img_size, model.discriminator.img_size)
            x = torch.cat([img_b, emb], dim=1)
            f = feature_extractor(x).view(img_b.size(0), -1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Path to cGAN checkpoint")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--out", type=str, default="docs/report/images/umap_cgan.png", help="Output file path")
    parser.add_argument("--title", type=str, default="cGAN — UMAP (Discriminator Features)\nCircles = Real, Squares = Fake")
    parser.add_argument("--run_prefix", type=str, default="cgan", help="Run prefix for auto checkpoint discovery")
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

    model = _load_model(
        "cgan", ckpt_path,
        config.latent_dim, config.base_channels, device,
        num_classes=config.num_classes,
        use_spectral_norm=config.use_spectral_norm,
    )

    num_samples = args.num_samples
    num_classes = config.num_classes

    # --- Load real images with labels ---
    print("Loading real images...")
    train_loader, _, class_names = get_dataloaders(
        batch_size=64,
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

    real_imgs, real_labels = [], []
    with torch.no_grad():
        for imgs, lbls in train_loader:
            real_imgs.append(imgs)
            real_labels.append(lbls)
            if sum(x.size(0) for x in real_imgs) >= num_samples:
                break
    real_imgs = torch.cat(real_imgs, dim=0)[:num_samples]
    real_labels_np = torch.cat(real_labels, dim=0)[:num_samples].numpy()

    # --- Generate fake images with known class labels ---
    print("Generating fake images...")
    samples_per_class = num_samples // num_classes
    fake_imgs_list, fake_labels_list = [], []
    model.eval()
    with torch.no_grad():
        for c in tqdm(range(num_classes), desc="classes"):
            remaining = samples_per_class
            while remaining > 0:
                batch = min(64, remaining)
                z = torch.randn(batch, model.latent_dim, device=device)
                lbls = torch.full((batch,), c, dtype=torch.long, device=device)
                imgs = model.generator(z, lbls)
                fake_imgs_list.append(imgs.cpu())
                fake_labels_list.append(lbls.cpu())
                remaining -= batch
    fake_imgs = torch.cat(fake_imgs_list, dim=0)
    fake_labels_np = torch.cat(fake_labels_list, dim=0).numpy()

    # --- Extract discriminator features ---
    print("Extracting discriminator features...")
    real_feats = extract_disc_features(model, real_imgs, torch.from_numpy(real_labels_np).long(), device)
    fake_feats = extract_disc_features(model, fake_imgs, torch.from_numpy(fake_labels_np).long(), device)

    all_feats = np.concatenate([real_feats, fake_feats], axis=0)
    all_labels = np.concatenate([real_labels_np, fake_labels_np], axis=0)
    is_real = np.concatenate([np.ones(len(real_feats)), np.zeros(len(fake_feats))], axis=0)

    # --- Run UMAP ---
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(all_feats)

    # --- Plot ---
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(12, 9))

    real_idx = is_real == 1
    fake_idx = is_real == 0

    scatter_real = ax.scatter(
        embedding[real_idx, 0], embedding[real_idx, 1],
        c=all_labels[real_idx], cmap="tab10", vmin=0, vmax=num_classes - 1,
        s=18, alpha=0.6, marker="o", edgecolors="none",
    )
    scatter_fake = ax.scatter(
        embedding[fake_idx, 0], embedding[fake_idx, 1],
        c=all_labels[fake_idx], cmap="tab10", vmin=0, vmax=num_classes - 1,
        s=18, alpha=0.6, marker="s", edgecolors="none",
    )

    # Legend: class colors + real/fake markers
    class_handles = [
        mpatches.Patch(color=cmap(i / (num_classes - 1)), label=class_names[i])
        for i in range(num_classes)
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Real"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="Fake"),
    ]
    legend1 = ax.legend(handles=class_handles, title="Class", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, title="Type", bbox_to_anchor=(1.01, 0.3), loc="upper left", fontsize=9)

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
