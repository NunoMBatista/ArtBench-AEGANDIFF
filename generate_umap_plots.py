import os
import sys

# FIX: Force the project's internal modules to be found before third-party packages with the same name.
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import umap
except ImportError:
    print("Error: umap-learn is not installed.")
    print("Please install it by running: pip install umap-learn")
    sys.exit(1)

import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import Resize

from src.evaluate import EvalConfig, _load_model, _find_latest_checkpoint
from src.utils.data_loader import get_dataloaders
from src.utils.seed_setter import set_global_seed

def find_config_for_ckpt(ckpt_path):
    """Search for config.yml in the checkpoint's directory or its parent."""
    dir_path = os.path.dirname(ckpt_path)
    # Check current dir
    cfg = os.path.join(dir_path, "config.yml")
    if os.path.exists(cfg):
        return cfg
    # Check parent dir
    cfg = os.path.join(os.path.dirname(dir_path), "config.yml")
    if os.path.exists(cfg):
        return cfg
    return None

def plot_umap(features, labels, title, filename, is_categorical=True, class_names=None):
    print(f"Running UMAP for {title}...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    if is_categorical:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
        if class_names:
            import matplotlib.patches as mpatches
            cmap = plt.get_cmap('tab10')
            n_classes = len(class_names)
            handles = [mpatches.Patch(color=cmap(i/(n_classes-1) if n_classes > 1 else 0), label=class_names[i]) for i in range(n_classes)]
            plt.legend(handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        colors = ['blue' if l == 1 else 'red' for l in labels]
        plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=15, alpha=0.5)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Real (1)'),
                           Patch(facecolor='red', label='Fake (0)')]
        plt.legend(handles=legend_elements, loc='upper right')
        
    plt.title(title, fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate UMAP plots for generative models.")
    parser.add_argument("--vae_ckpt", type=str, default="", help="Path to VAE checkpoint (.pt)")
    parser.add_argument("--dcgan_ckpt", type=str, default="", help="Path to DCGAN checkpoint (.pt)")
    parser.add_argument("--diff_ckpt", type=str, default="", help="Path to Diffusion checkpoint (.pt)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to use")
    parser.add_argument("--no_attention", action="store_true", help="Disable attention in Diffusion model (fixes some mismatches)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(42)
    
    # 1. Load Data
    config = EvalConfig()
    num_samples = args.num_samples 
    
    print("Loading Real Data...")
    try:
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
        
        real_images = []
        real_labels = []
        with torch.no_grad():
            for batch, labels in train_loader:
                real_images.append(batch)
                real_labels.append(labels)
                if sum(len(b) for b in real_images) >= num_samples:
                    break
                    
        real_images = torch.cat(real_images, dim=0)[:num_samples].to(device)
        real_labels = torch.cat(real_labels, dim=0)[:num_samples].cpu().numpy()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    out_dir = "docs/report/images"
    
    # --- VAE UMAP ---
    print("\n--- Processing VAE ---")
    try:
        vae_ckpt = args.vae_ckpt or _find_latest_checkpoint("vae")
        print(f"Using VAE checkpoint: {vae_ckpt}")
        # Explicitly check for config to ensure latent_dim/base_channels match
        cfg_path = find_config_for_ckpt(vae_ckpt)
        if cfg_path:
            print(f"Loading config from: {cfg_path}")
            config.update(vae_ckpt) # Note: EvalConfig.update logic might need more flexibility but let's try
        
        vae = _load_model("vae", vae_ckpt, config.latent_dim, config.base_channels, device)
        
        with torch.no_grad():
            mu, _ = vae.encode(real_images)
            vae_features = mu.cpu().numpy()
            
        plot_umap(vae_features, real_labels, "VAE Latent Space", os.path.join(out_dir, "umap_vae.png"), True, class_names)
    except Exception as e:
        print(f"Skipping VAE UMAP: {e}")

    # --- DCGAN UMAP ---
    print("\n--- Processing DCGAN ---")
    try:
        dcgan_ckpt = args.dcgan_ckpt or _find_latest_checkpoint("dcgan")
        print(f"Using DCGAN checkpoint: {dcgan_ckpt}")
        config.update(dcgan_ckpt)
        dcgan = _load_model("dcgan", dcgan_ckpt, config.latent_dim, config.base_channels, device, use_spectral_norm=config.use_spectral_norm)
        
        with torch.no_grad():
            fake_images = dcgan.sample(num_samples, device)
            feature_extractor = nn.Sequential(*list(dcgan.discriminator.main.children())[:-2])
            real_feats = feature_extractor(real_images).view(real_images.size(0), -1).cpu().numpy()
            fake_feats = feature_extractor(fake_images).view(fake_images.size(0), -1).cpu().numpy()
            
        dcgan_features = np.concatenate([real_feats, fake_feats], axis=0)
        dcgan_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)
        plot_umap(dcgan_features, dcgan_labels, "DCGAN Features (Real vs Fake)", os.path.join(out_dir, "umap_dcgan.png"), False)
    except Exception as e:
        print(f"Skipping DCGAN UMAP: {e}")
        
    # --- Diffusion ---
    print("\n--- Processing Diffusion Model ---")
    try:
        diff_ckpt = args.diff_ckpt or _find_latest_checkpoint("diffusion")
        print(f"Using Diffusion checkpoint: {diff_ckpt}")
        config.update(diff_ckpt)
        if args.no_attention:
            print("Disabling attention for Diffusion model load...")
            config.use_attention = False

        diff_model = _load_model(
            "diffusion", diff_ckpt, config.latent_dim, config.base_channels, device,
            image_size=config.image_size, img_channels=config.img_channels, num_classes=config.num_classes,
            num_diffusion_steps=config.num_diffusion_steps, cfg_dropout=config.cfg_dropout,
            sample_steps=config.sample_steps, guidance_scale=config.guidance_scale,
            class_conditional=config.class_conditional, use_attention=config.use_attention
        )
        
        fake_images_diff = []
        batch_size = 64
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Sampling Diffusion"):
                curr_batch = min(batch_size, num_samples - i)
                fake_images_diff.append(diff_model.sample(curr_batch, device))
        fake_images_diff = torch.cat(fake_images_diff, dim=0)[:num_samples]
        
        print("Extracting Inception Features...")
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        inception.fc = nn.Identity()
        inception = inception.to(device).eval()
        resize = Resize((299, 299))
        
        def get_inception_feats(imgs):
            feats = []
            with torch.no_grad():
                for i in range(0, imgs.size(0), batch_size):
                    batch = imgs[i:i+batch_size]
                    if batch.min() < 0: batch = (batch + 1.0) / 2.0
                    batch = resize(batch)
                    feats.append(inception(batch).cpu().numpy())
            return np.concatenate(feats, axis=0)
            
        real_inc_feats = get_inception_feats(real_images)
        fake_inc_feats = get_inception_feats(fake_images_diff)
        diff_features = np.concatenate([real_inc_feats, fake_inc_feats], axis=0)
        diff_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)
        plot_umap(diff_features, diff_labels, "Inception Features (Real vs Fake)", os.path.join(out_dir, "umap_diffusion.png"), False)
    except Exception as e:
        print(f"Skipping Diffusion UMAP: {e}")

if __name__ == "__main__":
    main()
