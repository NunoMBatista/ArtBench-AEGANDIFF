import os
import sys

# FIX: Force the project's internal modules to be found before third-party packages with the same name.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from src.evaluate import EvalConfig, _load_model, _find_latest_checkpoint

@torch.no_grad()
def sample_with_steps(model, num_samples, device, num_steps=None, guidance_scale=None):
    """
    Modified sampling loop to capture exactly 21 intermediate denoising steps.
    """
    guidance = model.guidance_scale if guidance_scale is None else float(guidance_scale)
    sampling_steps = model.sample_steps if num_steps is None else int(num_steps)
    
    # We want 21 snapshots total. Capture 20 during the loop + 1 final.
    capture_indices = torch.linspace(0, sampling_steps - 1, 20).long().tolist()
    
    x = torch.randn(num_samples, model.img_channels, model.image_size, model.image_size, device=device)
    
    if model.class_conditional:
        labels = torch.randint(0, model.num_classes, (num_samples,), device=device)
    else:
        labels = None

    step_indices = torch.linspace(
        model.num_diffusion_steps - 1,
        0,
        sampling_steps,
        device=device,
    ).long()

    snapshots = []
    
    for i, t_scalar in enumerate(tqdm(step_indices, desc="Denoising")):
        if i in capture_indices:
            # Normalize to [0, 1] for visualization
            snap = x.clone().add(1.0).div(2.0).clamp(0.0, 1.0)
            snapshots.append(snap)

        t = torch.full((num_samples,), int(t_scalar.item()), device=device, dtype=torch.long)
        eps = model._predict_eps_with_cfg(x, t, labels, guidance)

        alpha_bar_t = model._gather(model.alphas_cumprod, t, x.shape)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t.clamp(min=1e-8)

        if i == len(step_indices) - 1:
            alpha_bar_next = torch.ones_like(alpha_bar_t)
        else:
            next_t = torch.full((num_samples,), int(step_indices[i + 1].item()), device=device, dtype=torch.long)
            alpha_bar_next = model._gather(model.alphas_cumprod, next_t, x.shape)

        x = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * eps

    # Add final result (the 21st snapshot)
    final = x.add(1.0).div(2.0).clamp(0.0, 1.0)
    snapshots.append(final)
    
    return snapshots

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_type = "diffusion"
    try:
        ckpt_path = _find_latest_checkpoint(model_type)
        print(f"Found latest {model_type} checkpoint: {ckpt_path}")
    except FileNotFoundError:
        print(f"No checkpoint found for {model_type}. Please train the model first.")
        return

    config = EvalConfig()
    config.update(ckpt_path)
    
    try:
        model = _load_model(
            model_type, 
            ckpt_path, 
            config.latent_dim, 
            config.base_channels, 
            device,
            image_size=config.image_size,
            img_channels=config.img_channels,
            num_classes=config.num_classes,
            num_diffusion_steps=config.num_diffusion_steps,
            cfg_dropout=config.cfg_dropout,
            sample_steps=config.sample_steps,
            guidance_scale=config.guidance_scale,
            class_conditional=config.class_conditional,
            use_attention=config.use_attention
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Generate snapshots for ONE single image
    snapshots = sample_with_steps(model, 1, device)
    # snapshots is a list of 21 tensors of shape [1, 3, 32, 32]
    final_grid = torch.cat(snapshots, dim=0) # [21, 3, 32, 32]
    
    output_dir = os.path.join(project_root, "docs", "report", "images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "diffusion_denoising_smooth.png")
    
    # Save as 3 rows of 7 images
    save_image(final_grid, output_path, nrow=7, padding=2)
    print(f"Smooth denoising visualization saved to: {output_path}")

if __name__ == "__main__":
    main()
