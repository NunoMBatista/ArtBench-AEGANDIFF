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
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

# Ensure we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluate import EvalConfig, _load_model, _find_latest_checkpoint

def interpolate(z1, z2, steps=10):
    """Linearly interpolate between two vectors."""
    alphas = torch.linspace(0, 1, steps).to(z1.device)
    # View to allow broadcasting: (steps, 1)
    alphas = alphas.view(-1, 1)
    # (steps, latent_dim)
    interpolated_z = (1 - alphas) * z1 + alphas * z2
    return interpolated_z

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup config and find latest DCGAN checkpoint
    model_type = "dcgan"
    try:
        ckpt_path = _find_latest_checkpoint(model_type)
        print(f"Found latest {model_type} checkpoint: {ckpt_path}")
    except FileNotFoundError:
        print(f"No checkpoint found for {model_type}. Please train the model first.")
        return

    config = EvalConfig()
    config.update(ckpt_path)
    
    # 2. Load the model
    # Note: We use config values. If the checkpoint was trained with different ones,
    # you might need to pass them explicitly as discussed before.
    try:
        model = _load_model(
            model_type, 
            ckpt_path, 
            config.latent_dim, 
            config.base_channels, 
            device,
            use_spectral_norm=config.use_spectral_norm
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: If you see a size mismatch, try retraining with current code or overriding base_channels.")
        return

    model.eval()

    # 3. Create interpolation paths
    # We will create ONE smooth transition with 21 steps
    num_steps = 21
    
    with torch.no_grad():
        # Sample two random points in latent space
        z1 = torch.randn(1, config.latent_dim, device=device)
        z2 = torch.randn(1, config.latent_dim, device=device)
        
        # Interpolate
        z_path = interpolate(z1, z2, steps=num_steps)
        
        # Generate images
        grid_tensor = model.generator(z_path)
            
    # Normalize to [0, 1] for saving
    grid_tensor = (grid_tensor + 1.0) / 2.0
    grid_tensor = grid_tensor.clamp(0.0, 1.0)
    
    # Create the output directory
    output_dir = os.path.join(project_root, "docs", "report", "images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dcgan_interpolation_smooth.png")
    
    # Save the grid as 3 rows of 7 images
    save_image(grid_tensor, output_path, nrow=7, padding=2)
    print(f"Smooth interpolation grid saved to: {output_path}")

if __name__ == "__main__":
    main()
