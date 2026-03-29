# Best Hyperparameters per Model

Results from manual hyperparameter exploration on the 20% ArtBench subset.
All runs use `seed=67` for training. FID/KID are from the end-of-training evaluation
(single seed, 5000 generated vs 5000 real images).

---

## VAE

**Best FID: 339.9 | KID: 0.3121**

| Parameter | Value |
|-----------|-------|
| `model_type` | `vae` |
| `base_channels` | 128 |
| `latent_dim` | 256 |
| `beta` | 0.5 |
| `lr` | 0.002 |
| `batch_size` | 256 |
| `epochs` | 20 |
| `optimizer` | Adam (betas=[0.9, 0.999], weight_decay=0) |

**Notes:** `beta=0.5` (below standard β=1) prioritises reconstruction over KL
regularisation, which helps sample quality at the cost of disentanglement. Larger
`latent_dim=256` and `base_channels=128` give the model more capacity, which is
needed given ArtBench's 10-class visual diversity. VAE results are significantly
worse than diffusion/DCGAN — characteristic blurriness from the Gaussian decoder.

---

## DCGAN

**Best FID: 120.8 | KID: 0.0960**

| Parameter | Value |
|-----------|-------|
| `model_type` | `dcgan` |
| `base_channels` | 128 |
| `latent_dim` | 64 |
| `lr` | 0.0002 |
| `d_updates_per_g` | 1 |
| `use_spectral_norm` | false |
| `batch_size` | 128 |
| `epochs` | 20 |
| `optimizer` | Adam (betas=[0.0, 0.999], weight_decay=0) |

**Notes:** Classic DCGAN defaults (lr=2e-4, β₁=0, β₂=0.999) work well. Spectral
normalisation gave no improvement here (FID 120.8 vs 121.9 without it).
`d_updates_per_g=1` is sufficient — more D steps slightly hurt performance.
Best GAN variant overall, comfortably outperforming cGAN.

---

## cGAN

**Best FID: 301.6 | KID: 0.2562**

| Parameter | Value |
|-----------|-------|
| `model_type` | `cgan` |
| `base_channels` | 128 |
| `latent_dim` | 128 |
| `embed_dim` | 64 |
| `num_classes` | 10 |
| `lr` | 5e-05 |
| `d_updates_per_g` | 5 |
| `use_spectral_norm` | false |
| `batch_size` | 128 |
| `epochs` | 20 |
| `optimizer` | Adam (betas=[0.0, 0.999], weight_decay=0) |

**Notes:** Much lower lr than DCGAN (5e-5 vs 2e-4) and more D steps per G step
(5 vs 1) needed for stability with class conditioning. Despite these, unconditional
DCGAN significantly outperforms cGAN on this dataset — the class-conditioning
signal may be too noisy to help on only 20% of the data with 20 epochs.

---

## Diffusion (TinyUNet DDPM)

**Best FID: 71.4 | KID: 0.0388**

| Parameter | Value |
|-----------|-------|
| `model_type` | `diffusion` |
| `base_channels` | 128 |
| `latent_dim` | 128 |
| `lr` | 0.0002 |
| `epochs` | 50 |
| `use_attention` | false |
| `class_conditional` | false |
| `num_diffusion_steps` | 1000 |
| `sample_steps` | 100 |
| `guidance_scale` | 2.0 |
| `cfg_dropout` | 0.1 |
| `batch_size` | 32 |
| `optimizer` | AdamW (betas=[0.9, 0.999], weight_decay=1e-4) |

**Notes:** Best model overall. Attention at the bottleneck gave nearly identical
results (FID 71.8 with attention vs 71.4 without) at higher cost, so attention is
disabled. Unconditional generation (`class_conditional=false`) works better with
limited data. 50 epochs beats 40 epochs. Sweep runs with `base_channels=64` or
only 20 epochs score FID~150, confirming capacity and training length matter.

---

## Google DDPM (fine-tuned `google/ddpm-cifar10-32`)

**Best FID: 75.1 | KID: 0.0661**

| Parameter | Value |
|-----------|-------|
| `model_type` | `google_ddpm` |
| `pretrained_model_id` | `google/ddpm-cifar10-32` |
| `lr` | 0.0005 |
| `epochs` | 10 |
| `num_diffusion_steps` | 1000 |
| `sample_steps` | 100 |
| `batch_size` | 32 |
| `optimizer` | AdamW (weight_decay=1e-4) |

**Notes:** Fine-tuning a CIFAR-10 pretrained UNet on ArtBench. Very close to the
custom diffusion model (75.1 vs 71.4 FID) with only 10 fine-tuning epochs. Higher
lr (5e-4) outperforms conservative values (5e-5: FID 92.4, 2e-4: FID 89.5),
suggesting the model needs to adapt substantially from CIFAR-10 priors to ArtBench.

---

## Summary Table

| Model | FID ↓ | KID ↓ | Notes |
|-------|-------|-------|-------|
| Diffusion (TinyUNet) | **71.4** | **0.0388** | Best overall |
| Google DDPM (fine-tuned) | 75.1 | 0.0661 | Close second |
| DCGAN | 120.8 | 0.0960 | Best GAN |
| cGAN | 301.6 | 0.2562 | Worse than DCGAN |
| VAE | 339.9 | 0.3121 | Worst, as expected |

*All results: single seed (67), 20% data subset, 5000 samples.*
*Final results (10 seeds, mean ± std) still to be computed — use `src/final_eval.py`.*
