## VAE

### Hyperparameter Sweep (Bayesian, wandb)

| Parameter | Values | Justification |
|---|---|---|
| `lr` | {5e-5, 1e-4, 5e-4, 1e-3, 2e-3} | Learning rate is the single most impactful training hyperparameter. Too high causes instability, too low causes slow convergence. Log-spaced values cover the practical range for Adam on image VAEs. |
| `latent_dim` | {64, 128, 256} | Controls the bottleneck capacity. Too small and the model cannot encode enough variation; too large and the KL term struggles to regularize, leading to posterior collapse or blurry samples. |
| `base_channels` | {64, 128} | Determines the width of every conv layer (scales as 1x, 2x, 4x, 8x). Wider networks have more capacity but are slower. Two values are enough since the architecture already scales internally. |
| `beta` | {0.5, 1, 2, 4} | The KL weight (beta-VAE). beta=1 is the standard ELBO. beta<1 favors reconstruction quality (sharper but less disentangled). beta>1 encourages disentanglement at the cost of blurrier outputs. Sweeping this is critical because the reconstruction-vs-KL tradeoff directly governs sample quality and diversity. |
| `batch_size` | {64, 128, 256} | Affects gradient noise and effective learning rate. Larger batches give smoother gradients but can hurt generalization; smaller batches add regularization but slow training. |

**Metric:** FID (minimize)
**Stopping criterion:** Convergence of best FID (no improvement in ~10 consecutive runs), diminishing returns after ~25 runs, or a total of ~50 runs, whichever comes first.
**Search space size:** 5 × 3 × 2 × 4 × 3 = 360 combinations. Bayesian optimization does not need to explore all of them — ~10× the number of tuned parameters (~50 runs) is a standard heuristic for GP-based BO to converge.

### Final evaluation (best config from sweep)

- Train on 20% subset with the best hyperparameters from the sweep.
- Evaluate with 10 different seeds, each generating 5000 samples vs 5000 real images.
- Report FID (mean ± std) and KID (mean ± std, 50 subsets of 100).

---

## DCGAN & cGAN

### Hyperparameter Sweep (Bayesian, wandb)

| Parameter | Values | Justification |
|---|---|---|
| `lr` | {5e-5, 2e-4, 1e-3} | GAN training is notoriously sensitive to learning rate — too high destabilizes the D/G equilibrium, too low and the generator fails to converge. 2e-4 is the classic DCGAN default (Radford et al.), but we sweep around it since our dataset differs from LSUN/CelebA. |
| `latent_dim` | {64, 100, 128} | The noise vector dimensionality. 100 is the original DCGAN default. Smaller vectors may not capture enough variation, larger ones are harder for the generator to map meaningfully. |
| `base_channels` | {64, 128} | Controls generator and discriminator capacity symmetrically. Both G and D must be balanced in capacity — sweeping this tests whether more capacity helps on ArtBench's multi-style distribution. |
| `d_updates_per_g` | {1, 2, 5} | How many discriminator steps per generator step. 1:1 is standard DCGAN, but if D is too weak it provides poor gradients to G. More D steps can stabilize training at the cost of slower G improvement. This is important to sweep because the optimal ratio depends on relative D/G capacity. |
| `use_spectral_norm` | {true, false} | Spectral normalization constrains the Lipschitz constant of D, preventing it from becoming too powerful too quickly. This is one of the most effective GAN stabilization techniques but can also limit D's expressiveness. Testing both reveals which regime works better for ArtBench. |

**Metric:** FID (minimize)
**Stopping criterion:** Convergence of best FID, diminishing returns after ~15 runs, or ~30 runs.
**Search space size:** 3 × 3 × 2 × 3 × 2 = 108 combinations.

### Final evaluation (best config from sweep)

- Train on 20% subset with the best hyperparameters from the sweep.
- Evaluate with 10 different seeds, each generating 5000 samples vs 5000 real images.
- Report FID (mean ± std) and KID (mean ± std, 50 subsets of 100).

---

### Final evaluation (best config from sweep)

- Train on 20% subset with the best hyperparameters from the sweep.
- Evaluate with 10 different seeds, each generating 5000 samples vs 5000 real images.
- Report FID (mean ± std) and KID (mean ± std, 50 subsets of 100).

---

## Diffusion Model (custom TinyUNet DDPM)

Diffusion runs are much more expensive than VAE/DCGAN (iterative denoising at evaluation, larger model). The sweep is limited to the 3 parameters with the highest impact on final sample quality.

### Hyperparameter Sweep (Bayesian, wandb)

| Parameter | Values | Justification |
|---|---|---|
| `lr` | {5e-5, 2e-4, 5e-4} | The most impactful training hyperparameter. Diffusion MSE training is simpler than GAN dynamics, but the optimal lr still varies with architecture size and dataset. Range is centered around the 1e-4 to 2e-4 sweet spot from the DDPM literature. |
| `base_channels` | {64, 128} | The main capacity lever in the TinyUNet (only 2 down/up stages). 128 gives ~4× more parameters than 64 at the bottleneck. This directly determines whether the model can capture ArtBench's multi-style distribution. |
| `use_attention` | {true, false} | Self-attention at the bottleneck captures long-range spatial coherence (global composition in paintings). On 32×32 the overhead is manageable but not free. This is the key architectural decision for quality vs cost. |

**Metric:** FID (minimize)
**Stopping criterion:** Convergence of best FID, diminishing returns after ~10 runs, or ~20 runs.
**Search space size:** 3 × 2 × 2 = 12 combinations (small enough that bayesian search covers it thoroughly).

**Fixed values (not swept):** `num_diffusion_steps: 1000` (standard, no reason to deviate), `sample_steps: 100` (good DDIM default, can be tuned cheaply in a quick ablation after the sweep), `guidance_scale: 2.0` (only matters for class-conditional; can be ablated post-sweep with a single trained model).

**Note:** Reduce `eval_num_samples` to 1000 and `epochs` to 20 for sweep runs. Retrain the best config with full evaluation settings.

### Final evaluation (best config from sweep)

- Train on 20% subset with the best hyperparameters from the sweep.
- Evaluate with 10 different seeds, each generating 5000 samples vs 5000 real images.
- Report FID (mean ± std) and KID (mean ± std, 50 subsets of 100).

---

## Google DDPM (fine-tuned `google/ddpm-cifar10-32`)

Architecture is fixed (pretrained HF UNet), so only training dynamics need tuning. Runs are expensive due to the large UNet + iterative sampling, so the sweep is kept minimal.

### Hyperparameter Sweep (Bayesian, wandb)

| Parameter | Values | Justification |
|---|---|---|
| `lr` | {1e-5, 5e-5, 2e-4, 5e-4, 1e-2} | The most critical fine-tuning parameter. Too high causes catastrophic forgetting of the CIFAR-10 priors; too low means the model barely adapts to ArtBench. The range spans conservative (1e-5) to aggressive (2e-4, the from-scratch default). |
| `epochs` | {10, 20, 40} | Fine-tuning on only 20% of ArtBench can overfit quickly. Too many epochs may destroy the pretrained knowledge without learning ArtBench well. This parameter directly controls the forgetting-vs-adaptation tradeoff. |

**Metric:** FID (minimize)
**Stopping criterion:** Convergence of best FID, diminishing returns after ~10 runs, or ~20 runs.
**Search space size:** 5 × 3 = 15 combinations (exhaustively coverable).

**Fixed values (not swept):** `sample_steps: 100` (can ablate post-sweep cheaply), `weight_decay: 1e-4` (standard AdamW default, low risk of being wrong). `batch_size: 32` (limited by model size and memory).

### Final evaluation (best config from sweep)

- Train on 20% subset with the best hyperparameters from the sweep.
- Evaluate with 10 different seeds, each generating 5000 samples vs 5000 real images.
- Report FID (mean ± std) and KID (mean ± std, 50 subsets of 100).

---

## Best Model: Full Dataset Training

After comparing all model families on the 20% subset:

1. Select the best-performing model family based on FID/KID results.
2. Retrain on the **full 50k training set** using the best hyperparameters from the sweep.
3. Final evaluation: 10 seeds × 5000 generated vs 5000 real. Report FID and KID (mean ± std).
