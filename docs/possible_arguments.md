# Possible Arguments

## Variational Autoencoder

#### Posterior Collapse

- The inputs of the VAE with base_channels = 64 and beta = 1 were looking like just brown noise.
- Posterior collapse could explain reconstructions that look similar across inputs (washed out, averaged or everly smooth/blurry without style details)
- The encoder ignores inputs and outputs a latent close to the prior (N(0, I), in our case).
- KL term goes near zero, so the decoder learns to reconstruct without using z.
- Possible Mitigations: KL warm-up/annealing, lower beta, increase model capacity.
- Obviously, if we change the beta to something different than 1, this would technically be a beta-VAE.

## Google DDPM Fine-Tuning (google/ddpm-cifar10-32)

- Freeze encoder + bottleneck (`down_blocks` and `mid_block`), train only decoder (`up_blocks`).
	Reason: preserve pretrained low-level structure priors and adapt mainly style/texture decoding.
- Use a low learning rate (`1e-5` or `5e-6`).
	Reason: reduces catastrophic forgetting of pretrained diffusion weights.
- Keep scheduler identical to pretrained checkpoint (`linear`, `1000` timesteps).
	Reason: UNet weights are coupled to the training noise schedule; changing it during fine-tune can destabilize training.
- Train unconditionally (ignore class labels).
	Reason: the pretrained model is unconditional, and this gives a fair comparison against the repo's unconditional diffusion baseline.

#### Observed Behavior During Fine-Tune

- Better perceived shapes can still get worse FID/KID at 32x32.
	Reason: Inception-based metrics at this resolution over-weight local texture and color statistics versus global semantic structure.
- From-scratch diffusion can score better while looking less meaningful.
	Reason: it can match ArtBench low-level texture statistics closely, which metrics reward, even when samples are mostly texture-like.
- Fine-tuned pretrained diffusion may keep CIFAR-like structural priors with ArtBench style.
	Reason: this hybrid distribution can be visually better for humans but statistically farther from ArtBench under Inception features.
- Nearly flat diffusion loss does not imply no learning.
	Reason: diffusion loss is noise-prediction MSE, not a direct quality metric; a pretrained denoiser can start near a low-loss regime and then make small stylistic updates.
