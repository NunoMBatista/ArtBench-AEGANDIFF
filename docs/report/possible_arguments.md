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
