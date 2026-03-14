# Possible Arguments

## Variational Autoencoder

#### Posterior Collapse

- This could explain reconstructions that look similar across inputs (washed out, averaged or everly smooth/blurry without style details)
- The encoder ignores inputs and outputs a latent close to the prior (N(0, I), in our case).
- KL term goes near zero, so the decoder learns to reconstruct without using z.
- Possible Mitigations: KL warm-up/annealing, lower beta, increase model capacity.
- Obviously, if we change the beta to something different than 1, this would technically be a beta-VAE.
