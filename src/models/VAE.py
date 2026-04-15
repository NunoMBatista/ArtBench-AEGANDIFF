import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        img_channels: int = 3,
        base_channels: int = 64,
        decoder_activation: str = "tanh",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # Higher spatial resolution at the bottleneck (8x8 instead of 2x2) 
        # to preserve texture and color details.
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 32x32 -> 16x16 -> 8x8
        self.enc_out_dim = base_channels * 2 * 8 * 8

        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)

        if decoder_activation.lower() == "tanh":
            final_act = nn.Tanh()
        elif decoder_activation.lower() == "sigmoid":
            final_act = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported decoder_activation: {decoder_activation}")

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels, img_channels, kernel_size=4, stride=2, padding=1),
            final_act,
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), self.base_channels * 2, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    # Mirror the friend's stable math: sum then divide ONLY by batch size.
    batch_size = x.size(0)
    
    recon_loss_sum = F.mse_loss(recon, x, reduction="sum")
    kl_loss_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    recon_loss = recon_loss_sum / batch_size
    kl_loss = kl_loss_sum / batch_size

    total = recon_loss + beta * kl_loss
    
    return total, {
        "recon_loss": float(recon_loss.item()),
        "kl_loss": float(kl_loss.item()),
    }
