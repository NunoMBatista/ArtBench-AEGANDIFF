import torch
from torch import nn

"""
cGAN (Conditional Generative Adversarial Network)

This implementation extends DCGAN with class-conditional generation following
Mirza & Osindero (2014). Both Generator and Discriminator receive class label
information so the model can generate images of a specific art style.

Key Differences from DCGAN:
1.  **Generator Conditioning**: The class label is embedded and concatenated with
    the noise vector before being fed into the transposed convolution stack.
    Rationale: This allows the generator to learn style-specific features by
    conditioning the entire generation process on the target class.
2.  **Discriminator Conditioning**: The class label is embedded, projected to a
    spatial map, and concatenated with the image as an extra channel.
    Rationale: This lets the discriminator judge not just "is this image real?"
    but "is this image a realistic example of the given class?"
3.  **Architecture**: Otherwise follows the same DCGAN heuristics — strided
    convolutions, batch normalization, ReLU/LeakyReLU activations, Tanh output.
"""


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int = 10,
        embed_dim: int = 64,
        img_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        in_channels = latent_dim + embed_dim

        # Input: (batch, latent_dim + embed_dim, 1, 1)
        self.main = nn.Sequential(
            # Layer 1: to 4x4
            nn.ConvTranspose2d(in_channels, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(),
            # Layer 2: 4x4 to 8x8
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            # Layer 3: 8x8 to 16x16
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            # Layer 4: 16x16 to 32x32
            nn.ConvTranspose2d(base_channels * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # z: (batch, latent_dim) or (batch, latent_dim, 1, 1)
        if z.ndim == 4:
            z = z.view(z.size(0), z.size(1))
        emb = self.label_emb(labels)  # (batch, embed_dim)
        x = torch.cat([z, emb], dim=1)  # (batch, latent_dim + embed_dim)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_channels: int = 3,
        img_size: int = 32,
        base_channels: int = 64,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        # Project label to a full spatial channel
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        def conv_block(in_f, out_f, kernel, stride, padding, use_bn=True):
            layers = []
            conv = nn.Conv2d(in_f, out_f, kernel, stride, padding, bias=False)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            if use_bn:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        # Input: (batch, img_channels + 1, 32, 32) — extra channel from label map
        self.main = nn.Sequential(
            # 32x32 -> 16x16
            conv_block(img_channels + 1, base_channels, 4, 2, 1, use_bn=False),
            # 16x16 -> 8x8
            conv_block(base_channels, base_channels * 2, 4, 2, 1),
            # 8x8 -> 4x4
            conv_block(base_channels * 2, base_channels * 4, 4, 2, 1),
            # 4x4 -> 1x1
            nn.Conv2d(base_channels * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        # Embed label as a spatial channel and concatenate with image
        emb = self.label_emb(labels)  # (batch, img_size * img_size)
        emb = emb.view(emb.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([x, emb], dim=1)  # (batch, img_channels + 1, H, W)
        return self.main(x).view(-1, 1)


class cGAN(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        embed_dim: int = 64,
        img_channels: int = 3,
        img_size: int = 32,
        base_channels: int = 64,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.generator = Generator(latent_dim, num_classes, embed_dim, img_channels, base_channels)
        self.discriminator = Discriminator(num_classes, img_channels, img_size, base_channels, use_spectral_norm)

    def forward(self, z, labels):
        return self.generator(z, labels)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
        return self.generator(z, labels)


def cgan_loss(d_out, target):
    return nn.functional.binary_cross_entropy(d_out.view(-1), target.view(-1))
