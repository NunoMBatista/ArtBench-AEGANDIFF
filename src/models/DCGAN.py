import torch
from torch import nn

"""
DCGAN (Deep Convolutional Generative Adversarial Network) 

This implementation follows the architectural heuristics described in Radford et al. (2015) 
and the specific course guidelines provided in the PDF.

Key Design Choices & Rationale:
1.  **Architecture**: We use a fully convolutional architecture. Pooling layers are replaced with 
    strided convolutions in the Discriminator and transposed convolutions in the Generator. 
    Rationale: This allows the network to learn its own downsampling and upsampling parameters.
2.  **Activations**: ReLU is used in the Generator (except for the output layer), and LeakyReLU 
    is used in the Discriminator. 
    Rationale: LeakyReLU in the Discriminator helps prevent the 'dying ReLU' problem, which is 
    crucial for providing gradients to the Generator even when the sample is clearly fake.
3.  **Normalization**: Batch Normalization is applied to both G and D (except for the final layer 
    of G and the first layer of D). 
    Rationale: This stabilizes training by normalizing the input to each layer to have 
    zero mean and unit variance, which helps mitigate initialization issues and prevents 
    gradients from vanishing/exploding.
4.  **Output**: Tanh is used as the final activation in the Generator. 
    Rationale: This maps the output to [-1, 1], which matches the normalization of the 
    training images.
5.  **Spectral Normalization**: Optional support in the Discriminator.
    Rationale: Highly recommended for stability as it constrains the Lipschitz constant of 
    the Discriminator, preventing it from becoming too strong too quickly.
"""

class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_channels: int = 3, base_channels: int = 64):
        super().__init__()
        # Input: (batch, latent_dim, 1, 1)
        self.main = nn.Sequential(
            # Layer 1: Latent to 4x4
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
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
            nn.Tanh()
        )

    def forward(self, z):
        if z.ndim == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels: int = 3, base_channels: int = 64, use_spectral_norm: bool = False):
        super().__init__()
        
        def conv_block(in_f, out_f, kernel, stride, padding, use_bn=True):
            layers = []
            conv = nn.Conv2d(in_f, out_f, kernel, stride, padding, bias=False)
            if use_spectral_norm:
                # Customization Note: Spectral Norm helps D be less sensitive to noise
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            if use_bn:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        # Input: (batch, 3, 32, 32)
        self.main = nn.Sequential(
            # 32x32 -> 16x16
            conv_block(img_channels, base_channels, 4, 2, 1, use_bn=False),
            # 16x16 -> 8x8
            conv_block(base_channels, base_channels * 2, 4, 2, 1),
            # 8x8 -> 4x4
            conv_block(base_channels * 2, base_channels * 4, 4, 2, 1),
            # 4x4 -> 1x1
            nn.Conv2d(base_channels * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)

class DCGAN(nn.Module):
    def __init__(self, latent_dim: int = 100, img_channels: int = 3, base_channels: int = 64, use_spectral_norm: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, img_channels, base_channels)
        self.discriminator = Discriminator(img_channels, base_channels, use_spectral_norm)

    def forward(self, z):
        # Implementation Note: Usually GANs don't use 'forward' for joint inference, 
        # but we provide it for compatibility with simple profiling if needed.
        return self.generator(z)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z)

def dcgan_loss(d_out, target):
    """
    Standard GAN Loss using binary cross entropy.
    For Discriminator: 
        - Real images should have target 1
        - Fake images should have target 0
    For Generator:
        - Generated images should aim for target 1 (fooling the discriminator)
    """
    return nn.functional.binary_cross_entropy(d_out.view(-1), target.view(-1))