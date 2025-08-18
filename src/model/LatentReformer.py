import torch
import torch.nn as nn
import torch.optim as optim

class LatentReformer(nn.Module):
    def __init__(self, bottleneck_h=14, bottleneck_w=14, use_null_latent=True):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),   # Input has 2 channels after concat
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(1, 1, kernel_size=3, padding=1)
        )
        self.bottleneck_h = bottleneck_h
        self.bottleneck_w = bottleneck_w
        self.use_null_latent = use_null_latent

    def forward(self, img, latent_bottleneck=None):
        x = self.encoder(img)  

        if self.use_null_latent:
            # Create a zero tensor with same shape as latent_bottleneck
            B, _, H, W = x.shape
            latent_bottleneck = torch.zeros(
                (B, 1, self.bottleneck_h, self.bottleneck_w),
                device=x.device, dtype=x.dtype
            )

        x = torch.cat([x, latent_bottleneck], dim=1)  
        x = self.decoder(x)
        return x
