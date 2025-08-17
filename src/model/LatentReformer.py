import torch
import torch.nn as nn
import torch.optim as optim

class LatentReformer(nn.Module):
    def __init__(self, bottleneck_h=14, bottleneck_w=14):
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
            nn.Conv2d(2, 1, kernel_size=3, padding=1),   # Now input has 2 channels after concat
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(1, 1, kernel_size=3, padding=1)
        )
        self.bottleneck_h = bottleneck_h
        self.bottleneck_w = bottleneck_w

    def forward(self, img, latent_bottleneck):
        x = self.encoder(img)  
        x = torch.cat([x, latent_bottleneck], dim=1)  
        x = self.decoder(x)
        return x
