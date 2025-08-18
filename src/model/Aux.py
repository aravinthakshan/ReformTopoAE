import torch
import torch.nn as nn
import torch.optim as optim

class LatentNet(nn.Module):
    # Small MLP to process latent vector of shape [batch_size, 2]
    def __init__(self, latent_dim=2, bottleneck_h=16, bottleneck_w=16):
        super().__init__()
        # The fully connected neural net to expand latent -> bottleneck size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_h * bottleneck_w)
        )
        self.bottleneck_h = bottleneck_h
        self.bottleneck_w = bottleneck_w

    def forward(self, z):
        out = self.fc(z)                 # (batch_size, bottleneck_h * bottleneck_w)
        out = out.view(-1, 1, self.bottleneck_h, self.bottleneck_w)  # (batch_size, 1, H, W)
        return out # 196
    
