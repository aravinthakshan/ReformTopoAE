import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                    
            nn.Linear(14*14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)               
        )

    def forward(self, x):
        return self.model(x)
