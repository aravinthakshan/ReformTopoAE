import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np

class MNISTTopoDataset(Dataset):
    def __init__(self, clean_images, topo_images, labels, latents):
        self.clean_images = clean_images
        self.topo_images = topo_images
        self.labels = labels
        self.latents = latents

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean = torch.tensor(self.clean_images[idx], dtype=torch.float32)
        topo = torch.tensor(self.topo_images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        latent = torch.tensor(self.latents[idx], dtype=torch.float32)
        return clean, topo, label, latent
    
def get_mnist_topo_loaders(npz_path, batch_size=64, val_split=0.1):
    data = np.load(npz_path)
    clean_images = data['clean']
    topo_images = data['reconstructed']
    labels = data['label']
    latents = data['latent']

    full_dataset = MNISTTopoDataset(clean_images, topo_images, labels, latents)

    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def get_advmnist_topo_loaders(npz_path, batch_size=64, val_split=0.1):
    data = np.load(npz_path)
    clean_images = data['original_images']
    topo_images = data['reconstructed_images']
    labels = data['labels']
    latents = data['latents']

    full_dataset = MNISTTopoDataset(clean_images, topo_images, labels, latents)

    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    return full_loader