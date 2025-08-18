import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.model.Aux import LatentNet
from src.model.LatentReformer import LatentReformer
from src.model.mnist_cnn import MNIST_CNN
from src.utils import train_latent_autoencoder, evaluate_f1_topo_vs_reconstruction
from dataloader import get_mnist_topo_loaders, get_advmnist_topo_loaders
import numpy as np

def train_model(config):

    train_loader, val_loader = get_mnist_topo_loaders("/kaggle/input/invi_mnist_64/pytorch/default/1/mnist_train_complete.npz")

    bottleneck_h = 14
    bottleneck_w = 14
    latent_dim = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_reformer = LatentReformer(bottleneck_h=bottleneck_h, bottleneck_w=bottleneck_w)
    latent_nn = LatentNet(latent_dim=latent_dim, bottleneck_h=bottleneck_h, bottleneck_w=bottleneck_w)

    latent_reformer.to(device)
    latent_nn.to(device)


    model = MNIST_CNN()
    model.load_state_dict(torch.load('/kaggle/input/classifiers/Pretrained_classifiers/mnist.pth', map_location='cpu'))
    model.to(device)
    model.eval()

    latent_reformer,latent_nn,simple_classifier = train_latent_autoencoder(latent_reformer, latent_nn, model,simple_classifier, train_loader, val_loader, epochs=50, lr=1e-3, device=device)

    evaluate_f1_topo_vs_reconstruction(model, latent_reformer, latent_nn, val_loader, device)
    
    adv_loader = get_advmnist_topo_loaders("/kaggle/input/invi_mnist_64/pytorch/default/1/adversarial_mnist_cw_strong_complete.npz")

    evaluate_f1_topo_vs_reconstruction(model, latent_reformer, latent_nn, adv_loader, device)