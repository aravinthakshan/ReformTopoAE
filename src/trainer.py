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


def train(
    epochs,
    batch_size,
    train_dir,
    test_dir,
    device,
    lr,
    dataset_name,
):
    print(f"Training on dataset: {dataset_name}")
    # Load dataset
    train_loader, val_loader = get_mnist_topo_loaders(train_dir, batch_size=batch_size)
    # Evaluate on adversarial set
    adv_loader = get_advmnist_topo_loaders(test_dir, batch_size=batch_size)

    bottleneck_h = 14
    bottleneck_w = 14
    latent_dim = 64

    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Pretrained classifier
    model = MNIST_CNN()
    model.load_state_dict(
        torch.load('/kaggle/input/classifiers/Pretrained_classifiers/mnist.pth', map_location="cpu")
    )
    model.to(device)
    model.eval()

    # Simple classifier on latent space
    # simple_classifier = nn.Linear(latent_dim, 10).to(device)

    # Just Classifier 
    # Add Code here for this 

    # Latent autoencoder parts
    latent_reformer = LatentReformer(bottleneck_h=bottleneck_h, bottleneck_w=bottleneck_w, use_null_latent=True).to(device)
    latent_nn = LatentNet(latent_dim=latent_dim, bottleneck_h=bottleneck_h, bottleneck_w=bottleneck_w).to(device)

    # Classifier + TopoAE
    # Classifier + TopoAE + Reformer 
    latent_reformer, latent_nn, simple_classifier = train_latent_autoencoder(
        latent_reformer, latent_nn, model,
        train_loader, val_loader,
        epochs=epochs, lr=lr, device=device
    )

    evaluate_f1_topo_vs_reconstruction(model, latent_reformer, latent_nn, val_loader, device)
    # Evaluate on validation set
    evaluate_f1_topo_vs_reconstruction(model, latent_reformer, latent_nn, val_loader, device)

    latent_reformer = LatentReformer(bottleneck_h=bottleneck_h, bottleneck_w=bottleneck_w).to(device)
    latent_nn = LatentNet(latent_dim=latent_dim, bottleneck_h=bottleneck_h, bottleneck_w=bottleneck_w).to(device)


    # Classifier + TopoAE + Reformer + AUX
    latent_reformer, latent_nn, simple_classifier = train_latent_autoencoder(
        latent_reformer, latent_nn, model, simple_classifier,
        train_loader, val_loader,
        epochs=epochs, lr=lr, device=device
    )

    # Evaluate on validation set
    evaluate_f1_topo_vs_reconstruction(model, latent_reformer, latent_nn, val_loader, device)
    evaluate_f1_topo_vs_reconstruction(model, latent_reformer, latent_nn, adv_loader, device)


def train_model(config):
    train(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        train_dir=config['train_dir'],
        test_dir=config['test_dir'],  
        device=config['device'],
        lr=config['lr'],
        dataset_name=config['dataset_name'], 
    )
