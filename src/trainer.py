import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

<<<<<<< HEAD
def train_latent_autoencoder(latent_reformer,latent_nn,model,simple_classifier,train_loader, val_loader,  
    epochs, lr, device,alpha=0.5, beta=2.0, gamma=1.0):
    latent_reformer.to(device)
    latent_nn.to(device)
    model.to(device)
    simple_classifier.to(device)

    optimizer = torch.optim.Adam(
        list(latent_reformer.parameters()) +
        list(latent_nn.parameters()) +
        list(simple_classifier.parameters()), lr=lr
=======
def train_model(config):
    train(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        train_dir=config['train_dir'],
        test_dir=config['test_dir'],  
        wandb_debug=config['wandb'], 
        device=config['device'],
        lr=config['lr'],
        dataset_name=config['dataset_name'], 
        noise_level = config['noise_level']
>>>>>>> b7cd914 (copied older repo trainer and train code)
    )

latent_reformer,latent_nn,simple_classifier = train_latent_autoencoder(latent_reformer, latent_nn, model,simple_classifier, train_loader, val_loader, epochs=50, lr=1e-3, device=device)