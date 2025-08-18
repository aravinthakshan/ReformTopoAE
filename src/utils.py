import torch
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_f1_topo_vs_reconstruction(classifier, latent_reformer, latent_nn, val_loader, device):
    classifier.eval()
    latent_reformer.eval()
    latent_nn.eval()
    
    all_labels = []
    all_preds_topo = []
    all_preds_recon = []

    with torch.no_grad():
        for clean_img, topo_img, label, latent_vec in val_loader:
            topo_img = topo_img.to(device)
            latent_vec = latent_vec.to(device)
            label = label.to(device)
            
            # -- Classify topo images --
            logits_topo = classifier(topo_img)
            preds_topo = torch.argmax(logits_topo, dim=1)  # Shape: (batch,)

            # -- Generate reconstruction from topo images + latent vector --
            latent_bottleneck = latent_nn(latent_vec)
            rec_img = latent_reformer(topo_img, latent_bottleneck)  # Shape: (batch, H, W) or (batch, 1, H, W)
            
            # If classifier expects (batch, 1, H, W) and rec_img is (batch, H, W), unsqueeze channel
            if rec_img.ndim == 3:
                rec_img = rec_img.unsqueeze(1)
            
            logits_recon = classifier(rec_img)
            preds_recon = torch.argmax(logits_recon, dim=1)

            all_labels.extend(label.cpu().numpy())
            all_preds_topo.extend(preds_topo.cpu().numpy())
            all_preds_recon.extend(preds_recon.cpu().numpy())

    f1_topo = f1_score(all_labels, all_preds_topo, average='macro')
    f1_recon = f1_score(all_labels, all_preds_recon, average='macro')

    print(f"F1 score (topo images): {f1_topo:.4f}")
    print(f"F1 score (recon images): {f1_recon:.4f}")
    return f1_topo, f1_recon

def train_latent_autoencoder(latent_reformer,latent_nn,model,train_loader, val_loader,  
    epochs, lr, device,alpha=2.0, beta=2.0, gamma=1.0):
    latent_reformer.to(device)
    latent_nn.to(device)
    model.to(device)

    optimizer = torch.optim.Adam(
        list(latent_reformer.parameters()) +
        list(latent_nn.parameters()),
        lr=lr
    )
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        latent_nn.train()
        latent_reformer.train()
        model.eval()  

        train_loss = 0.0
        for clean_img, topo_img, label, latent_vec in train_loader:
            clean_img = clean_img.to(device)
            topo_img = topo_img.to(device)
            latent_vec = latent_vec.to(device)
            label = label.to(device)

            # Forward pass
            latent_bottleneck = latent_nn(latent_vec)  
            recon_output = latent_reformer(topo_img, latent_bottleneck)

            # Reconstruction loss
            loss_recon = criterion_recon(recon_output, clean_img)
            
            with torch.no_grad():
                for param in model.parameters():
                    param.requires_grad = False
            logits_class = model(recon_output)
            loss_class = criterion_class(logits_class, label)

            loss_simple = criterion_class(logits_simple, label)

            # Total loss (as per routing logic)
            loss = alpha * loss_recon + beta * loss_class + gamma * loss_simple

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * clean_img.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        latent_reformer.eval()
        latent_nn.eval()
        simple_classifier.eval()
        val_loss = 0.0
        total_psnr, total_ssim, samples = 0.0, 0.0, 0

        with torch.no_grad():
            for clean_img, topo_img, label, latent_vec in val_loader:
                clean_img = clean_img.to(device)
                topo_img = topo_img.to(device)
                latent_vec = latent_vec.to(device)
                label = label.to(device)

                latent_bottleneck = latent_nn(latent_vec)
                recon_output = latent_reformer(topo_img, latent_bottleneck)

                # Reconstruction loss
                loss_recon = criterion_recon(recon_output, clean_img)

                
                logits_class = model(recon_output)
                loss_class = criterion_class(logits_class, label)

                # Simple classifier loss
                
                logits_simple = simple_classifier(latent_bottleneck)
                loss_simple = criterion_class(logits_simple, label)

                loss = alpha * loss_recon + beta * loss_class + gamma * loss_simple
                val_loss += loss.item() * clean_img.size(0)

                # PSNR & SSIM calculation
                output_np = recon_output.detach().cpu().numpy()
                clean_np = clean_img.detach().cpu().numpy()
                if output_np.ndim == 4 and output_np.shape[1] == 1:
                    output_np = output_np.squeeze(1)
                if clean_np.ndim == 4 and clean_np.shape == 1:
                    clean_np = clean_np.squeeze(1)
                for o, c in zip(output_np, clean_np):
                    c = c.squeeze(0)
                    psnr = peak_signal_noise_ratio(c, o, data_range=1.0)
                    ssim = structural_similarity(c, o, data_range=1.0)
                    total_psnr += psnr
                    total_ssim += ssim
                    samples += 1

        val_loss = val_loss / len(val_loader.dataset)
        avg_psnr = total_psnr / samples
        avg_ssim = total_ssim / samples
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {avg_psnr:.3f} | SSIM: {avg_ssim:.3f}")

    print("Training Complete!")
    return latent_reformer, latent_nn, simple_classifier


# evaluate_f1_topo_vs_reconstruction(model, latent_reformer, latent_nn, val_loader, device)


# adv_loader = get_advmnist_topo_loaders("/kaggle/input/invi_mnist_64/pytorch/default/1/adversarial_mnist_cw_strong_complete.npz")

# evaluate_f1_topo_vs_reconstruction(model, latent_reformer, latent_nn, adv_loader, device)

