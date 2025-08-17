import torch
from sklearn.metrics import f1_score

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
