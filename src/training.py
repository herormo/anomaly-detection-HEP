import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna
import gc

def loss_function(x, x_recon, mu, logvar, epoch=None, recon_loss_history=None, kld_weight_final=0.01, start_kld_epoch=10, reconstruction_loss_threshold=0.1):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # KL divergence loss calculation
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # If we are in testing mode (no epoch or recon_loss_history), use fixed KLD weight
    if epoch is None or recon_loss_history is None:
        kld_weight = kld_weight_final  # Fixed KLD weight for testing
    else:
        # Sigmoid-based gradual introduction of KLD weight during training
        if epoch >= start_kld_epoch and recon_loss_history[-1] <= reconstruction_loss_threshold:
            # Convert to tensor before applying torch.exp
            kld_weight = kld_weight_final / (1 + torch.exp(-0.2 * torch.tensor(float(epoch - start_kld_epoch), device=x.device)))
        else:
            kld_weight = 0.0  # No KLD until reconstruction loss improves sufficiently

    total_loss = recon_loss + kld_weight * kld_loss
    return total_loss, recon_loss, kld_loss


def objective(trial, train_loader, val_loader, input_dim=7, num_epochs=200, device='cuda'):
    latent_dim = trial.suggest_int('latent_dim', 5, 50)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    kld_weight_final = trial.suggest_uniform('kld_weight_final', 0.001, 0.1)
    warmup_epochs = trial.suggest_int('warmup_epochs', 10, 50)

    vae = VAE(input_dim, latent_dim).to(device)
    vae.apply(weights_init)
    optimizer = AdamW(vae.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    recon_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # (code from the training loop)
        pass

    return best_val_loss

def train_model(train_loader, val_loader, input_dim=7, latent_dim=45, num_epochs=200, device='cuda'):
   # Training loop
    recon_loss_history = []

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch = batch.view(batch.size(0), -1).to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = vae(batch)
            total_loss, recon_loss, kld_loss = loss_function(batch, x_recon, mu, logvar, epoch, recon_loss_history)
            total_loss.backward()
            train_loss += total_loss.item()
            optimizer.step()

        average_train_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Train Loss: {average_train_loss:.4f}, Recon Loss: {recon_loss.item():.4f}, KLD Loss: {kld_loss.item():.4f}')

        # Append the latest reconstruction loss to the history
        recon_loss_history.append(recon_loss.item())

        # Validation step
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0]
                batch = batch.view(batch.size(0), -1).to(device)
                x_recon, mu, logvar = vae(batch)
                total_loss, recon_loss, kld_loss = loss_function(batch, x_recon, mu, logvar, epoch, recon_loss_history)
                val_loss += total_loss.item()

        average_val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch+1}, Validation Loss: {average_val_loss:.4f}, Recon Loss: {recon_loss.item():.4f}, KLD Loss: {kld_loss.item():.4f}')
        
    model_save_path = f"models/vae_v1.pth"
    torch.save(vae.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")