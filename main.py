import torch
from src.data_loader import create_datasets
from src.training import train_model, objective
from src.model import VAE, weights_init
from src.utils import (
    print_random_samples,
    plot_particle_type_distribution,
    plot_distribution_from_loader,
    visualize_reconstructions,
    plot_reconstruction_error,
    pca_tsne_analysis
)
import optuna

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
background_file = 'dataset/background_for_training.h5'
signal_files = [
    'dataset/Ato4l_lepFilter_13TeV_filtered.h5',
    'dataset/hChToTauNu_13TeV_PU20_filtered.h5',
    'dataset/hToTauTau_13TeV_PU20_filtered.h5',
    'dataset/leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5'
]
blackbox_file = 'dataset/BlackBox_background_mix.h5'

train_loader, val_loader, test_loader, blackbox_loader = create_datasets(
    bkg_file=background_file,
    signals_files=signal_files,
    blackbox_file=blackbox_file,
    events=200000,  # Process only the first 200,000 events
    test_size=0.2,
    val_size=0.2,
    input_shape=7,
    batch_size=64
)

# Visualize data samples
print("Training Data Random Samples:")
print_random_samples(train_loader, num_samples=3)

# Plot particle type distribution
plot_particle_type_distribution(train_loader, title="Training Data Particle Type Distribution (Excluding Padding)")

# Plot feature distributions
plot_distribution_from_loader(train_loader, title='Training Data Distribution')

# Hyperparameter optimization with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, train_loader, val_loader, device=device), n_trials=20)

# Retrieve best hyperparameters
best_params = study.best_params
print("Best hyperparameters: ", best_params)

# Initialize VAE with best hyperparameters
vae = VAE(input_dim=7, latent_dim=best_params['latent_dim']).to(device)
vae.apply(weights_init)

# Train the model
train_model(train_loader, val_loader, vae, best_params, num_epochs=200, device=device)

# Save the trained model
model_save_path = f"models/vae_v2.pth"
torch.save(vae.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the saved model
vae.load_state_dict(torch.load(model_save_path, map_location=device))
vae.to(device)

# Visualize reconstructions on validation and training data
visualize_reconstructions(vae, val_loader, device)
visualize_reconstructions(vae, train_loader, device)

# Evaluate and plot reconstruction errors
plot_reconstruction_error(test_loader, vae, device)
plot_reconstruction_error(val_loader, vae, device)
plot_reconstruction_error(blackbox_loader, vae, device)

# Perform PCA and t-SNE analysis on blackbox data
pca_tsne_analysis(vae, blackbox_loader, device)
