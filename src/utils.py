import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

# Function to print random samples from a DataLoader
def print_random_samples(data_loader, num_samples=5):
    # Retrieve one batch from the DataLoader
    data_iter = iter(data_loader)
    batch = next(data_iter)
    
    # Convert batch to a NumPy array for easier manipulation
    batch_np = batch[0].numpy() 
    
    # Randomly select a few samples from the batch
    indices = np.random.choice(range(batch_np.shape[0]), num_samples, replace=False)
    random_samples = batch_np[indices]
    
    # Iterate through the random samples and print
    for i, sample in enumerate(random_samples):
        print(f"Sample {i+1}:\n", sample, "\n")


def plot_particle_type_distribution(loader, title="Particle Type Distribution (One-Hot Encoded)"):
    """
    Plots the distribution of particle types (e.g., MET, electrons, muons, jets)
    across all events in the given DataLoader, based on one-hot encoded particle type data.
    
    Args:
    - loader: PyTorch DataLoader that contains particle data
    - title: Title of the plot
    """
    # Initialize counter for particle types (ignoring padding)
    particle_counter = {
        'MET': 0,
        'Electrons': 0,
        'Muons': 0,
        'Jets': 0
    }
    
    # Loop over batches in the DataLoader
    for data in loader:
        if isinstance(data, (list, tuple)): 
            data = data[0]  # Extract the actual data
        
        if data.ndim == 3:  # Shape (batch_size, seq_length, features)
            # Extract the one-hot encoded particle type (last 4 features, assuming 4 particle types)
            one_hot_particle_types = data[:, :, -4:]  # Adjusted for 4 particle types
        elif data.ndim == 2:  # Shape (batch_size, features)
            one_hot_particle_types = data[:, -4:]  # Adjusted for 4 particle types
        else:
            raise ValueError(f"Unsupported data dimension: {data.ndim}")
        
         
        # Sum across the batch dimension to get the total counts for each particle type
        total_counts = np.sum(one_hot_particle_types.cpu().numpy(), axis=0)
        
        # Update the particle counter (adjusted indices for 4 types)
        particle_counter['MET'] += total_counts[0]
        particle_counter['Electrons'] += total_counts[1]
        particle_counter['Muons'] += total_counts[2]
        particle_counter['Jets'] += total_counts[3]

    # Prepare data for plotting
    labels = list(particle_counter.keys())
    values = list(particle_counter.values())
    
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
    plt.title(title)
    plt.xlabel('Particle Type')
    plt.ylabel('Count')
    plt.show()



def plot_distribution_from_loader(loader, title='Data Distribution', num_features=3):
    """
    Plots the distribution of each continuous feature from the DataLoader.

    Args:
    - loader: PyTorch DataLoader containing the data.
    - title: Title of the plot.
    - num_features: Number of continuous features to plot (default: 3 for pT, η, ϕ).
    """
    # Accumulate all the data from the DataLoader
    all_data = []
    for batch in loader:
        batch_data = batch[0].numpy()  # Assuming single input (X) in each batch
        all_data.append(batch_data)
    
    all_data = np.concatenate(all_data, axis=0)  # Combine batches into one array
    
    # Print the shape of all_data to debug
    print("All data shape:", all_data.shape)
    
    # Debugging: print the first few rows of the data
    print("First few rows of data:")
    print(all_data[:5])

    # Check if the data is 2D or 3D
    if all_data.ndim == 3:
        # 3D case (batch_size, seq_length, features)
        continuous_features = all_data[:, :, :num_features]  # Take the first `num_features` features
    elif all_data.ndim == 2:
        # 2D case (batch_size, features)
        continuous_features = all_data[:, :num_features]  # Take the first `num_features` features
    else:
        raise ValueError(f"Unsupported data dimension: {all_data.ndim}")
    
    # Plot the distribution of continuous features
    plt.figure(figsize=(15, 5))
    
    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        plt.hist(continuous_features[:, i].flatten(), bins=50, alpha=0.7)
        plt.title(f'Feature {i+1}')
    
    plt.suptitle(f'{title} (Continuous Features)')
    plt.tight_layout()
    plt.show()


def visualize_reconstructions(vae, data_loader, device, num_samples=5):
    vae.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0].to(device)
            batch = batch.view(batch.size(0), -1) # Flatten the input
            x_recon, _, _ = vae(batch)
            
            # Select a few samples
            samples = batch[:num_samples]
            reconstructions = x_recon[:num_samples]
            break  # We only need one batch for visualization

    # Convert tensors to numpy arrays
    samples_np = samples.cpu().numpy()
    reconstructions_np = reconstructions.cpu().numpy()

    # Print shapes for debugging
    print(f'Sample shape: {samples_np.shape}')
    print(f'Reconstruction shape: {reconstructions_np.shape}')

    # Plot input vectors and reconstructions
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2))
    for i in range(num_samples):
        # Plot original sample
        axes[i, 0].plot(samples_np[i], marker='o')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # Plot reconstruction
        axes[i, 1].plot(reconstructions_np[i], marker='o')
        axes[i, 1].set_title('Reconstruction')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
visualize_reconstructions(vae, val_loader, device)

def plot_reconstruction_error(test_loader, vae, device):
    vae.eval()
    recon_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].view(batch[0].size(0), -1).to(device)
            x_recon, mu, logvar = vae(batch)
            recon_loss = F.mse_loss(x_recon, batch, reduction='none').mean(dim=1)
            recon_errors.extend(recon_loss.cpu().numpy())
    
    plt.figure(figsize=(8, 6))
    plt.hist(recon_errors, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Number of Events')
    plt.title('Distribution of Reconstruction Errors')
    plt.grid(True)
    plt.show()

def pca_tsne_analysis(vae, data_loader, device, subsample_ratio=0.5, threshold_percentile=95):
    """
    Performs PCA and t-SNE analysis on the latent representations and visualizes anomalies.
    """
    vae.eval()
    reconstruction_errors = []
    latent_representations = []
    originals = []
    max_batches = 100
    processed_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            if processed_batches >= max_batches:
                break
            processed_batches += 1

            batch_data = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_data = batch_data.view(batch_data.size(0), -1)
            num_samples = int(len(batch_data) * subsample_ratio)
            subsample_indices = np.random.choice(len(batch_data), num_samples, replace=False)
            batch_data = batch_data[subsample_indices]

            batch_data = batch_data.to(device)
            x_recon, mu, logvar = vae(batch_data)

            error = compute_reconstruction_error(batch_data, x_recon)
            reconstruction_errors.extend(error.cpu().numpy())
            latent_representations.extend(mu.cpu().numpy())
            originals.extend(batch_data.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)
    latent_representations = np.array(latent_representations)
    originals = np.array(originals)

    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    anomalies = reconstruction_errors > threshold

    # Apply PCA
    pca = PCA(n_components=5, random_state=42)
    latent_pca = pca.fit_transform(latent_representations)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, n_iter=500)
    latent_tsne = tsne.fit_transform(latent_pca)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=anomalies, cmap='coolwarm', marker='o')
    plt.colorbar(label='Anomaly (1 = Anomaly, 0 = Normal)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE of Latent Space with Anomalies')
    plt.show()