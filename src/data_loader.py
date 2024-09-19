import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gc
from sklearn.model_selection import train_test_split

def normalize_data(data, min_vals=None, max_vals=None, num_classes=5):
    features = data[:, :, :-1]  # Extract features, exclude last column (particle type)
    particle_type_index = data[:, :, -1].astype(int)  # Last column as particle type index
    
    # Mask to identify valid rows (where particle_type_index != 0)
    valid_mask = (particle_type_index != 0)
    
    # Only consider non-zero rows for min/max calculation
    valid_features = features[valid_mask]
    
    # Ensure min_vals and max_vals are computed based on valid (non-zero) features
    if min_vals is None or max_vals is None:
        min_vals = valid_features.min(axis=0)
        max_vals = valid_features.max(axis=0)
    
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    
    # Normalize the non-zero features
    normalized_features = np.zeros_like(features)
    normalized_features[valid_mask] = (features[valid_mask] - min_vals) / range_vals
    
    # Initialize with zeros for 4 one-hot encoded columns (for 4 particle types excluding padding)
    one_hot_particle_types = np.zeros((data.shape[0], data.shape[1], 4))  # Array for 4 one-hot columns

    # Generate one-hot encoding for 4 particle types (exclude padding, particle_type_index == 0)
    one_hot_encoded = np.eye(num_classes - 1)[particle_type_index[valid_mask] - 1]  # 4 one-hot columns

    # Assign the one-hot encoded particle types to the correct locations
    one_hot_particle_types[valid_mask] = one_hot_encoded  # Push the one-hot encoded vectors into the array

    # Combine normalized features with one-hot encoded particle types (3 continuous + 4 one-hot encoded = 7 columns)
    normalized_data = np.concatenate([normalized_features, one_hot_particle_types], axis=-1)
    
    # Now filter out the padded rows completely from the data
    non_padded_rows_mask = np.any(normalized_features != 0, axis=2)
    filtered_data = normalized_data[non_padded_rows_mask]
    
    return filtered_data, min_vals, max_vals


def create_datasets(bkg_file, signals_files, blackbox_file, events=None, test_size=0.2, val_size=0.2, input_shape=7, batch_size=64):
    # Step 1: Load and process BACKGROUND data (SM data)
    with h5py.File(bkg_file, 'r') as file:
        full_data = file['Particles'][:, :, :input_shape]
        np.random.shuffle(full_data)  # Shuffle the data
        if events:
            full_data = full_data[:events, :, :]  # Limit the number of events
    
    # Normalize background data
    full_data, bkg_min_vals, bkg_max_vals = normalize_data(full_data)
    
    # Step 2: Split the data into training, validation, and testing sets
    X_train, X_temp = train_test_split(full_data, test_size=test_size, shuffle=True)
    X_val, X_test = train_test_split(X_temp, test_size=val_size)
    
    # Clean up memory
    del full_data, X_temp
    gc.collect()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Clean up memory
    del X_train, X_val, X_train_tensor, X_val_tensor
    gc.collect()

    # Step 3: Load and process SIGNAL data (BSM data)
    bsm_data_list = []
    for signal_file in signals_files:
        with h5py.File(signal_file, 'r') as f:
            signal_data = f['Particles'][:, :, :input_shape]
            # Normalize signal data using background normalization parameters
            signal_data, _, _ = normalize_data(signal_data, min_vals=bkg_min_vals, max_vals=bkg_max_vals)
            bsm_data_list.append(signal_data)
    
    # Concatenate all BSM data
    bsm_data = np.concatenate(bsm_data_list, axis=0)
    
    # Convert BSM data to a PyTorch tensor
    bsm_tensor = torch.tensor(bsm_data, dtype=torch.float32)
    
    # Create TensorDataset and DataLoader for test data
    test_dataset = TensorDataset(bsm_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Clean up memory
    del bsm_data_list, bsm_data, bsm_tensor
    gc.collect()
    
    # Step 4: Load and process BLACKBOX data
    with h5py.File(blackbox_file, 'r') as f:
        blackbox_data = f['Particles'][:, :, :input_shape]
        # Normalize blackbox data using background normalization parameters
        blackbox_data, _, _ = normalize_data(blackbox_data, min_vals=bkg_min_vals, max_vals=bkg_max_vals)
    
    # Convert blackbox data to a PyTorch tensor
    blackbox_tensor = torch.tensor(blackbox_data, dtype=torch.float32)
    
    # Create TensorDataset and DataLoader for blackbox data
    blackbox_dataset = TensorDataset(blackbox_tensor)
    blackbox_loader = DataLoader(blackbox_dataset, batch_size=batch_size, shuffle=False)
    
    # Clean up memory
    del blackbox_data, blackbox_tensor
    gc.collect()
    
    return train_loader, val_loader, test_loader, blackbox_loader

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
    events=200000,  # Process only the first 100,000 events
    test_size=0.2,
    val_size=0.2,
    input_shape=7,
    batch_size=64
)