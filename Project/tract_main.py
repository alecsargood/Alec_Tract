# %%
# Import necessary libraries
import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_processing.prepare_combined import prepare_data_loaders, preprocess_and_store_hdf5
from data_processing.prepare_fodf_data import calculate_num_sh_coefficients
from models.architectures import create_diffusion, create_autoencoder
from training.train_diff import train_model
from utils import visualize_tract_results, setup_torch

from comet_ml import Experiment

def main():
    # Define directories (replace with your actual paths)
    base_dir = '/cluster/project2/CU-MONDAI/Alec_Tract/Project'
    data_dir = os.path.join(base_dir, 'Data')
    latents_dir = os.path.join(data_dir, 'latents')
    hdf5_dir = os.path.join(base_dir, 'HDF5_Data')
    FOD_dir = os.path.join(data_dir, 'FODF_2')
    # Setup device
    device = setup_torch()
    print(f'Device setup complete. Using device: {device}')

    # Create and load the pretrained VAE model
    sh_order = 2
    num_sh = calculate_num_sh_coefficients(sh_order)  # Ensure this function is imported correctly
    vae = create_autoencoder(num_sh, device, latent_channels=256, norm_num_groups=8)
    vae_path = os.path.join(base_dir, 'results/dMRI/128/best_autoencoder_kl.pth')

    

    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("Pretrained VAE loaded and set to eval mode.")

    # === Process FODF files and save latents ===
    print("Processing FODF files to extract latents...")

    # Get list of FODF files
    fodf_pattern = os.path.join(FOD_dir, 'FOD_*.npy')
    fodf_files = glob.glob(fodf_pattern)
    fodf_files.sort()  # Ensure consistent order

    if not fodf_files:
        raise FileNotFoundError(f"No FODF files found in {FOD_dir} with pattern 'FOD_*.npy'")


    # Adjust batch size based on your GPU memory
    #batch_size = 512

    #num_files = len(fodf_files)
    #num_batches = (num_files + batch_size - 1) // batch_size
    #count = 0
    #for batch_idx in range(num_batches):
    #    batch_files = fodf_files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    #    batch_data = []
    #    n_values = []

        #for file_path in batch_files:
        #    # Extract 'n' from the filename 'FOD_n.npy'
        #    base_name = os.path.basename(file_path)
        #    n_part = base_name.replace('FOD_', '').replace('.npy', '')
        #    n = int(n_part)
        #    n_values.append(n)

            # Load FODF data
            #fodf = np.load(file_path)
            # Normalize or preprocess the FODF data if required
            #batch_data.append(fodf)

        # Stack the batch data into a NumPy array
        #batch_array = np.stack(batch_data)  # Shape: (batch_size, ...)

        # Convert to torch.Tensor and move to device
        #batch_tensor = torch.from_numpy(batch_array).float().to(device)

        # Adjust tensor shape if necessary
        # For example, if VAE expects input shape (batch_size, num_sh, H, W, D)
        #print(f'FODF SHAPE: {batch_tensor.shape}')
        #if batch_tensor.ndim == 5:
            # Assuming input shape is (batch_size, H, W, D, num_sh)
        #    batch_tensor = batch_tensor.permute(0, 4, 1, 2, 3)  # Now shape: (batch_size, num_sh, H, W, D)
        #else:
        #    raise ValueError(f"Unexpected input shape: {batch_tensor.shape}")

        # Process through the VAE encoder to get latents
        #print('processing latents')

        #with torch.no_grad():
            # Assuming your VAE's encode method returns (mu, logvar)
        #    mu, _ = vae.encode(batch_tensor)
        #    latents = mu.cpu().numpy()  # Convert to NumPy array for saving

        #print('latents computed, saving files')
        # Save latents
        #for i, n in enumerate(n_values):
        #    latent_vector = latents[i]
        #    latent_file_path = os.path.join(latents_dir, f'z_{n}.npy')
        #    np.save(latent_file_path, latent_vector)

        #count += batch_tensor.shape[0]

        #print(f'processed: {count} / {total_num} files')
    #print("Latents extracted and saved successfully.")

    

    # Preprocess data and store into HDF5
    print("Starting data preprocessing and HDF5 storage...")
    #preprocess_and_store_hdf5(
    #    base_dir=data_dir,
    #    latents_dir=latents_dir,
    #    num_points=32,
    #    hdf5_dir=hdf5_dir
    #)
    print("Data preprocessing and HDF5 storage completed.")

    # Prepare DataLoaders
    batch_size = 4096
    num_workers = 4
    cache_size = 500000  # Adjust based on available RAM

    print("Preparing DataLoaders...")
    train_loader, val_loader, _ = prepare_data_loaders(
        hdf5_dir=hdf5_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_size=cache_size
    )
    print('DataLoaders prepared and ready.')

    # === Inspect a batch from the training DataLoader ===
    print('\n--- Inspecting a sample batch from the training DataLoader ---')
    try:
        sample_batch = next(iter(train_loader))
    except StopIteration:
        raise ValueError("Training DataLoader is empty. Please check your HDF5 files and preprocessing steps.")

    # Print the keys in the batch dictionary
    print(f"Batch keys: {sample_batch.keys()}")  # Should be dict_keys(['streamline', 'latent', 'mri_path'])

    # Extract elements from the batch
    streamlines = sample_batch['streamline']  # Tensor of shape (batch_size, 3, num_points)
    latents = sample_batch['latent']           # Tensor of shape (batch_size, latent_dim)
    mri_paths = sample_batch['mri_path']       # List of MRI file paths

    # Print the shapes of the tensors
    print(f"Streamlines shape: {streamlines.shape}")  # Expected: (batch_size, 3, num_points)
    print(f"Latents shape: {latents.shape}")          # Expected: (batch_size, latent_dim)
    print(f"Number of MRI paths: {len(mri_paths)}")   # Expected: batch_size

    # Check batch size consistency
    actual_batch_size = streamlines.shape[0]
    print(f"Batch size (number of streamlines): {actual_batch_size}")
    assert latents.shape[0] == actual_batch_size, "Mismatch in batch size between streamlines and latents"

    # Ensure that the number of streamlines and points per streamline are as expected
    num_points = streamlines.shape[2]
    assert streamlines.shape[1] == 3, "Each streamline should have 3 dimensions (x, y, z)"
    assert num_points == 32, f"Each streamline should have exactly 32 points, got {num_points}"

    # Print data types
    print(f"Streamlines dtype: {streamlines.dtype}")  # Expected: torch.float32
    print(f"Latents dtype: {latents.dtype}")          # Expected: torch.float32

    # Optionally, move data to the device and check again
    streamlines = streamlines.to(device)
    latents = latents.to(device)
    print(f"Streamlines device: {streamlines.device}")
    print(f"Latents device: {latents.device}")

    # Now, obtain the latent size for the diffusion model
    latent_size = latents.shape[1]  # Assuming latents have shape (batch_size, latent_dim)
    print(f"Latent vector size: {latent_size}")

    # === Create Diffusion Model ===
    print("\nCreating diffusion model...")


    model, scheduler, inferer = create_diffusion(device, cross_attention_dim=latent_size*2)
    print('Diffusion model created successfully.')

    # === Training Loop Parameters ===
    n_epochs = 2000
    val_interval = 100  # Validate every 5 epochs


    experiment = Experiment(
    api_key='eQ6pfPreHQzFB4frzYlCeLiEr',
    project_name='tract',
    workspace='gentract',
    auto_output_logging="simple",  # Logs stdout and stderr
    auto_metric_logging=True,      # Automatically log metrics
    )

    # === Start Training ===
    print("\nStarting training...")
    train_model(
        model=model,
        inferer=inferer,
        train_loader=train_loader,
        val_loader=val_loader,
        diffusion_scheduler=scheduler,
        n_epochs=n_epochs,
        val_interval=val_interval,
        device=device,
        experiment=experiment
    )
    print('Training completed.')

    # === Visualize Results ===
    print("\nGenerating streamlines for visualization...")

    model.load_state_dict(torch.load(f'/cluster/project2/CU-MONDAI/Alec_Tract/Project/results/Tract/test_best_model_2.pth', map_location=device))
    model.eval()

    test_hdf5_path = os.path.join(hdf5_dir, 'test.hdf5')
    save_dir = os.path.join(base_dir, 'results/Tracts')
    visualize_tract_results(
        test_hdf5_path=test_hdf5_path,
        latents_dir=latents_dir,
        model=model,
        scheduler=scheduler,
        inferer=inferer,
        device=device,
        num_generate=16,  # Adjust as needed
        save_dir=save_dir
    )
    print('Visualization complete.')


    experiment.end()
# -----------------------------
# Execute the Main Function
# -----------------------------

if __name__ == "__main__":
    main()
