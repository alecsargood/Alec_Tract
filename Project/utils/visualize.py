import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
import os

def visualize_reconstruction(autoencoder, test_ds, device, channels=(0,), save_dir='./results/dMRI'):
    """
    Visualizes and saves original and reconstructed slices for a single image from the test dataset.

    Parameters:
    - autoencoder (nn.Module): Trained autoencoder model.
    - test_ds (Dataset): Test dataset.
    - device (torch.device): Device to perform computations on.
    - channels (tuple): Tuple of channel indices to visualize.
    - save_dir (str): Directory where the visualizations will be saved.

    Returns:
    - None
    """

    sample = test_ds[100]
    image = sample["image"].unsqueeze(0).to(device)

    # Perform reconstruction
    autoencoder.eval()
    with torch.no_grad():
        reconstruction, _, _ = autoencoder(image)

    image_np = image.cpu().squeeze(0).numpy()          # Shape: (C, H, W, D)
    recon_np = reconstruction.cpu().squeeze(0).numpy()  # Shape: (C, H, W, D)    

    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    for channel in channels:
        file_name = f'Reconstructions_ch_{channel}.png'

        # Select the specified channel
        original = image_np[channel]          # Shape: (H, W, D)
        reconstructed = recon_np[channel]     # Shape: (H, W, D)

        # Determine slice indices
        depth = original.shape[-1]
        slice_indices = [int(np.round(depth/4)), int(np.round(depth/2)), int(np.round(depth*3/4))]
    
        # Plotting
        fig, axes = plt.subplots(3, 2, figsize=(8, 4 * 3))

        for i, slice_idx in enumerate(slice_indices):
            # Original slice
            axes[i, 0].imshow(original[:, :, slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Original Slice {slice_idx}')
            axes[i, 0].axis('off')

            # Reconstructed slice
            axes[i, 1].imshow(reconstructed[:, :, slice_idx], cmap='gray')
            axes[i, 1].set_title(f'Reconstructed Slice {slice_idx}')
            axes[i, 1].axis('off')

        plt.tight_layout()

        # Define the full save path
        save_path = os.path.join(save_dir, file_name)

        # Save the figure
        plt.savefig(save_path)
        print(f"Reconstruction comparison saved to {save_path}")

        # Close the plot to free memory
        plt.close(fig)




def visualize_tract_results(test_hdf5_path, latents_dir, model, scheduler, inferer, device, num_generate=16, save_dir='results/Tracts'):
    """
    Visualize generated streamlines compared to ground truth for a specific MRI latent.
    
    Args:
        test_hdf5_path (str): Path to the test HDF5 file.
        latents_dir (str): Directory where latent vectors are stored.
        model (nn.Module): Trained diffusion model.
        scheduler: Diffusion scheduler.
        inferer: Diffusion inferer.
        device (torch.device): Device to perform computations on.
        num_generate (int): Number of streamlines to generate.
        save_dir (str): Directory to save the visualization.
    
    Saves:
        A figure comparing ground truth and generated streamlines.
    """
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    
    model.to(device)
    model.eval()
    #print(f"Loaded model weights from {best_model_path} and set to evaluation mode.")
    
    # Open the test HDF5 file
    with h5py.File(test_hdf5_path, 'r') as h5f:
        streamlines = h5f['streamlines'][:]  # Shape: (num_samples, 3, num_points)
        mri_paths = h5f['mri_paths'][:]      # Shape: (num_samples,)
        
        print(len(mri_paths))
        # Decode bytes to strings if necessary
        if isinstance(mri_paths[0], bytes):
            mri_paths = [path.decode('utf-8') for path in mri_paths]
    
    # Get all unique 'n's
    unique_n = list(set(mri_paths))
    print(f"Found {len(unique_n)} unique MRI identifiers in the test set.")
    
    if not unique_n:
        raise ValueError("No MRI identifiers found in the test HDF5 file.")
    
    # Select an 'n' to visualize
    selected_n = unique_n[0]  # Change as needed
    print(f"Selected MRI identifier: {selected_n}")
    
    # Find all indices with the selected 'n'
    indices = [i for i, n in enumerate(mri_paths) if n == selected_n]
    print(f"Number of streamlines for n={selected_n}: {len(indices)}")
    
    if len(indices) == 0:
        raise ValueError(f"No streamlines found for n={selected_n}.")
    
    # Extract the streamlines
    selected_streamlines = streamlines[indices]  # Shape: (num_streamlines, 3, num_points)
    
    # Load the corresponding latent
    latent_path = os.path.join(latents_dir, f'z_{selected_n}.npy')
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent file for n={selected_n} not found at {latent_path}")
    
    latent = np.load(latent_path)
    latent = torch.tensor(latent, dtype=torch.float32).squeeze(1).squeeze(1).squeeze(1).unsqueeze(0).unsqueeze(0)
    print(f'Latent shape: {latent.shape}')
    latent = latent.to(device)
    
    num_streamlines = selected_streamlines.shape[0]
    num_generate = num_streamlines  # To generate the same number of streamlines as ground truth
    
    print(f"Visualizing results for dMRI identifier: {selected_n}")
    print(f"Number of ground truth streamlines: {num_streamlines}")
    print(f"Number of streamlines to generate: {num_generate}")
    
    # Duplicate the latent as needed
    latent = latent.repeat(num_generate, 1, 1)  # Shape: [num_generate, 1, latent_dim]
    
    # Set the scheduler's timesteps for full sampling
    scheduler.set_timesteps(num_inference_steps=1000)
    
    # Generate streamlines
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn((num_generate, 3, selected_streamlines.shape[-1])).to(device)  # Shape: [num_generate, 3, num_points]
        
        # Perform sampling
        generated_streamlines = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            conditioning=latent,           # Shape: [num_generate, 1, latent_dim]
            mode='crossattn'                # Assuming 'crossattn' mode handles conditioning
        )
        
        # Move to CPU and convert to numpy
        generated_streamlines = generated_streamlines.cpu().numpy()  # Shape: [num_generate, 3, num_points]
    
    # Ground truth streamlines are already in numpy
    # selected_streamlines = selected_streamlines  # Shape: [num_streamlines, 3, num_points]
    
    # Plotting
    fig = plt.figure(figsize=(18, 8))
    
    # Ground Truth Streamlines
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Ground Truth Streamlines')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    for streamline in selected_streamlines[:num_streamlines]:
        x, y, z = streamline[0], streamline[1], streamline[2]
        ax1.plot(x, y, z, color='blue', alpha=0.6)
    ax1.view_init(elev=20., azim=120)  # Adjust viewing angle as needed
    
    # Generated Streamlines
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Generated Streamlines')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    for streamline in generated_streamlines[:num_generate]:
        x, y, z = streamline[0], streamline[1], streamline[2]
        ax2.plot(x, y, z, color='red', alpha=0.6)
    ax2.view_init(elev=20., azim=120)  # Adjust viewing angle as needed
    
    plt.suptitle(f'Comparison of Ground Truth and Generated Streamlines\nOriginal dMRI Identifier: {selected_n}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
    output_path = os.path.join(save_dir, f'comparison_streamlines_{selected_n}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {output_path}")
