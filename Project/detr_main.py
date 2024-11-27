import os
import glob
import numpy as np
import nibabel as nib
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from training import chamfer_loss, hungarian_mse_loss_single_sample  # Ensure this function is correctly implemented
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from models import create_detr  # Ensure this import is correct and accessible
import random

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_streamlines(ground_truth, predictions,  save_path='test.png'):
    """
    Plots ground truth and predicted streamlines side by side.

    Args:
        ground_truth (torch.Tensor): Ground truth streamlines [batch, num_streamlines, 32, 3].
        predictions (torch.Tensor): Predicted streamlines [batch, num_streamlines, 32, 3].
        num_streamlines (int): Number of streamlines to plot from the batch.
        save_path (str): Path to save the plot.
    """
    # Select the first instance in the batch
    gt_streamlines = ground_truth.cpu().numpy()
    pred_streamlines = predictions.cpu().numpy()

    # Create subplots
    fig = plt.figure(figsize=(16, 8))

    # Ground Truth Streamlines
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Ground Truth Streamlines')
    for i in range(gt_streamlines.shape[0]):
        ax1.plot(gt_streamlines[i, :, 0], gt_streamlines[i, :, 1], gt_streamlines[i, :, 2], label=f'Streamline {i+1}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Predicted Streamlines
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Predicted Streamlines')
    for i in range(gt_streamlines.shape[0]):
        ax2.plot(pred_streamlines[i, :, 0], pred_streamlines[i, :, 1], pred_streamlines[i, :, 2], label=f'Streamline {i+1}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory


# Step 1: Data Processing - Filter and Resample
def preprocess_and_save_to_hdf5(n_list, latent_dir, strands_dir, hdf5_path, fixed_length=32, min_streamlines=16):
    """
    Processes the data for a given list of 'n' identifiers and saves them into an HDF5 file.

    Args:
        n_list (list): List of 'n' identifiers.
        latent_dir (str): Directory containing latent .npy files.
        strands_dir (str): Directory containing streamline .tck files.
        hdf5_path (str): Path to save the HDF5 file.
        fixed_length (int): Number of points to resample each streamline to.
        min_streamlines (int): Minimum number of streamlines required.
    """
    with h5py.File(hdf5_path, 'w') as h5f:
        # Create groups for latent vectors and streamlines
        latent_grp = h5f.create_group('latents')
        streamlines_grp = h5f.create_group('streamlines')

        for idx, n in enumerate(n_list):
            latent_file = os.path.join(latent_dir, f'z_{n}.npy')
            strands_file = os.path.join(strands_dir, f'strands_{n}.tck')

            # Check if both files exist
            if not os.path.exists(latent_file):
                print(f"Latent file {latent_file} does not exist. Skipping n={n}.")
                continue
            if not os.path.exists(strands_file):
                print(f"Strands file {strands_file} does not exist. Skipping n={n}.")
                continue

            # Load latent vector
            try:
                z_n = np.load(latent_file)
            except Exception as e:
                print(f"Error loading latent file {latent_file}: {e}")
                continue

            # Load streamline data
            try:
                tractogram = nib.streamlines.load(strands_file)
                streamlines = tractogram.streamlines
            except Exception as e:
                print(f"Error loading strands file {strands_file}: {e}")
                continue

            # Check streamline count
            if len(streamlines) < min_streamlines:
                print(f"n={n} has insufficient streamlines: {len(streamlines)} < {min_streamlines}. Skipping.")
                continue

            # Resample streamlines
            processed_streamlines = []
            for streamline in streamlines:
                resampled = resample_streamline(streamline, fixed_length)
                processed_streamlines.append(resampled)

            if not processed_streamlines:
                print(f"No streamlines processed for n={n}. Skipping.")
                continue

            # Convert to numpy array
            processed_streamlines = np.stack(processed_streamlines)  # Shape: [num_streamlines, fixed_length, 3]

            # Save latent vector and streamlines
            try:
                latent_grp.create_dataset(str(n), data=z_n, compression="gzip")
                streamlines_grp.create_dataset(f'strands_{n}', data=processed_streamlines, compression="gzip")
                print(f"Processed and saved data for n={n} into {hdf5_path} ({idx+1}/{len(n_list)})")
            except Exception as e:
                print(f"Error saving data for n={n} into {hdf5_path}: {e}")

    print(f"Data preprocessing and saving to {hdf5_path} completed.")


def resample_streamline(streamline, num_points=32):
    """
    Resamples a single streamline to a fixed number of points.

    Args:
        streamline (array): Original streamline coordinates [N, 3].
        num_points (int): Number of points to resample to.

    Returns:
        resampled_sl (array): Resampled streamline coordinates [num_points, 3].
    """
    original_length = streamline.shape[0]
    if original_length < 2:
        # If streamline has less than 2 points, repeat the single point
        resampled_sl = np.repeat(streamline, num_points, axis=0)
    else:
        # Create interpolation indices
        new_indices = np.linspace(0, original_length - 1, num_points)
        resampled_sl = np.vstack([
            np.interp(new_indices, np.arange(original_length), streamline[:, dim])
            for dim in range(3)
        ]).T

    return resampled_sl  # Shape: [num_points, 3]


# Step 2: Get Valid 'n' Identifiers
def get_valid_n(latent_dir, strands_dir, min_streamlines=32):
    """
    Retrieves a list of valid 'n' identifiers where corresponding latent and strand files exist
    and meet the minimum streamline length requirement.

    Args:
        latent_dir (str): Directory containing latent .npy files.
        strands_dir (str): Directory containing streamline .tck files.
        min_streamlines (int): Minimum number of streamlines required.

    Returns:
        list: List of valid 'n' identifiers.
    """
    latent_files = glob.glob(os.path.join(latent_dir, 'z_*.npy'))
    valid_n = []

    for latent_file in latent_files:
        try:
            n = int(os.path.basename(latent_file).split('_')[1].split('.')[0])
        except (IndexError, ValueError) as e:
            print(f"Error parsing 'n' from filename {latent_file}: {e}")
            continue

        strands_file = os.path.join(strands_dir, f'strands_{n}.tck')

        if os.path.exists(strands_file):
            # Load the streamline data
            try:
                tractogram = nib.streamlines.load(strands_file)
                streamlines = tractogram.streamlines
                if len(streamlines) >= min_streamlines:
                    valid_n.append(n)
                else:
                    print(f"n={n} has insufficient streamlines: {len(streamlines)} < {min_streamlines}")
            except Exception as e:
                print(f"Error loading {strands_file}: {e}")
        else:
            print(f"Strands file not found for n={n}")

    print(f"Total valid 'n' identifiers: {len(valid_n)}")
    return valid_n


# Step 3: Split the Data
def split_data(valid_n, val_size=0.1, test_size=0.1, random_state=42):
    """
    Splits the valid 'n' identifiers into training, validation, and test sets.

    Args:
        valid_n (list): List of valid 'n' identifiers.
        val_size (float): Proportion of data to use for validation.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (train_n, val_n, test_n)
    """
    train_val_n, test_n = train_test_split(valid_n, test_size=test_size, random_state=random_state)
    val_relative_size = val_size / (1 - test_size)  # Adjust validation size relative to train_val
    train_n, val_n = train_test_split(train_val_n, test_size=val_relative_size, random_state=random_state)
    print(f"Train size: {len(train_n)}, Val size: {len(val_n)}, Test size: {len(test_n)}")
    return train_n, val_n, test_n


# Step 4: Create Custom Dataset Class
class HDF5StreamlineDataset(Dataset):
    """
    Custom Dataset for loading streamlines and latent vectors from an HDF5 file.
    """

    def __init__(self, n_list, hdf5_path, num_streamlines=32, fixed_length=32, preload_latents=False):
        """
        Initializes the dataset.

        Args:
            n_list (list): List of 'n' identifiers.
            hdf5_path (str): Path to the HDF5 file.
            num_streamlines (int): Number of streamlines per sample.
            fixed_length (int): Number of points per streamline.
            preload_latents (bool): If True, preload latent vectors into memory.
        """
        self.n_list = n_list
        self.hdf5_path = hdf5_path
        self.num_streamlines = num_streamlines
        self.fixed_length = fixed_length
        self.preload_latents = preload_latents

        # Open HDF5 file
        self.h5f = h5py.File(self.hdf5_path, 'r')

        if self.preload_latents:
            # Preload latent vectors into memory
            self.latent_vectors = {}
            for n in self.n_list:
                try:
                    z_n = self.h5f['latents'][str(n)][()]
                    self.latent_vectors[n] = torch.from_numpy(z_n).float()
                except KeyError:
                    print(f"Latent vector for n={n} not found in {self.hdf5_path}")
        else:
            self.latent_vectors = None

    def __len__(self):
        return len(self.n_list)

    def __getitem__(self, idx):
        """
        Retrieves the 'n' index, latent vector, and a randomly sampled set of streamlines for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (n_tensor, z_n_tensor, streamline_tensor)
        """
        # Map index to n
        n = self.n_list[idx]

        # Load latent vector z_n
        if self.preload_latents:
            z_n_tensor = self.latent_vectors.get(n, torch.zeros(1))
        else:
            try:
                z_n = self.h5f['latents'][str(n)][()]
                z_n_tensor = torch.from_numpy(z_n).float()
            except KeyError:
                print(f"Latent vector for n={n} not found in {self.hdf5_path}")
                z_n_tensor = torch.zeros(1)  # Default to zero tensor

        # Load all streamlines for this n
        try:
            streamlines = self.h5f['streamlines'][f'strands_{n}'][()]  # Shape: [num_streamlines, fixed_length, 3]
        except KeyError:
            print(f"Streamline data for n={n} not found in {self.hdf5_path}")
            streamlines = np.zeros((self.num_streamlines, self.fixed_length, 3), dtype=np.float32)

        num_available_streamlines = streamlines.shape[0]

        # Randomly sample num_streamlines streamlines without replacement
        if num_available_streamlines >= self.num_streamlines:
            sampled_indices = np.random.choice(num_available_streamlines, self.num_streamlines, replace=False)
        else:
            # If not enough streamlines, sample with replacement
            sampled_indices = np.random.choice(num_available_streamlines, self.num_streamlines, replace=True)

        sampled_streamlines = streamlines[sampled_indices]  # Shape: [num_streamlines, fixed_length, 3]

        # Convert to tensor
        streamline_tensor = torch.from_numpy(sampled_streamlines).float()  # Shape: [num_streamlines, fixed_length, 3]

        # Convert 'n' to tensor
        n_tensor = torch.tensor(n, dtype=torch.int64)

        return n_tensor, z_n_tensor, streamline_tensor

    def close(self):
        """
        Closes the HDF5 file.
        """
        if self.h5f:
            self.h5f.close()


# Step 5: Create DataLoaders with Separate HDF5 Files
def get_dataloader(n_list, hdf5_path, batch_size, shuffle=True, num_workers=0, preload_latents=False):
    """
    Creates a DataLoader for a given dataset split.

    Args:
        n_list (list): List of 'n' identifiers.
        hdf5_path (str): Path to the HDF5 file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.
        preload_latents (bool): If True, preload latent vectors into memory.

    Returns:
        DataLoader: Configured DataLoader.
    """
    dataset = HDF5StreamlineDataset(n_list, hdf5_path, preload_latents=preload_latents)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True  # Recommended for CUDA
    )
    return dataloader


def worker_init_fn(worker_id):
    """
    Initializes each worker by opening its own HDF5 file handle.

    Args:
        worker_id (int): Worker identifier.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Each worker has its own file handle
    dataset.h5f = h5py.File(dataset.hdf5_path, 'r')



# Step 7: Training Loop with Validation and Test Evaluation
def train_model(latent_dir, strands_dir, hdf5_train_path, hdf5_val_path, hdf5_test_path,
               batch_size=512, num_epochs=10, learning_rate=5e-5, device='cuda'):
    """
    Orchestrates the training process, including data preparation, model training,
    validation, and testing.

    Args:
        latent_dir (str): Directory containing latent .npy files.
        strands_dir (str): Directory containing streamline .tck files.
        hdf5_train_path (str): Path to save/load the training HDF5 file.
        hdf5_val_path (str): Path to save/load the validation HDF5 file.
        hdf5_test_path (str): Path to save/load the test HDF5 file.
        batch_size (int): Number of samples per batch.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str or torch.device): Device to train on ('cuda' or 'cpu').
    """
    # Step 1: Get valid 'n' integers

    set_seed(42)

    valid_n = get_valid_n(latent_dir, strands_dir, min_streamlines=32)
    if not valid_n:
        print("No valid data found. Exiting training.")
        return
    print(f"Total valid samples: {len(valid_n)}")

    # Step 2: Split data into train, val, test
    train_n, val_n, test_n = split_data(valid_n, val_size=0.1, test_size=0.1)
    print(f"Training samples: {len(train_n)}, Validation samples: {len(val_n)}, Test samples: {len(test_n)}")

    # Step 3: Preprocess and save data into separate HDF5 files
    # Check if HDF5 files already exist to avoid reprocessing
    if not os.path.exists(hdf5_train_path):
        print("Preprocessing and saving training data...")
        preprocess_and_save_to_hdf5(train_n, latent_dir, strands_dir, hdf5_train_path)
    else:
        print(f"Training HDF5 file {hdf5_train_path} already exists. Skipping preprocessing.")

    if not os.path.exists(hdf5_val_path):
        print("Preprocessing and saving validation data...")
        preprocess_and_save_to_hdf5(val_n, latent_dir, strands_dir, hdf5_val_path)
    else:
        print(f"Validation HDF5 file {hdf5_val_path} already exists. Skipping preprocessing.")

    if not os.path.exists(hdf5_test_path):
        print("Preprocessing and saving test data...")
        preprocess_and_save_to_hdf5(test_n, latent_dir, strands_dir, hdf5_test_path)
    else:
        print(f"Test HDF5 file {hdf5_test_path} already exists. Skipping preprocessing.")

    # Step 4: Prepare DataLoaders
    print('Creating DataLoaders...')
    train_loader = get_dataloader(train_n, hdf5_train_path, batch_size, shuffle=True, num_workers=4, preload_latents=True)
    val_loader = get_dataloader(val_n, hdf5_val_path, batch_size, shuffle=False, num_workers=4, preload_latents=True)
    test_loader = get_dataloader(test_n, hdf5_test_path, batch_size, shuffle=False, num_workers=4, preload_latents=True)

    # Optional: Create a mapping from 'n' to file paths for easy reference during testing
    n_to_file_path = {n: os.path.join(latent_dir, f'z_{n}.npy') for n in valid_n}

    # Step 5: Initialize the model
    print("Initializing the model...")
    model = create_detr(num_streamlines=32, device=device)
    model.to(device)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f'Number of batches in training loader: {len(train_loader)}')

    val_interval = 5  # Validate every 'val_interval' epochs

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (n_batch, z_n, streamlines) in enumerate(train_loader):
            # Move data to device (CPU/GPU)
            z_n = z_n.to(device).squeeze(2).squeeze(2).squeeze(2)
            streamlines = streamlines.to(device) / 0.2  # Normalize as per original code
            optimizer.zero_grad()
            random_idx = torch.randint(0, streamlines.shape[0], (1,)).item()     
            with autocast():
                # Forward pass through the model
                outputs = model(z_n)
                pred_sample = outputs[random_idx]  # [m, D, C]
                target_sample = streamlines[random_idx]   # [m, D, C]
                # Compute loss between outputs and ground truth streamlines
                loss = chamfer_loss(outputs, streamlines)  # Assuming chamfer_loss is appropriate
                hungarian_mse = hungarian_mse_loss_single_sample(pred_sample, target_sample)
                loss += hungarian_mse
            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_loss:.4f}')

        # Validation Step
        if ((epoch + 1) % val_interval) == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch_idx, (n_val, z_n_val, streamlines_val) in enumerate(val_loader):
                    z_n_val = z_n_val.to(device).squeeze(2).squeeze(2).squeeze(2)
                    streamlines_val = streamlines_val.to(device) / 0.2

                    with autocast():
                        val_outputs = model(z_n_val)
                        loss_val = chamfer_loss(val_outputs, streamlines_val)

                    val_loss += loss_val.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

    # Testing after training
    print("Evaluating on Test Set...")
    model.eval()
    test_loss = 0
    plot_count = 0
    with torch.no_grad():
        for test_batch_idx, (n_test, z_n_test, streamlines_test) in enumerate(test_loader):
            z_n_test = z_n_test.to(device).squeeze(2).squeeze(2).squeeze(2)
            streamlines_test = streamlines_test.to(device) / 0.2

            with autocast():
                test_outputs = model(z_n_test)
                loss_test = chamfer_loss(test_outputs, streamlines_test)

            test_loss += loss_test.item()

            if plot_count < 10:
                # For plotting, select the first sample in the batch
                streamlines_sample = streamlines_test[0].cpu()
                test_outputs_sample = test_outputs[0].cpu()
                n_val = n_test[0].item()
                print(f"Plotting streamlines for n={n_val}")
                plot_streamlines(streamlines_sample, test_outputs_sample, save_path=f'test_plot_n_{n_val}.png')
                plot_count += 1

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')

    # Close datasets
    train_loader.dataset.close()
    val_loader.dataset.close()
    test_loader.dataset.close()

    print("Training and evaluation completed successfully.")


# Entry Point
if __name__ == '__main__':
    # Set directories
    latent_dir = '/cluster/project2/CU-MONDAI/Alec_Tract/Project/Data/latents'
    strands_dir = '/cluster/project2/CU-MONDAI/Alec_Tract/Project/Data/Tracts'

    # Define HDF5 paths for each split
    hdf5_train_path = '/cluster/project2/CU-MONDAI/Alec_Tract/Project/Data/TractsH5/train.h5'
    hdf5_val_path = '/cluster/project2/CU-MONDAI/Alec_Tract/Project/Data/TractsH5/val.h5'
    hdf5_test_path = '/cluster/project2/CU-MONDAI/Alec_Tract/Project/Data/TractsH5/test.h5'

    # Ensure the output directory exists
    os.makedirs('/cluster/project2/CU-MONDAI/Alec_Tract/Project/Data/TractsH5', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train the model
    train_model(
        latent_dir=latent_dir,
        strands_dir=strands_dir,
        hdf5_train_path=hdf5_train_path,
        hdf5_val_path=hdf5_val_path,
        hdf5_test_path=hdf5_test_path,
        batch_size=256,
        num_epochs=5000,  # Adjust as needed
        learning_rate=5e-5,
        device=device
    )
