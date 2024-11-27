# data_processing/data_processing.py

import os
import glob
import random
import numpy as np
import nibabel as nib
import torch
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import OrderedDict
import matplotlib.pyplot as plt


class ToTensor:
    def __call__(self, sample):
        streamline = sample['streamline']  # shape: (3, num_points)
        # Debugging print
        # If streamline is already a tensor, no need to convert
        if not isinstance(streamline, torch.Tensor):
            streamline = torch.tensor(streamline, dtype=torch.float32)
        sample['streamline'] = streamline
        return sample


def resample_streamline(streamline, num_points):
    streamline = streamline / 0.2
    original_length = streamline.shape[0]
    if original_length == 1:
        resampled_sl = np.repeat(streamline, num_points, axis=0)
    else:
        new_indices = np.linspace(0, original_length - 1, num_points)
        resampled_sl = np.vstack([
            np.interp(new_indices, np.arange(original_length), streamline[:, dim])
            for dim in range(3)
        ]).T
    return resampled_sl.T  # Shape: (3, num_points)


def preprocess_and_store_hdf5(
    base_dir, latents_dir, num_points=32, hdf5_dir='hdf5_data', min_streamlines=16, seed=42
):
    os.makedirs(hdf5_dir, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tracts_dir = os.path.join(base_dir, 'Tracts')
    fod_dir = os.path.join(base_dir, 'FODF_2')

    print(f'Tract directory: {tracts_dir}')
    print(f'FOD directory: {fod_dir}')
    
    strands_files = sorted(glob.glob(os.path.join(tracts_dir, 'strands_*.tck')))
    fod_files = sorted(glob.glob(os.path.join(fod_dir, 'FOD_*.npy')))

    print(f"Found {len(strands_files)} tractogram files.")
    print(f"Found {len(fod_files)} FOD files.")

    # Create mappings based on 'n'
    strands_dict = {os.path.basename(f).replace('strands_', '').replace('.tck', ''): f for f in strands_files}
    fod_dict = {os.path.basename(f).replace('FOD_', '').replace('.npy', ''): f for f in fod_files}

    latent_files = glob.glob(os.path.join(latents_dir, 'z_*.npy'))
    latent_ns = set([os.path.basename(f).replace('z_', '').replace('.npy', '') for f in latent_files])

    matched_ns = set(strands_dict.keys()) & set(fod_dict.keys()) & latent_ns

    if not matched_ns:
        raise ValueError("No matching tract, FOD, and latent files found.")

    print(f"Total matched 'n's before filtering: {len(matched_ns)}")

    streamlines_dict = {}
    latents_dict = {}
    mri_paths_dict = {}

    for n in tqdm(matched_ns, desc="Processing Tractograms"):
        strands_path = strands_dict[n]
        try:

            # Attempt to load tractogram
            tract = nib.streamlines.load(strands_path)

            streamlines = tract.streamlines

            # Debugging: Streamline length
            if len(streamlines) == 0:
                print(f"Warning: No streamlines found in file: {strands_path}")
                continue

            if len(streamlines) < min_streamlines:
                print(f"Skipping {strands_path} with {len(streamlines)} streamlines (minimum required: {min_streamlines})")
                continue

            # Load latent vector
            latent_path = os.path.join(latents_dir, f'z_{n}.npy')

            # Check if latent file exists
            if not os.path.exists(latent_path):
                print(f"Latent file {latent_path} does not exist. Skipping n={n}.")
                continue

            try:

                # Attempt to load the latent file
                latent_data = np.load(latent_path, allow_pickle=True)  # allow_pickle=True for compatibility with some formats

                # Convert latent data to torch tensor
                if isinstance(latent_data, np.ndarray):
                    z = torch.tensor(latent_data, dtype=torch.float32).squeeze(1).squeeze(1).squeeze(1)
                else:
                    raise TypeError(f"Unexpected latent data type: {type(latent_data)}. Expected np.ndarray.")

            except Exception as e:
                print(f"Error loading latent vector {latent_path}: {e}. Skipping n={n}.")
                continue

            resampled_streamlines = [resample_streamline(sl, num_points) for sl in streamlines]

            streamlines_dict[n] = resampled_streamlines
            latents_dict[n] = [z] * len(resampled_streamlines)
            mri_paths_dict[n] = [n] * len(resampled_streamlines)

        except Exception as e:
            print(f"Error processing {strands_path}: {e}. Skipping.")
            continue

    eligible_ns = list(streamlines_dict.keys())
    print(f"Total eligible 'n's after filtering: {len(eligible_ns)}")

    if not eligible_ns:
        raise ValueError("No eligible 'n's found after filtering based on minimum streamlines.")

    # Splitting and storing into HDF5
    random.shuffle(eligible_ns)
    num_train = int(0.7 * len(eligible_ns))
    num_val = int(0.15 * len(eligible_ns))

    train_ns = eligible_ns[:num_train]
    val_ns = eligible_ns[num_train:num_train + num_val]
    test_ns = eligible_ns[num_train + num_val:]

    print(f"Train 'n's: {len(train_ns)}")
    print(f"Validation 'n's: {len(val_ns)}")
    print(f"Test 'n's: {len(test_ns)}")

    splits = {
        'train': train_ns,
        'val': val_ns,
        'test': test_ns
    }

    # Function to write HDF5
    def write_hdf5(split, ns, stream_dict, latent_dict, mri_dict, hdf5_dir, num_points, latent_dim=256):
        hdf5_path = os.path.join(hdf5_dir, f"{split}.hdf5")
        all_streamlines = []
        all_latents = []
        all_mri_paths = []

        for n in ns:
            all_streamlines.extend(stream_dict[n])
            all_latents.extend(latent_dict[n])
            all_mri_paths.extend(mri_dict[n])

        num_samples = len(all_streamlines)
        print(f"Writing {split} data to {hdf5_path}. Number of samples: {num_samples}")

        with h5py.File(hdf5_path, 'w') as h5f:
            h5f.create_dataset('streamlines', shape=(num_samples, 3, num_points), dtype='float32')
            h5f.create_dataset('latents', shape=(num_samples, latent_dim), dtype='float32')
            dt = h5py.string_dtype(encoding='utf-8')
            h5f.create_dataset('mri_paths', shape=(num_samples,), dtype=dt)

            for i, (sl, z, n) in enumerate(tqdm(zip(all_streamlines, all_latents, all_mri_paths),
                                               total=num_samples, desc=f"Writing {split} data")):
                h5f['streamlines'][i] = sl
                h5f['latents'][i] = z
                h5f['mri_paths'][i] = n

        print(f"{split.capitalize()} data written to {hdf5_path} with {num_samples} streamlines.")

    for split, ns in splits.items():
        write_hdf5(split, ns, streamlines_dict, latents_dict, mri_paths_dict, hdf5_dir, num_points)

    print("Preprocessing and HDF5 storage completed.")




# -----------------------------
# Define the CombinedHDF5Dataset Class with Caching
# -----------------------------

class CombinedHDF5Dataset(Dataset):
    def __init__(self, hdf5_path, transform_tract=None, cache_size=10000):
        """
        Initializes the dataset to read from an HDF5 file with caching.

        Args:
            hdf5_path (str): Path to the HDF5 file containing streamlines and latents.
            transform_tract (callable, optional): Transformations to apply to tract data.
            cache_size (int, optional): Number of samples to cache in memory. Defaults to 10000.
        """
        self.hdf5_path = hdf5_path
        self.transform_tract = transform_tract
        self.cache_size = cache_size
        self.cache = OrderedDict()

        # Open the HDF5 file
        self.h5f = h5py.File(self.hdf5_path, 'r')
        self.streamlines = self.h5f['streamlines']
        self.latents = self.h5f['latents']
        self.mri_paths = self.h5f['mri_paths']

        print(f"Loaded HDF5 file: {self.hdf5_path} with {len(self)} samples.")

    def __len__(self):
        return self.streamlines.shape[0]

    def __getitem__(self, idx):
        if idx in self.cache:
            streamline, z, mri_path = self.cache[idx]
            # Move the accessed item to the end to show that it was recently used
            self.cache.move_to_end(idx)
        else:
            streamline = self.streamlines[idx]  # Shape: (3, num_points)
            z = self.latents[idx]               # Shape: (latent_dim,)
            mri_path = self.mri_paths[idx].decode('utf-8')  # Convert bytes to string

            # Add to cache
            self.cache[idx] = (streamline, z, mri_path)
            if len(self.cache) > self.cache_size:
                # Remove the first (least recently used) item
                removed_idx, _ = self.cache.popitem(last=False)

        # Apply transformations
        if self.transform_tract:
            transformed = self.transform_tract({'streamline': streamline})  # Transforms expect (num_points, 3)
            streamline_tensor = transformed['streamline']
        else:
            streamline_tensor = torch.tensor(streamline, dtype=torch.float32)

        z_tensor = torch.tensor(z, dtype=torch.float32)

        return {
            'streamline': streamline_tensor,  # Shape: (3, num_points)
            'latent': z_tensor,              # Shape: (latent_dim,)
            'mri_path': mri_path             # MRI file path (string)
        }

    def __del__(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()
            print(f"Closed HDF5 file: {self.h5f.filename}")


# -----------------------------
# Prepare DataLoaders
# -----------------------------

def prepare_data_loaders(hdf5_dir, batch_size=1024, num_workers=4, cache_size=10000):
    """
    Prepares DataLoaders for training, validation, and testing datasets.

    Args:
        hdf5_dir (str): Directory containing HDF5 files ('train.hdf5', 'val.hdf5', 'test.hdf5').
        batch_size (int, optional): Batch size for DataLoaders. Defaults to 1024.
        num_workers (int, optional): Number of worker processes for DataLoaders. Defaults to 4.
        cache_size (int, optional): Number of samples to cache in memory. Defaults to 10000.

    Returns:
        tuple: Train DataLoader, Validation DataLoader, Test DataLoader.
    """
    # Define transforms
    common_transforms = transforms.Compose([
        ToTensor(),
    ])

    # Paths to HDF5 files
    train_hdf5 = os.path.join(hdf5_dir, 'train.hdf5')
    val_hdf5 = os.path.join(hdf5_dir, 'val.hdf5')
    test_hdf5 = os.path.join(hdf5_dir, 'test.hdf5')

    # Initialize Datasets
    train_dataset = CombinedHDF5Dataset(
        hdf5_path=train_hdf5,
        transform_tract=common_transforms,
        cache_size=cache_size
    )

    val_dataset = CombinedHDF5Dataset(
        hdf5_path=val_hdf5,
        transform_tract=common_transforms,
        cache_size=cache_size
    )

    test_dataset = CombinedHDF5Dataset(
        hdf5_path=test_hdf5,
        transform_tract=common_transforms,
        cache_size=cache_size
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    print("DataLoaders for training, validation, and testing are ready.")

    return train_loader, val_loader, test_loader

