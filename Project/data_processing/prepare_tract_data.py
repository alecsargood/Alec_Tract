import os
import glob
import numpy as np
import nibabel as nib
import re
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class StreamlineDataset(Dataset):
    def __init__(self, tractogram_paths, num_points=32, transform=None):
        self.tractogram_paths = tractogram_paths
        self.num_points = num_points
        self.transform = transform
        self.streamlines = []
        self._load_and_resample_streamlines()

    def _load_and_resample_streamlines(self):
        print(len(self.tractogram_paths))
        print("Loading and resampling streamlines...")
        for tract_path in self.tractogram_paths:
            try:
                tract = nib.streamlines.load(tract_path)
            except Exception as e:
                print(f"Error loading {tract_path}: {e}")
                continue

            streamlines = list(tract.streamlines)

            for sl in streamlines:
                resampled_sl = self._resample_streamline(sl, self.num_points)
                self.streamlines.append(resampled_sl)

    def _resample_streamline(self, streamline, num_points):
        original_length = streamline.shape[0]
        if original_length == 1:
            resampled_sl = np.repeat(streamline, num_points, axis=0)
        else:
            new_indices = np.linspace(0, original_length - 1, num_points)
            resampled_sl = np.vstack([
                np.interp(new_indices, np.arange(original_length), streamline[:, dim])
                for dim in range(3)
            ]).T
        return resampled_sl

    def __len__(self):
        return len(self.streamlines)

    def __getitem__(self, idx):
        streamline = self.streamlines[idx]
        sample = {'streamline': streamline}
        if self.transform:
            sample = self.transform(sample)
        return sample

def prepare_tract_data(base_dir, num_points=32):
    tractogram_dirs = glob.glob(os.path.join(base_dir, 'tractogram_*'))
    tractogram_numbers = sorted([int(re.search(r'tractogram_(\d+)', d).group(1)) for d in tractogram_dirs if re.search(r'tractogram_(\d+)', d)])

    if not tractogram_numbers:
        raise ValueError("No tractogram directories found in 'generated_data'.")

    n = max(tractogram_numbers)
    train_indices = tractogram_numbers[:int(n * 0.7)]
    val_indices = tractogram_numbers[int(n * 0.7):]

    train_paths = [os.path.join(base_dir, f'tractogram_{i}', 'strands.tck') for i in train_indices if os.path.exists(os.path.join(base_dir, f'tractogram_{i}', 'strands.tck'))]
    val_paths = [os.path.join(base_dir, f'tractogram_{i}', 'strands.tck') for i in val_indices if os.path.exists(os.path.join(base_dir, f'tractogram_{i}', 'strands.tck'))]


    train_transforms = transforms.Compose([
    ToTensor(),
    ])

    val_transforms = transforms.Compose([
    ToTensor(),
    ])

    print('Number of train tractograms: ')
    train_ds = StreamlineDataset(train_paths, num_points=num_points, transform = train_transforms)
    print('Number of val tractograms: ')
    val_ds = StreamlineDataset(val_paths, num_points=num_points, transform = val_transforms)

    return train_ds, val_ds


class ToTensor:
    def __call__(self, sample):
        streamline = sample['streamline']  # shape: (num_points, 3)
        # Transpose to shape (3, num_points) to match expected input shape for 1D Conv
        streamline = torch.tensor(streamline, dtype=torch.float32).transpose(0, 1)
        sample['streamline'] = streamline
        return sample


