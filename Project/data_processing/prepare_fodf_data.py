import os
import numpy as np
import nibabel as nib
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data import Dataset
from data_processing.data_utils import split_data, get_file_paths

def calculate_num_sh_coefficients(sh_order):
    return int((sh_order + 1) * (sh_order + 2) / 2)

def convert_nii_to_npy(source_dir, data_dir, num_spherical_harmonics):
    """
    Convert .nii FOD files to .npy format, limited to spherical harmonics.

    Args:
        source_dir (str): Directory containing source FOD files.
        data_dir (str): Directory to store the output .npy files.
        num_spherical_harmonics (int): Number of spherical harmonics to keep.
    """
    for idx, folder_name in enumerate(os.listdir(source_dir)):
        if (idx+1)%100 == 0:
            print(f'Converted image {idx+1}')
        folder_path = os.path.join(source_dir, folder_name)
        fod_file_path = os.path.join(folder_path, "FOD.nii")
        if os.path.isfile(fod_file_path):
            sh_img = nib.load(fod_file_path)
            sh_data = sh_img.get_fdata()[:, :, :, :num_spherical_harmonics]
            dest_file_path = os.path.join(data_dir, f"FOD_{idx+1}.npy")
            np.save(dest_file_path, sh_data)

def prepare_fodf_data(data_dir):
    """
    Prepare FODF datasets for VAE training.

    Args:
        data_dir (str): Directory containing processed .npy FODF data.

    Returns:
        tuple: Train, validation, and test datasets.
        
    """

    image_files = get_file_paths(data_dir)
    train_files, val_files, test_files = split_data(image_files)

    data_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    ])

    train_ds = Dataset(data=train_files, transform=data_transforms)
    val_ds = Dataset(data=val_files, transform=data_transforms)
    test_ds = Dataset(data=test_files, transform=data_transforms)

    return train_ds, val_ds, test_ds
