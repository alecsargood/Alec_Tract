import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data import Dataset

def calculate_num_sh_coefficients(sh_order):
    return int((sh_order + 1) * (sh_order + 2) / 2)

def load_data(source_dir, data_dir, num_spherical_harmonics):
    # Uncomment and adjust the following block if you need to convert .nii files to .npy
    
    # for idx, folder_name in enumerate(os.listdir(source_dir)):
    #     print(f'Processing data {idx}')
    #     folder_path = os.path.join(source_dir, folder_name)
    #     fod_file_path = os.path.join(folder_path, "FOD.nii")
    #     if os.path.isfile(fod_file_path):
    #         sh_img = nib.load(fod_file_path)
    #         sh_data = sh_img.get_fdata()[:,:,:,:num_spherical_harmonics]
    #         dest_file_path = os.path.join(data_dir, f"FOD_{idx+1}.npy")
    #         np.save(dest_file_path, sh_data)

    image_files = [
        {"image": os.path.join(data_dir, fname)}
        for fname in os.listdir(data_dir) if fname.endswith('.npy')
    ]
    return image_files

def split_data(image_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train_files, temp_files = train_test_split(image_files, test_size=(1 - train_ratio), random_state=42, shuffle=True)
    val_size = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(temp_files, test_size=(1 - val_size), random_state=42, shuffle=True)
    return train_files, val_files, test_files

def create_datasets(train_files, val_files, test_files):

        # Apply MONAI transforms to the loaded data
    data_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    ])

    train_ds = Dataset(data=train_files, transform=data_transforms)
    val_ds = Dataset(data=val_files, transform=data_transforms)
    test_ds = Dataset(data=test_files, transform=data_transforms)
    return train_ds, val_ds, test_ds
