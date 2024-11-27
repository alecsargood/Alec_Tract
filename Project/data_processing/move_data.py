import os
import nibabel as nib
import numpy as np
import shutil

# Define directories
data_dir = '/cluster/project2/CU-MONDAI/Alec_Tract/NFG/generated_data'
fodf2_dir = '/cluster/project2/CU-MONDAI/Alec_Tract/Project/Data/FODF_2'
tracts_dir = '/cluster/project2/CU-MONDAI/Alec_Tract/Project/Data/Tracts'

# Create destination directories if they don't exist
os.makedirs(fodf2_dir, exist_ok=True)
os.makedirs(tracts_dir, exist_ok=True)

# Iterate over each tractogram directory in 'data'
for dir_name in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, dir_name)

    if os.path.isdir(dir_path) and dir_name.startswith('tractogram_'):
        # Extract 'n' from 'tractogram_n'
        parts = dir_name.split('_')
        if len(parts) > 1:
            n = parts[1]
        else:
        
            continue  # Skip if 'n' is not found
        # Define original file paths
        fod_path = os.path.join(dir_path, 'FOD.nii')
        strands_path = os.path.join(dir_path, 'strands.tck')
        # Check if both files exist
        if os.path.exists(fod_path) and os.path.exists(strands_path):
            # Load the MRI data from FOD.nii
            mri = nib.load(fod_path).get_fdata()
            # Process MRI data if needed (e.g., select specific channels)
            # For example, if you need only the first few channels:
            # num_sh = desired_number_of_channels
            mri = mri[..., :6]
            # Save the MRI data as a .npy file in FODF_2
            mri_npy_name = f'FOD_{n}.npy'
            mri_npy_path = os.path.join(fodf2_dir, mri_npy_name)
            np.save(mri_npy_path, mri)
            # Move and rename the strands.tck file to Tracts directory
            strands_new_name = f'strands_{n}.tck'
            strands_destination = os.path.join(tracts_dir, strands_new_name)
            shutil.copy(strands_path, strands_destination)

    print(f'Moved tracogram: {n}')