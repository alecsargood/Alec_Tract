import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(data_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train, validation, and test sets.

    Args:
        data_files (list): List of data files to split.
        train_ratio (float): Ratio of the training set.
        val_ratio (float): Ratio of the validation set.
        test_ratio (float): Ratio of the test set.

    Returns:
        tuple: train, validation, and test datasets as lists.
    """
    train_files, temp_files = train_test_split(data_files, test_size=(1 - train_ratio), random_state=42, shuffle=True)
    val_size = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(temp_files, test_size=(1 - val_size), random_state=42, shuffle=True)
    return train_files, val_files, test_files

def get_file_paths(data_dir, file_extension='.npy'):
    """
    Get file paths with the specified extension from a directory.

    Args:
        data_dir (str): Directory to search.
        file_extension (str): Extension to filter by.

    Returns:
        list: List of file paths.
    """
    print(f'Data dir is: {data_dir}')
    return [
        {"image": os.path.join(data_dir, fname)}
        for fname in os.listdir(data_dir) if fname.endswith(file_extension)
    ]
