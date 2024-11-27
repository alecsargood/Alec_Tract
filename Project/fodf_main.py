# main.py

import argparse
import json
import os
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from data_processing.prepare_fodf_data import calculate_num_sh_coefficients, prepare_fodf_data
from models import create_autoencoder, create_discriminator
from training import train_model_no
from utils import visualize_reconstruction, setup_torch

from comet_ml import Experiment

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    """
    Main function to initialize data, models, and start training.
    """
    # Set seeds for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    device = setup_torch()
    print(f"Using device: {device}")

    # Calculate number of spherical harmonics coefficients
    num_spherical_harmonics = calculate_num_sh_coefficients(args.sh_order)
    print(f"Number of Spherical Harmonics Coefficients: {num_spherical_harmonics}")

    # Prepare datasets
    data_dir = f'/cluster/project2/CU-MONDAI/Alec_Tract/Project/Data/FODF_{args.sh_order}'
    train_ds, val_ds, test_ds = prepare_fodf_data(data_dir)
    print('Datasets prepared.')

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )
    print('DataLoaders initialized.')

    # Create models
    autoencoder = create_autoencoder(
        num_spherical_harmonics=num_spherical_harmonics,
        device=device,
        num_channels=args.autoencoder_num_channels,
        latent_channels=args.latent_channels,
        num_res_blocks=args.autoencoder_num_res_blocks,
        norm_num_groups=args.autoencoder_norm_num_groups,
        attention_levels=args.autoencoder_attention_levels
    )
    discriminator = create_discriminator(
        num_spherical_harmonics=num_spherical_harmonics,
        device=device,
        num_channels=args.discriminator_num_channels,
        num_layers_d=args.discriminator_num_layers
    )


    experiment = Experiment(
    api_key='eQ6pfPreHQzFB4frzYlCeLiEr',
    project_name='bench',
    workspace='gentract',
    auto_output_logging="simple",  # Logs stdout and stderr
    auto_metric_logging=True,      # Automatically log metrics
)

    experiment.log_parameters(vars(args))

    # Start training

    best_recon_loss = train_model_no(
        autoencoder=autoencoder,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        val_interval=args.val_interval,
        device=device,
        experiment=experiment
    )

    # Save the best validation reconstruction loss
    if args.save_best_loss:
        with open('best_recon_loss.txt', 'w') as f:
            f.write(str(best_recon_loss))
        print(f"Best validation reconstruction loss saved: {best_recon_loss}")

    # Visualize reconstruction on test set
    channels = range(num_spherical_harmonics)
    autoencoder.load_state_dict(torch.load(f'/cluster/project2/CU-MONDAI/Alec_Tract/Project/results/dMRI/{args.batch_size}/best_autoencoder_kl.pth', map_location=device))
    autoencoder.eval()
    save_dir = f'/cluster/project2/CU-MONDAI/Alec_Tract/Project/results/dMRI/{args.batch_size}'
    visualize_reconstruction(autoencoder, test_ds, device, channels=channels, save_dir = save_dir)
    print("Reconstruction visualization completed.")


    experiment.end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder with Hyperparameter Tuning")
    
    # Hyperparameters to tune via Optuna
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr_g", type=float, default=5e-5, help="Learning rate for generator")
    parser.add_argument("--lr_d", type=float, default=5e-5, help="Learning rate for discriminator")
    parser.add_argument("--kl_weight", type=float, default=1e-3, help="Weight for KL divergence loss")
    parser.add_argument("--adv_weight", type=float, default=1e-3, help="Weight for adversarial loss")
    
    # Model architecture hyperparameters
    parser.add_argument("--autoencoder_num_channels", type=str, default="16,32,64,128,256,512", help="Comma-separated channels per layer for autoencoder")
    parser.add_argument("--latent_channels", type=int, default=256, help="Latent space size for autoencoder")
    parser.add_argument("--autoencoder_num_res_blocks", type=int, default=1, help="Number of residual blocks per level in autoencoder")
    parser.add_argument("--autoencoder_norm_num_groups", type=int, default=8, help="Number of groups for GroupNorm in autoencoder")
    parser.add_argument("--autoencoder_attention_levels", type=str, default="False,False,False,False,False,True", help="Comma-separated attention levels for autoencoder (e.g., True,False,True,False,True,False)")
    parser.add_argument("--discriminator_num_channels", type=int, default=4, help="Number of channels for discriminator")
    parser.add_argument("--discriminator_num_layers", type=int, default=1, help="Number of layers for discriminator")
    
    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=1000, help="Total number of training epochs")
    parser.add_argument("--val_interval", type=int, default=50, help="Validation interval in epochs")
    
    # Other parameters
    parser.add_argument("--sh_order", type=int, default=2, help="Spherical Harmonics order")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save_best_loss", action='store_true', help="Flag to save the best reconstruction loss")
    
    args = parser.parse_args()
    
    # Convert comma-separated strings to tuples of appropriate types
    args.autoencoder_num_channels = tuple(map(int, args.autoencoder_num_channels.split(',')))
    args.autoencoder_attention_levels = tuple(map(lambda x: x.strip().lower() == 'true', args.autoencoder_attention_levels.split(',')))
    
    main(args)
