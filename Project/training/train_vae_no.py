import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from .losses import compute_generator_loss, compute_discriminator_loss, get_recon_loss, get_adv_patch_loss
import gc  # Import garbage collection module
import psutil  # Import psutil for CPU memory monitoring


def train_model_no(autoencoder, discriminator, train_loader, val_loader, n_epochs, val_interval, device, experiment):
    """
    Manages the overall training loop across multiple epochs, including validation and model checkpointing.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        discriminator (nn.Module): The discriminator model.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        n_epochs (int): Total number of training epochs.
        device (torch.device): Device to run computations on.
        experiment (object): Experiment logging object.

    Returns:
        tuple: (best_val_loss, best_epoch)
    """
    # Initialize loss functions
    reconstruction_loss_fn = get_recon_loss(device)
    adv_loss_fn = get_adv_patch_loss(device)

    # Set loss weights and optimizer learning rates to align with MONAI
    kl_weight, adv_weight = 1e-6, 1e-2  # kl_weight=1e-6, adv_weight=1e-2
    optimizer_g = Adam(autoencoder.parameters(), lr=5e-5)  # Generator optimizer with lr=1e-4
    optimizer_d = Adam(discriminator.parameters(), lr=1e-5)  # Discriminator optimizer with lr=5e-4

    # Initialize gradient scalers for mixed precision training
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(n_epochs):
        print(f'\nRunning epoch: {epoch + 1}/{n_epochs}')
        autoencoder.train()
        discriminator.train()

        epoch_rec_loss = 0.0
        epoch_gen_adv_loss = 0.0
        epoch_disc_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            
            if epoch == 1 and batch_idx == 1:
                batch_size = images.shape[0]
                print(f"Batch Size: {batch_size}")

            # Zero gradients for generator
            optimizer_g.zero_grad(set_to_none=True)

            with autocast():
                # Compute generator loss and related components
                loss_g, recon_loss, gen_adv_loss, kl_loss = compute_generator_loss(
                    autoencoder, discriminator, images, reconstruction_loss_fn,
                    adv_loss_fn, kl_weight, adv_weight, device, training=True
                )

            if loss_g is not None:
                # Backward pass and optimizer step for generator
                scaler_g.scale(loss_g).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()

            # Reuse the reconstruction from generator loss computation
            with torch.no_grad():
                reconstruction = autoencoder(images)[0].detach().contiguous().float()

            # Zero gradients for discriminator
            optimizer_d.zero_grad(set_to_none=True)

            with autocast():
                # Compute discriminator loss
                loss_d = compute_discriminator_loss(discriminator, images, reconstruction, adv_loss_fn, adv_weight, device)

            # Backward pass and optimizer step for discriminator
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # Accumulate losses for logging
            epoch_disc_loss += loss_d.item()
            if gen_adv_loss is not None:
                epoch_gen_adv_loss += gen_adv_loss.item()
            epoch_rec_loss += recon_loss.item()

            # Optional: Print batch-wise loss for monitoring
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}] - '
                      f'Recon Loss: {recon_loss.item():.4f}, '
                      f'Gen Adv Loss: {gen_adv_loss.item():.4f}, '
                      f'Disc Loss: {loss_d.item():.4f}')

        # Compute average losses for the epoch
        avg_rec = epoch_rec_loss / len(train_loader)
        avg_gen_adv = epoch_gen_adv_loss / len(train_loader) if epoch_gen_adv_loss > 0 else 0.0
        avg_disc = epoch_disc_loss / len(train_loader) if epoch_disc_loss > 0 else 0.0

        # Log training metrics
        experiment.log_metric("train_recon_loss", avg_rec, step=epoch)
        experiment.log_metric("train_gen_adv_loss", avg_gen_adv, step=epoch)
        experiment.log_metric("train_disc_loss", avg_disc, step=epoch)
        experiment.log_metric("kl_loss", kl_loss.item(), step=epoch)

        print(f'Epoch [{epoch + 1}/{n_epochs}] - '
              f'Avg Recon Loss: {avg_rec:.4f}, '
              f'Avg Gen Adv Loss: {avg_gen_adv:.4f}, '
              f'Avg Disc Loss: {avg_disc:.4f}')

        # Validation phase
        if ((epoch + 1) % val_interval) == 0:
            print('\nRunning Validation')
            autoencoder.eval()
            discriminator.eval()

            val_recon_loss = 0.0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    images = batch["image"].to(device)
                    reconstruction, _, _ = autoencoder(images)
                    recon_loss = reconstruction_loss_fn(reconstruction, images)
                    val_recon_loss += recon_loss.item()

                    # Optional: Print validation batch-wise loss for monitoring
                    if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(val_loader):
                        print(f'Validation Batch [{batch_idx + 1}/{len(val_loader)}] - Recon Loss: {recon_loss.item():.4f}')

            avg_val_loss = val_recon_loss / len(val_loader)
            experiment.log_metric('val_recon_loss', avg_val_loss, step=epoch)
            print(f"\nValidation Epoch [{epoch + 1}/{n_epochs}] - Avg Recon Loss: {avg_val_loss:.4f}")

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                torch.save(autoencoder.state_dict(), f"results/dMRI/{batch_size}/best_autoencoder_kl.pth")
                torch.save(discriminator.state_dict(), f"results/dMRI/{batch_size}/best_discriminator.pth")
                print(f"New best model found at epoch {best_epoch} with validation loss {best_val_loss:.4f}. Models saved.")

            # Switch back to training mode
            autoencoder.train()
            discriminator.train()

        # Garbage collection and memory management after each epoch
        gc.collect()
        torch.cuda.empty_cache()

    # Ensure that the following functions and classes are defined or imported correctly
    # PatchAdversarialLoss should be defined in .losses or correctly imported

