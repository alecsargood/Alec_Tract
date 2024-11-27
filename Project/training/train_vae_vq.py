import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from .losses import get_recon_loss, get_adv_patch_loss
import gc  # Import garbage collection module
import psutil  # Import psutil for CPU memory monitoring
from torch.nn import L1Loss


def train_model_vq(autoencoder, discriminator, train_loader, val_loader, n_epochs, val_interval, device, experiment):
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
    l1_loss = get_recon_loss(device)
    adv_loss = get_adv_patch_loss(device)

    # Set loss weights and optimizer learning rates to align with MONAI
    kl_weight, adv_weight = 1e-6, 1e-2  # kl_weight=1e-6, adv_weight=1e-2
    optimizer_g = Adam(autoencoder.parameters(), lr=1e-4)  # Generator optimizer with lr=1e-4
    optimizer_d = Adam(discriminator.parameters(), lr=5e-4)  # Discriminator optimizer with lr=5e-4

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
                reconstruction, quantization_loss = autoencoder(images=images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                recons_loss = l1_loss(reconstruction.float(), images.float())
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = recons_loss + quantization_loss + adv_weight * generator_loss


                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss


                epoch_rec_loss += recons_loss.item()
                epoch_gen_adv_loss += generator_loss.item()
                epoch_disc_loss += discriminator_loss.item()

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()


            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

        # Compute average losses for the epoch
        avg_rec = epoch_rec_loss / len(train_loader)
        avg_gen_adv = epoch_gen_adv_loss / len(train_loader) if epoch_gen_adv_loss > 0 else 0.0
        avg_disc = epoch_disc_loss / len(train_loader) if epoch_disc_loss > 0 else 0.0

        # Log training metrics
        experiment.log_metric("train_recon_loss", avg_rec, step=epoch)
        experiment.log_metric("train_gen_adv_loss", avg_gen_adv, step=epoch)
        experiment.log_metric("train_disc_loss", avg_disc, step=epoch)




        if (epoch + 1) % val_interval == 0:
            autoencoder.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)

                    reconstruction, quantization_loss = autoencoder(images=images)

                    recon_loss = l1_loss(reconstruction.float(), images.float())

                    val_loss += recons_loss.item()

                # Optional: Print validation batch-wise loss for monitoring
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(val_loader):
                    print(f'Validation Batch [{batch_idx + 1}/{len(val_loader)}] - Recon Loss: {recon_loss.item():.4f}')

            avg_val_loss = val_loss / len(val_loader)
            experiment.log_metric('val_recon_loss', avg_val_loss, step=epoch)
            print(f"\nValidation Epoch [{epoch + 1}/{n_epochs}] - Avg Recon Loss: {avg_val_loss:.4f}")

        # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                torch.save(autoencoder.state_dict(), f"results/dMRI/VQ/{batch_size}/best_autoencoder_kl.pth")
                torch.save(discriminator.state_dict(), f"results/dMRI/VQ/{batch_size}/best_discriminator.pth")
                print(f"New best model found at epoch {best_epoch} with validation loss {best_val_loss:.4f}. Models saved.")

        # Switch back to training mode
        autoencoder.train()
        discriminator.train()

    # Garbage collection and memory management after each epoch
    gc.collect()
    torch.cuda.empty_cache()

    # Ensure that the following functions and classes are defined or imported correctly
    # PatchAdversarialLoss should be defined in .losses or correctly imported

