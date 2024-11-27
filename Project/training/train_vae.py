# training.py

import logging
import torch
import optuna
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
import os
import psutil
import gc  # Import garbage collector
from .losses import compute_generator_loss, compute_discriminator_loss, get_recon_loss, get_adv_patch_loss

# Configure logging for training
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def log_memory(step_description):
    """
    Logs the current GPU and CPU memory usage.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # Convert to MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)    # Convert to MB

        logging.info(
            f"{step_description} - GPU Allocated: {allocated:.2f} MB, "
            f"GPU Reserved: {reserved:.2f} MB, "
            f"GPU Max Allocated: {max_allocated:.2f} MB, "
            f"GPU Max Reserved: {max_reserved:.2f} MB"
        )
        print(
            f"{step_description} - GPU Allocated: {allocated:.2f} MB, "
            f"GPU Reserved: {reserved:.2f} MB, "
            f"GPU Max Allocated: {max_allocated:.2f} MB, "
            f"GPU Max Reserved: {max_reserved:.2f} MB"
        )
    else:
        logging.info(f"{step_description} - CUDA not available.")
        print(f"{step_description} - CUDA not available.")

    # Log system (CPU) memory
    virtual_mem = psutil.virtual_memory()
    used_gb = virtual_mem.used / (1024 ** 3)     # Convert to GB
    available_gb = virtual_mem.available / (1024 ** 3)  # Convert to GB
    logging.info(
        f"{step_description} - CPU Memory: {used_gb:.2f} GB used, "
        f"{available_gb:.2f} GB available"
    )
    print(
        f"{step_description} - CPU Memory: {used_gb:.2f} GB used, "
        f"{available_gb:.2f} GB available"
    )


def process_epoch(
    autoencoder,
    discriminator,
    data_loader,
    reconstruction_loss_fn,
    adv_loss_fn,
    kl_weight,
    adv_weight,
    autoencoder_warmup_epochs,
    epoch,
    device,
    optimizer_g=None,
    optimizer_d=None,
    scaler=None,
    training=True,
    trial=None
):
    """
    Processes a single epoch for training or validation.
    """
    phase = 'Training' if training else 'Validation'
    log_memory(f"Start of {phase} Epoch {epoch + 1}")

    if training:
        autoencoder.train()
        discriminator.train()
    else:
        autoencoder.eval()
        discriminator.eval()

    epoch_recon_loss, epoch_gen_adv_loss, epoch_disc_loss = 0.0, 0.0, 0.0
    total_samples = 0

    # Disable gradient tracking during validation
    with torch.set_grad_enabled(training):
        for batch_idx, batch in enumerate(data_loader):
            images = batch["image"].to(device, non_blocking=True)
            batch_size = images.size(0)
            total_samples += batch_size

            if training and optimizer_g is not None:
                optimizer_g.zero_grad(set_to_none=True)

            # Use autocast for mixed precision
            with autocast():
                # Compute generator loss
                loss_g, recon_loss, gen_adv_loss = compute_generator_loss(
                    autoencoder,
                    discriminator,
                    images,
                    reconstruction_loss_fn,
                    adv_loss_fn,
                    kl_weight,
                    adv_weight,
                    autoencoder_warmup_epochs,
                    epoch,
                    device,
                    training=training
                )

            # loss_g is used for backward; do NOT detach
            if training and scaler is not None:
                scaler.scale(loss_g).backward()
                scaler.step(optimizer_g)
                scaler.update()

            # Train Discriminator
            if epoch >= autoencoder_warmup_epochs and training and optimizer_d is not None:
                optimizer_d.zero_grad(set_to_none=True)
                with autocast():
                    # Detach autoencoder output to prevent gradients flowing to autoencoder
                    recon_images = autoencoder(images)[0].detach()
                    loss_d = compute_discriminator_loss(
                        discriminator,
                        images,
                        recon_images,
                        adv_loss_fn,
                        adv_weight,
                        device
                    )
                if training and scaler is not None:
                    scaler.scale(loss_d).backward()
                    scaler.step(optimizer_d)
                    scaler.update()

                # Accumulate discriminator loss
                if gen_adv_loss is not None:
                    epoch_gen_adv_loss += gen_adv_loss.item() * batch_size  # Multiply by batch size
                epoch_disc_loss += loss_d.item() * batch_size  # Multiply by batch size

                del loss_d  # Delete to free memory

            # Accumulate reconstruction loss
            epoch_recon_loss += recon_loss.item() * batch_size  # Multiply by batch size

            # Clean up intermediate variables to free memory
            if gen_adv_loss is not None:
                del gen_adv_loss
                gen_adv_loss = None
            del images, loss_g, recon_loss
            images, loss_g, recon_loss = None, None, None
            gc.collect()
            torch.cuda.empty_cache()

    # Calculate average losses over all samples
    epoch_recon_loss /= total_samples
    epoch_gen_adv_loss /= total_samples if epoch_gen_adv_loss != 0 else 1
    epoch_disc_loss /= total_samples if epoch_disc_loss != 0 else 1

    # Log epoch losses
    if training:
        logging.info(
            f"Epoch {epoch + 1} Training - Recon Loss: {epoch_recon_loss:.4f}, "
            f"Gen Adv Loss: {epoch_gen_adv_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}"
        )
    else:
        logging.info(
            f"Epoch {epoch + 1} Validation - Recon Loss: {epoch_recon_loss:.4f}, "
            f"Gen Adv Loss: {epoch_gen_adv_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}"
        )

    log_memory(f"End of {phase} Epoch {epoch + 1}")

    return epoch_recon_loss, epoch_gen_adv_loss, epoch_disc_loss


def train_model(
    autoencoder,
    discriminator,
    train_loader,
    val_loader,
    n_epochs,
    autoencoder_warmup_epochs,
    val_interval,
    device,
    lr_g=1e-4,
    lr_d=1e-4,
    kl_weight=1e-6,
    adv_weight=1e-4,
    trial=None
):
    """
    Trains the autoencoder and discriminator models.

    Returns:
    - best_recon_loss (float): Best validation reconstruction loss achieved.
    - best_model_path (str): Path to the saved best model weights within this trial.
    """
    log_memory("Start of Training")

    reconstruction_loss_fn = get_recon_loss(device)
    adv_loss_fn = get_adv_patch_loss(device)

    optimizer_g = Adam(autoencoder.parameters(), lr=lr_g)
    optimizer_d = Adam(discriminator.parameters(), lr=lr_d)

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.5)

    scaler = GradScaler()

    best_recon_loss = float('inf')
    best_epoch = -1

    # Initialize variable to store the path to the best model weights in this trial
    best_model_path = None

    for epoch in range(n_epochs):
        # Training phase
        train_recon_loss, train_gen_adv_loss, train_disc_loss = process_epoch(
            autoencoder,
            discriminator,
            train_loader,
            reconstruction_loss_fn,
            adv_loss_fn,
            kl_weight,
            adv_weight,
            autoencoder_warmup_epochs,
            epoch,
            device,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scaler=scaler,
            training=True,
            trial=trial
        )

        # Print and log training results
        print(
            f"Epoch [{epoch + 1}/{n_epochs}] Training - Recon Loss: {train_recon_loss:.4f}, "
            f"Gen Adv Loss: {train_gen_adv_loss:.4f}, Disc Loss: {train_disc_loss:.4f}"
        )
        logging.info(
            f"Epoch {epoch + 1} Training - Recon Loss: {train_recon_loss:.4f}, "
            f"Gen Adv Loss: {train_gen_adv_loss:.4f}, Disc Loss: {train_disc_loss:.4f}"
        )

        # Validation phase
        if (epoch + 1) % val_interval == 0 and (epoch + 1) >= autoencoder_warmup_epochs:
            print('Running Validation')
            logging.info(f"Running Validation for Epoch {epoch + 1}")
            with torch.no_grad():
                val_recon_loss, val_gen_adv_loss, val_disc_loss = process_epoch(
                    autoencoder,
                    discriminator,
                    val_loader,
                    reconstruction_loss_fn,
                    adv_loss_fn,
                    kl_weight,
                    adv_weight,
                    autoencoder_warmup_epochs,
                    epoch,
                    device,
                    training=False,
                    trial=trial
                )

            # Compute average validation reconstruction loss
            print(f"Validation Epoch [{epoch + 1}/{n_epochs}], Recon Loss: {val_recon_loss:.4f}")
            logging.info(f"Validation Epoch [{epoch + 1}/{n_epochs}], Recon Loss: {val_recon_loss:.4f}")

            # Pruning: Report the average validation recon loss and check for pruning
            if trial is not None:
                trial.report(val_recon_loss, epoch)
                if trial.should_prune():
                    logging.info(f"Trial {trial.number} pruned at epoch {epoch + 1} during validation")
                    print(f"Trial {trial.number} pruned at epoch {epoch + 1}")
                    raise optuna.exceptions.TrialPruned()

            # Save the best model based on reconstruction loss within this trial
            if val_recon_loss < best_recon_loss:
                best_recon_loss = val_recon_loss
                best_epoch = epoch + 1
                # Save model weights to a temporary file specific to this trial
                model_dir = f'results/dMRI/trial_{trial.number}' if trial else 'results/dMRI'
                os.makedirs(model_dir, exist_ok=True)
                best_model_path = os.path.join(
                    model_dir,
                    f'best_autoencoder_trial_{trial.number}.pth' if trial else 'best_autoencoder.pth'
                )
                torch.save(autoencoder.state_dict(), best_model_path)
                logging.info(
                    f"Trial {trial.number if trial else ''} - New best model at epoch {best_epoch} with recon loss {best_recon_loss:.4f}"
                )
                print(
                    f"Trial {trial.number if trial else ''} - New best model found at epoch {best_epoch} with recon loss {best_recon_loss:.4f}."
                )

        # Zero out gradients to release memory
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        # Step the schedulers
        scheduler_g.step()
        scheduler_d.step()

    # Clean up after training
    del autoencoder
    del discriminator
    del train_loader
    del val_loader
    del optimizer_g
    del optimizer_d
    del scheduler_g
    del scheduler_d

    autoencoder, discriminator, train_loader, val_loader = None, None, None, None
    optimizer_g, optimizer_d, scheduler_g, scheduler_d = None, None, None, None

    torch.cuda.empty_cache()
    gc.collect()
    log_memory("After cleaning up resources post Training")

    logging.info("Training completed and cleaned up memory.")
    print("Training completed and cleaned up memory.")

    return best_recon_loss, best_model_path
