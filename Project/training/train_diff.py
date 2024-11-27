import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F

def process_epoch(model, inferer, data_loader, epoch, device, optimizer, diffusion_scheduler, n_epochs, scaler=None, training=True):
    model.train() if training else model.eval()

    epoch_loss = 0

    for batch in data_loader:


        # Print batch information for the first few steps

        streamlines = batch['streamline'].to(device)  # Corrected key
        z_mri = batch['latent'].to(device)
        z_mri = z_mri.unsqueeze(1)

        B = streamlines.shape[0]

        # Sample random timesteps for each streamline
        timesteps = torch.randint(
            0, diffusion_scheduler.num_train_timesteps, (B,), device=device
        ).long()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast():
            # Generate random noise
            noise = torch.randn_like(streamlines).to(device)

            # Get model prediction
            noise_pred = inferer(
                inputs=streamlines,        # Ensure inputs shape matches expectations
                diffusion_model=model,
                noise=noise,
                timesteps=timesteps,
                condition=z_mri,
                mode='crossattn'
            )

            # Compute loss
        loss = F.mse_loss(noise_pred.float(), noise.float())
        epoch_loss += loss.item()

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


    return epoch_loss



def train_model(model, inferer, train_loader, val_loader, diffusion_scheduler, n_epochs, val_interval, device, experiment):
    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(n_epochs):
        # Training phase
        train_loss = process_epoch(
            model, inferer, train_loader, epoch, device, optimizer, diffusion_scheduler, n_epochs, scaler=scaler, training=True
        )
        avg_train_loss = train_loss / len(train_loader)
        print(f"Train Ep [{epoch+1}/{n_epochs}], RL: {avg_train_loss:.4f}")
        experiment.log_metric('train_loss', avg_train_loss, step=epoch)
        # Validation phase
        if (epoch+1) % val_interval == 0:
            print('Running Validation')
            val_loss = process_epoch(
                model, inferer, val_loader, epoch, device, optimizer, diffusion_scheduler, n_epochs, scaler=scaler, training=False
            )

            avg_val_loss = val_loss / len(val_loader)
            print(f"Val Ep [{epoch+1}/{n_epochs}], RL: {avg_val_loss:.4f}")
            experiment.log_metric('val_loss', avg_val_loss, step=epoch)
            # Save best model
            current_val_loss = avg_val_loss
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), "results/Tracts/test_best_model_2.pth")
                print(f"New best model found at epoch {best_epoch} with validation loss {best_val_loss:.4f}.")

        lr_scheduler.step()  # Adjust learning rate using the learning rate scheduler
