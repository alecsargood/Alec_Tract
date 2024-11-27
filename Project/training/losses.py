
import torch
from generative.losses import PatchAdversarialLoss
from torch.nn import L1Loss
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F



def kl_divergence(z_mu, z_sigma, kl_weight):
    """
    Computes the Kullback-Leibler divergence loss.

    Args:
        z_mu (Tensor): Mean of the latent distribution.
        z_sigma (Tensor): Standard deviation of the latent distribution.
        kl_weight (float): Weight for the KL divergence loss.

    Returns:
        Tensor: Scaled KL divergence loss.
    """
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=[1, 2, 3, 4]
    )
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
    return kl_loss * kl_weight


def compute_generator_loss(autoencoder, discriminator, images, reconstruction_loss_fn, adv_loss_fn,
                          kl_weight, adv_weight, device, training=True):
    """
    Computes the generator (autoencoder) loss.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        discriminator (nn.Module): The discriminator model.
        images (Tensor): Batch of input images.
        reconstruction_loss_fn (nn.Module): Reconstruction loss function.
        adv_loss_fn (callable): Adversarial loss function.
        kl_weight (float): Weight for KL divergence loss.
        adv_weight (float): Weight for adversarial loss.
        device (torch.device): Device to run computations on.
        training (bool, optional): Flag indicating training or evaluation mode.

    Returns:
        tuple: (loss_g, recon_loss, gen_adv_loss, kl_loss)
    """
    reconstruction, z_mu, z_sigma = autoencoder(images)
    recon_loss = reconstruction_loss_fn(reconstruction, images)

    if training:
        kl_loss = kl_divergence(z_mu, z_sigma, kl_weight)
        logits_fake = discriminator(reconstruction.contiguous().detach().float())[-1]  # Access the last output
        gen_adv_loss = adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = recon_loss + (adv_weight * gen_adv_loss) + kl_loss
    else:
        loss_g = None
        gen_adv_loss = None
        kl_loss = None

    return loss_g, recon_loss, gen_adv_loss, kl_loss


def compute_discriminator_loss(discriminator, images, reconstruction, adv_loss_fn, adv_weight, device):
    """
    Computes the discriminator loss.

    Args:
        discriminator (nn.Module): The discriminator model.
        images (Tensor): Batch of real images.
        reconstruction (Tensor): Batch of reconstructed (fake) images.
        adv_loss_fn (callable): Adversarial loss function.
        adv_weight (float): Weight for adversarial loss.
        device (torch.device): Device to run computations on.

    Returns:
        Tensor: Discriminator loss.
    """
    # Ensure tensors are contiguous and detached
    images = images.contiguous().detach().float()
    reconstruction = reconstruction.contiguous().detach().float()

    logits_real = discriminator(images)[-1]  # Access the last output if discriminator returns a list
    loss_d_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

    logits_fake = discriminator(reconstruction)[-1]  # Access the last output
    loss_d_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)

    discriminator_loss = (loss_d_real + loss_d_fake) * 0.5
    loss_d = adv_weight * discriminator_loss

    return loss_d


def get_adv_patch_loss(device):
    """
    Initializes the adversarial patch loss.

    Args:
        device (torch.device): Device to run computations on.

    Returns:
        PatchAdversarialLoss: Initialized adversarial loss function.
    """
    return PatchAdversarialLoss(criterion="least_squares").to(device)


def get_recon_loss(device):
    """
    Initializes the reconstruction loss.

    Args:
        device (torch.device): Device to run computations on.

    Returns:
        nn.Module: Initialized reconstruction loss function.
    """
    return L1Loss().to(device)


def chamfer_loss(recon_x, x):
    """
    Computes the Chamfer loss between sets of streamlines x and recon_x,
    accounting for the fact that streamlines are not directional (flipping a streamline is equivalent).
    The loss is computed by taking the minimum MSE between the original streamline and both the 
    original and reversed versions of the reconstructed streamline.
    
    Args:
        recon_x (torch.Tensor): Reconstructed streamlines, shape (batch, m, 256, 3)
        x (torch.Tensor): Original streamlines, shape (batch, m, 256, 3)
        mu (torch.Tensor): Mean tensor for KL divergence, shape (batch, latent_dim)
        logvar (torch.Tensor): Log variance tensor for KL divergence, shape (batch, latent_dim)
        kl_weight (float): Weight for the KL divergence term
    
    Returns:
        total_loss (torch.Tensor): Combined loss
        chamfer_loss (torch.Tensor): Chamfer distance loss
        kl_loss (torch.Tensor): KL divergence loss
    """
    batch_size, m, n_pts, dims = x.size()
    
    # Reshape streamlines to (batch, m, n_pts * dims) to compute MSE between streamlines
    x_flat = x.reshape(batch_size, m, -1)  # Shape: (batch, m, 768)
    recon_x_flat = recon_x.reshape(batch_size, m, -1)  # Shape: (batch, m, 768)
    
    # Reverse the reconstructed streamlines along the streamline points (axis=2)
    recon_x_reversed = torch.flip(recon_x, dims=[2])
    recon_x_reversed_flat = recon_x_reversed.reshape(batch_size, m, -1)  # Shape: (batch, m, 768)
    
    # Expand dimensions to compute pairwise MSE between streamlines
    x_exp = x_flat.unsqueeze(2)  # Shape: (batch, m, 1, 768)
    recon_x_exp = recon_x_flat.unsqueeze(1)  # Shape: (batch, 1, m, 768)
    recon_x_reversed_exp = recon_x_reversed_flat.unsqueeze(1)  # Shape: (batch, 1, m, 768)

    # Compute squared differences (MSE) for both original and reversed configurations
    diff_sq_original = (x_exp - recon_x_exp) ** 2  # Shape: (batch, m, m, 768)
    diff_sq_reversed = (x_exp - recon_x_reversed_exp) ** 2  # Shape: (batch, m, m, 768)
    
    # Compute MSE by averaging over the last dimension (768)
    mse_original = diff_sq_original.sum(dim=3)  # Shape: (batch, m, m)
    mse_reversed = diff_sq_reversed.sum(dim=3)  # Shape: (batch, m, m)

    # Take the minimum MSE between the original and reversed configurations
    mse = torch.min(mse_original, mse_reversed)  # Shape: (batch, m, m)
    
    # For each streamline in x, find the minimum MSE to any streamline in recon_x
    min_dist_x_to_recon, _ = torch.min(mse, dim=2)  # Shape: (batch, m)
    
    # For each streamline in recon_x, find the minimum MSE to any streamline in x
    min_dist_recon_to_x, _ = torch.min(mse, dim=1)  # Shape: (batch, m)
    
    # Compute the Chamfer loss by averaging the minimum distances
    chamfer_loss = (min_dist_x_to_recon.mean(dim=1) + min_dist_recon_to_x.mean(dim=1)) / 2  # Shape: (batch,)
    
    # Average the Chamfer loss over the batch
    chamfer_loss = chamfer_loss.mean()


    return chamfer_loss


def hungarian_mse_loss_single_sample(pred, target):
    """
    Computes the Hungarian-based MSE Loss between two sets of streamlines for a single sample.
    Accounts for directionality by considering both original and reversed predictions.
    
    Args:
        pred (torch.Tensor): Predicted streamlines [m, D, C].
        target (torch.Tensor): Target streamlines [m, D, C].
    
    Returns:
        torch.Tensor: Scalar tensor representing the loss.
    """
    m, D, C = pred.size()
    
    # Reshape pred and target to [m, D * C]
    pred_flat = pred.reshape(m, -1)       # [m, D * C]
    target_flat = target.reshape(m, -1)   # [m, D * C]
    
    # Reverse target along the D dimension and flatten
    target_reversed = target.flip(dims=[1])          # [m, D, C]
    target_reversed_flat = target_reversed.reshape(m, -1)  # [m, D * C]
    
    # Compute cost matrices using PyTorch operations
    cost_original = torch.cdist(pred_flat, target_flat, p=2)           # [m, m]
    cost_reversed = torch.cdist(pred_flat, target_reversed_flat, p=2)  # [m, m]
    
    # Choose the minimum cost for each pair
    cost = torch.min(cost_original, cost_reversed)  # [m, m]
    
    # Move cost to CPU and convert to numpy for linear_sum_assignment
    cost_cpu = cost.detach().cpu().numpy()  # [m, m]
    
    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_cpu)
    
    # Get matched pairs
    pred_matched = pred[row_ind]                # [m, D, C]
    target_matched = target[col_ind]            # [m, D, C]
    target_reversed_matched = target_reversed[col_ind]  # [m, D, C]
    
    # Decide whether to use reversed target based on cost comparison
    cost_original_selected = cost_original[row_ind, col_ind]
    cost_reversed_selected = cost_reversed[row_ind, col_ind]
    use_reversed = (cost_reversed_selected < cost_original_selected).float().unsqueeze(-1).unsqueeze(-1)
    target_final = target_matched * (1 - use_reversed) + target_reversed_matched * use_reversed
    
    # Compute MSE for matched pairs
    mse = F.mse_loss(pred_matched, target_final, reduction='sum')
    mse = mse / m
    
    return mse

