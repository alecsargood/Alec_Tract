from generative.networks.nets import AutoencoderKL
from generative.networks.nets import PatchDiscriminator
from models.diffusion import DiffusionModelUNet
from models.vq import VQVAE
from models.detr import DETRModel
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler




def create_autoencoder(num_spherical_harmonics, device, num_channels=(16, 32, 64, 128, 256, 512),
                       latent_channels=256, num_res_blocks=1, norm_num_groups=8, attention_levels=(False, False, False, False, False, False)):
    """
    Creates and returns the Autoencoder model based on the provided hyperparameters.
    """
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=num_spherical_harmonics,
        out_channels=num_spherical_harmonics,
        num_channels=num_channels,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
        attention_levels=attention_levels,
        use_flash_attention=True,
    )
    return autoencoder.to(device)

def create_vqvae(
    num_spherical_harmonics,
    device,
    num_channels=(16, 32, 64, 128),
    latent_channels=256,
    num_res_blocks=1,
    num_embeddings=512,
    embedding_dim=64,
    commitment_cost=0.25
):
    """
    Creates and returns the VQVAE model with 1 residual channel at each layer
    for computational efficiency.
    """
    # Define downsampling parameters: (stride, kernel_size, dilation, padding)
    downsample_parameters = [(2, 3, 1, 1)] * len(num_channels)
    
    # Define upsampling parameters: (stride, kernel_size, dilation, padding, output_padding)
    upsample_parameters = [(2, 2, 1, 0, 0)] * len(num_channels)

    # Set num_res_channels to 1 at each level for computational efficiency
    num_res_channels = [1] * len(num_channels)
    
    # Set embedding_dim to latent_channels to match the latent space size
    embedding_dim = latent_channels
    
    vqvae = VQVAE(
        spatial_dims=3,
        in_channels=num_spherical_harmonics,
        out_channels=num_spherical_harmonics,
        channels=num_channels,
        num_res_layers=num_res_blocks,
        num_res_channels=num_res_channels,
        downsample_parameters=downsample_parameters,
        upsample_parameters=upsample_parameters,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        dropout=0.1,
    )

    return vqvae.to(device)


def create_discriminator(num_spherical_harmonics, device, num_channels=8, num_layers_d=1):
    """
    Creates and returns the Discriminator model based on the provided hyperparameters.
    """
    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_channels=num_channels,
        in_channels=num_spherical_harmonics,
        out_channels=1,
        num_layers_d=num_layers_d,
        kernel_size=4,
        activation=("LEAKYRELU", {"negative_slope": 0.2}),
        norm="BATCH",
        bias=False,
        padding=1,
        dropout=0.1,
    )
    return discriminator.to(device)

def create_diffusion(device, cross_attention_dim = None, timesteps=1000):

    if cross_attention_dim == None:
        
        print('Loading non-conditional model')

        model = DiffusionModelUNet(
        spatial_dims=1,           # 1D data
        in_channels=3,            # x, y, z coordinates
        out_channels=3,           # Predict noise for x, y, z
        num_channels=(16, 32, 64),
        attention_levels=(False, False, True),  # Adjust as needed
        num_res_blocks=(1, 1, 1),
        num_head_channels=8,    # For attention layers
        norm_num_groups=8,
        transformer_num_layers=1,
        use_flash_attention=True

    )
    
    else:

        print('Loading conditional model')

        model = DiffusionModelUNet(
        spatial_dims=1,           # 1D data
        in_channels=3,            # x, y, z coordinates
        out_channels=3,           # Predict noise for x, y, z
        num_channels=(16, 32, 64),
        attention_levels=(False, False, True),  # Adjust as needed
        num_res_blocks=(1, 1, 1),
        num_head_channels=8,    # For attention layers
        norm_num_groups=8,
        norm_eps=1e-6,
        transformer_num_layers=3,
        with_conditioning=True,
        cross_attention_dim=cross_attention_dim,
        dropout_cattn=0.1,
        use_flash_attention=True,
        upcast_attention=True,
        resblock_updown=False

    )


    scheduler = DDIMScheduler(timesteps)
    inferer = DiffusionInferer(scheduler)
    
    return model.to(device), scheduler, inferer


def create_detr(latent_dim=256, model_dim=256, num_tokens=16, num_streamlines=16, num_encoder_layers=4, 
                num_decoder_layers=4, num_heads=8, dim_feedforward=1024, device='cuda'):
    """
    Creates and returns the Discriminator model based on the provided hyperparameters.
    """
    detr = DETRModel(
        latent_dim=latent_dim,
        model_dim=model_dim,
        num_tokens=num_tokens,
        num_streamlines=num_streamlines,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
    )
    return detr.to(device)
