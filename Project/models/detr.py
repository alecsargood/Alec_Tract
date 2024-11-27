import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import TransformerBlock

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder to process the latent embedding.

    Args:
        latent_dim (int): Dimension of the latent embedding (256).
        model_dim (int): Dimension of the model (e.g., 256).
        num_tokens (int): Number of tokens to split the embedding into.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
    """
    def __init__(
        self,
        latent_dim=256,
        model_dim=256,
        num_tokens=16,
        num_layers=6,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.num_tokens = num_tokens

        # Split the latent embedding into tokens
        self.token_proj = nn.Linear(latent_dim, model_dim * num_tokens)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(num_tokens, model_dim))

        # Transformer Encoder Blocks with flash attention enabled
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=model_dim,
                mlp_dim=dim_feedforward,
                num_heads=num_heads,
                dropout_rate=dropout,
                qkv_bias=True,
                causal=False,
                sequence_length=num_tokens,
                with_cross_attention=False,
                use_flash_attention=True  # Enabled flash attention
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Forward pass of the Transformer Encoder.

        Args:
            x (torch.Tensor): Latent embedding of shape [batch_size, latent_dim].

        Returns:
            torch.Tensor: Encoder output of shape [batch_size, num_tokens, model_dim].
        """

        batch_size = x.size(0)

        tokens = self.token_proj(x)  # Shape: [batch_size, model_dim * num_tokens]
        tokens = tokens.view(batch_size, self.num_tokens, self.model_dim)  # [batch_size, num_tokens, model_dim]

        # Add positional encoding
        tokens = tokens + self.positional_encoding.unsqueeze(0)  # [batch_size, num_tokens, model_dim]

        # Pass through Transformer Encoder Blocks
        for block in self.encoder_blocks:

            tokens = block(tokens)

        return tokens  # [batch_size, num_tokens, model_dim]


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder to process the latent embedding.
    
    Args:
        latent_dim (int): Dimension of the latent embedding (256).
        model_dim (int): Dimension of the model (e.g., 256).
        num_tokens (int): Number of tokens to split the embedding into.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
    """
    def __init__(
        self,
        latent_dim=256,
        model_dim=256,
        num_tokens=16,
        num_layers=6,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.num_tokens = num_tokens

        # Split the latent embedding into tokens
        self.token_proj = nn.Linear(latent_dim, model_dim * num_tokens)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(num_tokens, model_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Forward pass of the Transformer Encoder.

        Args:
            x (torch.Tensor): Latent embedding of shape [batch_size, latent_dim].

        Returns:
            torch.Tensor: Encoder output of shape [batch_size, num_tokens, model_dim].
        """
        batch_size = x.size(0)

        # Project and reshape to get tokens
        tokens = self.token_proj(x)  # Shape: [batch_size, model_dim * num_tokens]
        tokens = tokens.view(batch_size, self.num_tokens, self.model_dim)  # [batch_size, num_tokens, model_dim]

        # Add positional encoding
        tokens += self.positional_encoding.unsqueeze(0)  # [batch_size, num_tokens, model_dim]

        # Prepare for Transformer Encoder (requires shape [num_tokens, batch_size, model_dim])
        tokens = tokens.permute(1, 0, 2)  # [num_tokens, batch_size, model_dim]

        # Pass through Transformer Encoder
        encoder_output = self.transformer_encoder(tokens)  # [num_tokens, batch_size, model_dim]

        # Permute back to [batch_size, num_tokens, model_dim]
        encoder_output = encoder_output.permute(1, 0, 2)

        return encoder_output


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder to generate streamlines from encoder outputs.

    Args:
        model_dim (int): Dimension of the model (should match encoder).
        num_streamlines (int): Number of streamlines to generate (32).
        num_decoder_layers (int): Number of Transformer decoder layers.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
    """
    def __init__(
        self,
        model_dim=256,
        num_streamlines=16,
        num_decoder_layers=4,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super(TransformerDecoder, self).__init__()
        self.model_dim = model_dim
        self.num_streamlines = num_streamlines

        # Learnable query embeddings for streamlines
        self.query_embeddings = nn.Parameter(torch.randn(num_streamlines, model_dim))

        # Transformer Decoder Blocks with flash attention enabled
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=model_dim,
                mlp_dim=dim_feedforward,
                num_heads=num_heads,
                dropout_rate=dropout,
                qkv_bias=True,
                causal=False,  # Explanation below
                sequence_length=num_streamlines,
                with_cross_attention=True,
                use_flash_attention=True  # Enabled flash attention
            )
            for _ in range(num_decoder_layers)
        ])

        # Output MLP to map from model_dim to streamline points
        self.output_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 32 * 3)  # 32 points with (x, y, z)
        )

    def forward(self, encoder_output):
        """
        Forward pass of the Transformer Decoder.

        Args:
            encoder_output (torch.Tensor): Encoder output [batch_size, num_tokens, model_dim].

        Returns:
            torch.Tensor: Generated streamlines [batch_size, num_streamlines, 32, 3].
        """
        batch_size = encoder_output.size(0)

        # Prepare query embeddings
        queries = self.query_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_streamlines, model_dim]

        # Pass through Transformer Decoder Blocks
        x = queries
        for block in self.decoder_blocks:
            x = block(x, context=encoder_output)

        # Generate streamlines
        streamlines = self.output_mlp(x)  # [batch_size, num_streamlines, 96]
        streamlines = streamlines.view(batch_size, self.num_streamlines, 32, 3)  # [batch_size, num_streamlines, 32, 3]

        return streamlines

class DETRModel(nn.Module):
    """
    Complete model combining Transformer Encoder and Decoder for tract generation.

    Args:
        latent_dim (int): Dimension of the latent embedding (256).
        model_dim (int): Dimension of the model (256).
        num_tokens (int): Number of tokens to split the embedding into.
        num_streamlines (int): Number of streamlines to generate (32).
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
    """
    def __init__(
        self,
        latent_dim=256,
        model_dim=256,
        num_tokens=16,
        num_streamlines=16,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        dim_feedforward=2048,
    ):
        super(DETRModel, self).__init__()
        self.encoder = TransformerEncoder(
            latent_dim=latent_dim,
            model_dim=model_dim,
            num_tokens=num_tokens,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
        )
        self.decoder = TransformerDecoder(
            model_dim=model_dim,
            num_streamlines=num_streamlines,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, x):
        """
        Forward pass of the complete model.

        Args:
            x (torch.Tensor): Latent embedding [batch_size, latent_dim].

        Returns:
            torch.Tensor: Generated streamlines [batch_size, num_streamlines, 32, 3].
        """
        encoder_output = self.encoder(x)  # [batch_size, num_tokens, model_dim]
        streamlines = self.decoder(encoder_output)  # [batch_size, num_streamlines, 32, 3]
        return streamlines
