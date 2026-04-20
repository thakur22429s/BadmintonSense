"""Spatial-Temporal Transformer for stroke classification (BST-inspired)."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal sequences."""

    def __init__(self, embed_dim: int, max_len: int = 100):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.size(1), :]


class SpatialTransformerBlock(nn.Module):
    """Learns inter-joint relationships within a single frame."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch * T, K, embed_dim)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalTransformerBlock(nn.Module):
    """Captures temporal dynamics across frames."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, embed_dim)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class StrokeTransformer(nn.Module):
    """Spatial-Temporal Transformer for pose-based stroke classification.

    Architecture:
        1. Per-keypoint embedding
        2. Spatial transformer: learns joint relationships per frame
        3. Temporal transformer: captures stroke dynamics over time
        4. Classification head with global average pooling
    """

    def __init__(
        self,
        num_keypoints: int = 15,
        keypoint_dim: int = 3,
        num_classes: int = 10,
        spatial_embed_dim: int = 64,
        spatial_num_heads: int = 2,
        spatial_num_layers: int = 2,
        spatial_dropout: float = 0.1,
        temporal_embed_dim: int = 128,
        temporal_num_heads: int = 4,
        temporal_num_layers: int = 3,
        temporal_dropout: float = 0.1,
        keypoint_embed_dim: int = 32,
        max_seq_len: int = 30,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim

        # Per-keypoint linear embedding
        self.keypoint_embed = nn.Linear(keypoint_dim, spatial_embed_dim)
        self.keypoint_pos = nn.Parameter(
            torch.randn(1, num_keypoints, spatial_embed_dim) * 0.02
        )

        # Spatial transformer layers
        self.spatial_layers = nn.ModuleList([
            SpatialTransformerBlock(spatial_embed_dim, spatial_num_heads, spatial_dropout)
            for _ in range(spatial_num_layers)
        ])
        self.spatial_norm = nn.LayerNorm(spatial_embed_dim)

        # Project spatial output to temporal dimension
        self.spatial_to_temporal = nn.Linear(spatial_embed_dim * num_keypoints, temporal_embed_dim)

        # Temporal transformer layers
        self.temporal_pos = PositionalEncoding(temporal_embed_dim, max_seq_len)
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBlock(temporal_embed_dim, temporal_num_heads, temporal_dropout)
            for _ in range(temporal_num_layers)
        ])
        self.temporal_norm = nn.LayerNorm(temporal_embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(temporal_embed_dim, temporal_embed_dim),
            nn.GELU(),
            nn.Dropout(temporal_dropout),
            nn.Linear(temporal_embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, num_keypoints * keypoint_dim) — flattened input.
        Returns:
            logits: (batch, num_classes)
        """
        B, T, _ = x.shape

        # Reshape to separate keypoints: (B, T, K, D)
        x = x.view(B, T, self.num_keypoints, self.keypoint_dim)

        # Spatial processing per frame
        # Reshape: (B*T, K, D)
        x = x.reshape(B * T, self.num_keypoints, self.keypoint_dim)
        x = self.keypoint_embed(x) + self.keypoint_pos

        for layer in self.spatial_layers:
            x = layer(x)
        x = self.spatial_norm(x)

        # Pool spatial: (B*T, K, spatial_embed) -> (B*T, K*spatial_embed) -> (B, T, temporal_embed)
        x = x.reshape(B * T, -1)
        x = self.spatial_to_temporal(x)
        x = x.reshape(B, T, -1)

        # Temporal processing
        x = self.temporal_pos(x)
        for layer in self.temporal_layers:
            x = layer(x)
        x = self.temporal_norm(x)

        # Global average pooling over time
        x = x.mean(dim=1)  # (B, temporal_embed)

        return self.classifier(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings before classification head (for t-SNE)."""
        B, T, _ = x.shape
        x = x.view(B, T, self.num_keypoints, self.keypoint_dim)
        x = x.reshape(B * T, self.num_keypoints, self.keypoint_dim)
        x = self.keypoint_embed(x) + self.keypoint_pos

        for layer in self.spatial_layers:
            x = layer(x)
        x = self.spatial_norm(x)

        x = x.reshape(B * T, -1)
        x = self.spatial_to_temporal(x)
        x = x.reshape(B, T, -1)

        x = self.temporal_pos(x)
        for layer in self.temporal_layers:
            x = layer(x)
        x = self.temporal_norm(x)

        return x.mean(dim=1)


def build_transformer(config: dict) -> StrokeTransformer:
    """Build Transformer model from config."""
    s_cfg = config["model"]["transformer"]["spatial"]
    t_cfg = config["model"]["transformer"]["temporal"]
    return StrokeTransformer(
        num_keypoints=config["data"]["num_keypoints"],
        keypoint_dim=config["data"]["keypoint_dims"],
        num_classes=config["classes"]["num_classes"],
        spatial_embed_dim=s_cfg["embed_dim"],
        spatial_num_heads=s_cfg["num_heads"],
        spatial_num_layers=s_cfg["num_layers"],
        spatial_dropout=s_cfg["dropout"],
        temporal_embed_dim=t_cfg["embed_dim"],
        temporal_num_heads=t_cfg["num_heads"],
        temporal_num_layers=t_cfg["num_layers"],
        temporal_dropout=t_cfg["dropout"],
        keypoint_embed_dim=config["model"]["transformer"]["keypoint_embed_dim"],
        max_seq_len=config["data"]["sequence_length"],
    )
