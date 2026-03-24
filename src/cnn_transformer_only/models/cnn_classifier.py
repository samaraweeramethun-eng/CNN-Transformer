import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """Standalone CNN classifier for binary IDS classification.

    Uses the same two-layer Conv1d architecture as CNNTokenizer but replaces
    the Transformer encoder with Global Average Pooling + FC classification head.
    """

    def __init__(
        self,
        input_dim: int,
        conv_channels: int,
        fc_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(conv_channels),
            nn.Linear(conv_channels, fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        x = x.unsqueeze(1)                    # (batch, 1, input_dim)
        features = self.conv(x)                # (batch, conv_channels, input_dim)
        pooled = features.mean(dim=2)          # (batch, conv_channels)  — GAP
        logits = self.classifier(pooled)       # (batch, 2)
        return logits
