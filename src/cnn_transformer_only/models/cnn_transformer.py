import torch
import torch.nn as nn


class CNNTokenizer(nn.Module):
    def __init__(self, input_dim: int, conv_channels: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Linear(conv_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        tokens = self.conv(x).transpose(1, 2)
        tokens = self.projection(tokens)
        return self.norm(tokens)


class CNNTransformerIDS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        conv_channels: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.tokenizer = CNNTokenizer(input_dim, conv_channels, d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.positional = nn.Parameter(torch.empty(1, input_dim + 1, d_model))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self._init_weights()

    def _init_weights(self):
        """Proper initialization to prevent Transformer collapse."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        batch_size, seq_len, _ = tokens.size()
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.positional[:, : seq_len + 1]
        encoded = self.encoder(self.dropout(tokens))
        logits = self.classifier(encoded[:, 0])
        return logits
