from dataclasses import dataclass


@dataclass
class CNNTransformerConfig:
    input_path: str = "data/cicids2017/cicids2017.csv"
    output_dir: str = "artifacts"
    csv_chunksize: int = 200_000
    max_rows: int = 0  # 0 = load all rows
    val_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 42
    epochs: int = 25
    batch_size: int = 1024
    val_batch_size: int = 2048
    lr: float = 3e-5
    weight_decay: float = 5e-3
    label_smoothing: float = 0.1
    conv_channels: int = 64
    num_layers: int = 2
    num_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.3
    undersampling_ratio: float = 0.15
    ig_steps: int = 32
    ig_samples: int = 512
    num_workers: int = 2
    cnn_fc_dim: int = 128  # hidden dim for standalone CNN classifier head
    grouped_split: bool = True
    correlation_threshold: float = 0.95
    skew_threshold: float = 5.0
    near_dup_decimals: int = 3  # round features to N decimals before dedup (0 = disabled)
    warmup_epochs: int = 2  # LR warmup epochs (critical for Transformer stability)
    patience: int = 4  # early stopping patience (epochs without ROC-AUC improvement)
