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
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    conv_channels: int = 96
    num_layers: int = 3
    num_heads: int = 8
    d_model: int = 192
    d_ff: int = 768
    dropout: float = 0.2
    undersampling_ratio: float = 0.15
    ig_steps: int = 32
    ig_samples: int = 512
    num_workers: int = 2
    cnn_fc_dim: int = 128  # hidden dim for standalone CNN classifier head
