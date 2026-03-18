import argparse
import os

from cnn_transformer_only.config import CNNTransformerConfig
from cnn_transformer_only.training.cnn_trainer import train_cnn_transformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN-Transformer IDS (standalone)")
    p.add_argument("--data", required=True, help="Path to CICIDS CSV")
    p.add_argument("--output-dir", default="artifacts", help="Where to write artifacts")
    p.add_argument("--epochs", type=int, default=0, help="Override epochs (0 = config default)")
    p.add_argument("--batch-size", type=int, default=0, help="Override batch size (0 = config default)")
    p.add_argument("--sample", action="store_true", help="Use small/test-friendly hyperparameters")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = CNNTransformerConfig(input_path=args.data, output_dir=args.output_dir)

    if args.sample:
        cfg.epochs = 5
        cfg.batch_size = 64
        cfg.val_batch_size = 128
        cfg.d_model = 64
        cfg.conv_channels = 32
        cfg.num_layers = 1
        cfg.num_heads = 4
        cfg.d_ff = 256
        cfg.ig_steps = 8
        cfg.ig_samples = 128
        cfg.max_train_samples = 0

    if args.epochs and args.epochs > 0:
        cfg.epochs = args.epochs
    if args.batch_size and args.batch_size > 0:
        cfg.batch_size = args.batch_size

    os.makedirs(cfg.output_dir, exist_ok=True)

    ckpt = train_cnn_transformer(cfg)
    print(f"Checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
