import argparse
import os

from cnn_transformer_only.interpretability.shap_runner import run_shap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SHAP for CNN-Transformer checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to cnn_transformer_ids.pth")
    p.add_argument("--data", required=True, help="Path to CICIDS CSV")
    p.add_argument("--output-dir", default="artifacts/shap", help="Where to write SHAP outputs")
    p.add_argument("--sample", action="store_true", help="Use smaller SHAP sizes")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.sample:
        csv = run_shap(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            output_dir=args.output_dir,
            background_size=200,
            eval_size=200,
            eval_pool=500,
            chunk_size=256,
        )
    else:
        csv = run_shap(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            output_dir=args.output_dir,
        )

    print(f"SHAP CSV: {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
