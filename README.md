# CNN-Transformer (Testing)

Standalone subproject containing only the CNN-Transformer intrusion detection pipeline (training + evaluation + Integrated Gradients + Grad-CAM + SHAP).

## Install (editable)

From the repo root:

```bash
pip install -e cnn-transformer-only
```

## Train (local)

```bash
python cnn-transformer-only/scripts/train.py --data data/cicids2017/cicids2017_sample.csv --output-dir artifacts --epochs 5
```

For CICIDS2018 day-wise CSV files, pass the folder (or a glob):

```bash
python scripts/train.py --data data/cicids2018 --output-dir artifacts/cicids2018 --epochs 25
```

## Kaggle

Open the notebook at `cnn-transformer-only/notebooks/kaggle_cnn_transformer_only.ipynb`.

Notes:
- The notebook auto-detects a CICIDS CSV under `/kaggle/input/**`.
- Outputs are written to `/kaggle/working/artifacts`.
