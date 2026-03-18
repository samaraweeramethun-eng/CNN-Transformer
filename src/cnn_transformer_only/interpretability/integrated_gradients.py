import os

import numpy as np
import pandas as pd
import torch


def integrated_gradients(model, inputs, baseline=None, steps=32, target_class=1):
    model.eval()
    device = inputs.device
    if baseline is None:
        baseline = torch.zeros_like(inputs, device=device)
    total_gradients = torch.zeros_like(inputs)
    with torch.enable_grad():
        for alpha in torch.linspace(0, 1, steps, device=device):
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            outputs = model(interpolated)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            target = outputs[:, target_class].sum()
            grads = torch.autograd.grad(target, interpolated, retain_graph=False)[0]
            total_gradients += grads
    avg_gradients = total_gradients / steps
    return (inputs - baseline) * avg_gradients


def generate_ig_report(
    model,
    X_val,
    feature_names,
    output_dir,
    steps=32,
    sample_size=512,
    seed=42,
    prefix="cnn_transformer",
):
    if X_val.shape[0] == 0:
        return ""
    os.makedirs(output_dir, exist_ok=True)
    sample_count = min(sample_size, X_val.shape[0])
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(X_val.shape[0], sample_count, replace=False)
    data = torch.FloatTensor(X_val[sample_idx]).to(next(model.parameters()).device)
    baseline = torch.FloatTensor(X_val.mean(axis=0, keepdims=True)).to(next(model.parameters()).device)
    ig_values = []
    model.eval()
    for chunk in torch.split(data, 128):
        base_chunk = baseline.expand(chunk.size(0), -1)
        ig_chunk = integrated_gradients(model, chunk, baseline=base_chunk, steps=steps)
        ig_values.append(ig_chunk.detach().cpu())
    ig_tensor = torch.cat(ig_values, dim=0)
    importance = ig_tensor.abs().mean(dim=0).numpy()
    importance_df = (
        pd.DataFrame({"feature": feature_names, "avg_abs_integrated_grad": importance})
        .sort_values("avg_abs_integrated_grad", ascending=False)
    )
    csv_path = os.path.join(output_dir, f"{prefix}_integrated_gradients.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved Integrated Gradients ranking -> {csv_path}")
    return csv_path
