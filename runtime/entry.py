"""
entry.py — single entrypoint for the MELD runtime container.

    MELD_MODE=inference  (default)  →  load model from /artifact, predict, write CSV
    MELD_MODE=train                 →  load backbone + adapter from /input,
                                       train on /input/input.csv, write updated
                                       backbone + adapter + metadata to /output

One Docker image, two modes. The contract and data format are identical; the
only difference is the direction of the weights and whether the label column
is consumed.

Interface paths (created by the orchestrator or via shared volume):
    /input/input.csv        training/inference data
    /input/contract.yaml    the MELD contract
    /input/backbone.pt      (train only) current global backbone weights
    /input/adapter.pt       (train only, optional) site-local LoRA adapter
    /output/                writable; the container puts results here
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# These are baked into the image alongside entry.py (COPY in the Dockerfile).
from model import add_lora, backbone_state, adapter_state, build_from_config
from dwh_dataset import Preprocessor, build_xy


# ------------------------------------------------------------------ config

def _load_contract() -> dict:
    with open("/input/contract.yaml") as f:
        return yaml.safe_load(f)


def _load_model_config() -> dict:
    # Prefer a config shipped alongside the backbone in /input (so the model
    # architecture always matches the incoming weights), fall back to the one
    # baked into the image at /artifact.
    for path in ("/input/model_config.json", "/artifact/model_config.json"):
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("no model_config.json in /input or /artifact")


# ------------------------------------------------------------------ inference

def run_inference(df: pd.DataFrame, config: dict) -> None:
    """Standard MELD inference: load model, predict, write output CSV."""
    mc = _load_model_config()
    model = build_from_config(mc)
    model.load_state_dict(torch.load("/artifact/model.pt", weights_only=True))

    # If a site-local adapter exists, apply it.
    if os.path.isfile("/artifact/adapter.pt"):
        lora = mc.get("lora", {})
        add_lora(model, lora.get("rank", 4), lora.get("alpha", 12),
                 include_head=(lora.get("apply_to") == "all"))
        model.load_state_dict(torch.load("/artifact/adapter.pt", weights_only=True),
                              strict=False)

    pre = Preprocessor.load("/artifact/preprocessor.pkl")
    X = pre.transform(df)

    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32)).squeeze(-1).numpy()

    predictor_name = config.get("output_schema", {}).get("predictor", [{}])[0].get("name", "prediction")
    result = df.copy()
    result[predictor_name] = probs
    result.to_csv("/output/output.csv", index=False)
    print(f"Inference complete: {len(df)} rows → /output/output.csv")


# ------------------------------------------------------------------ training

def run_train(df: pd.DataFrame) -> None:
    """One local training round: load backbone (+adapter), train, write outputs."""
    mc = _load_model_config()
    model = build_from_config(mc)

    # Load current global backbone.
    if os.path.isfile("/input/backbone.pt"):
        model.load_state_dict(torch.load("/input/backbone.pt", weights_only=True))
        print("Loaded backbone from /input/backbone.pt")

    # Attach LoRA adapters.
    lora = mc.get("lora", {})
    add_lora(model, lora.get("rank", 4), lora.get("alpha", 12),
             include_head=(lora.get("apply_to") == "all"))

    # Load previous-round adapter if available.
    if os.path.isfile("/input/adapter.pt"):
        model.load_state_dict(torch.load("/input/adapter.pt", weights_only=True),
                              strict=False)
        print("Loaded adapter from /input/adapter.pt")

    # Prepare data.
    pre = Preprocessor.load("/artifact/preprocessor.pkl")
    target = mc.get("target", "admitted")
    X, y = build_xy(pre, df, target)
    n_samples = len(y)

    # Local training loop.
    epochs = int(os.environ.get("TRAIN_EPOCHS", "5"))
    batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "256"))
    lr = float(os.environ.get("TRAIN_LR", "0.001"))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        final_loss = epoch_loss / n_samples
        print(f"  epoch {epoch+1}/{epochs}  loss={final_loss:.4f}")

    # Write outputs.
    torch.save(backbone_state(model), "/output/backbone.pt")
    torch.save(adapter_state(model), "/output/adapter.pt")
    with open("/output/metadata.json", "w") as f:
        json.dump({"n_samples": n_samples, "train_loss": round(final_loss, 5),
                    "epochs": epochs}, f)

    print(f"Training complete: {n_samples} samples, {epochs} epochs → /output/")


# ------------------------------------------------------------------ main

def main() -> None:
    mode = os.environ.get("MELD_MODE", "inference").lower()
    print(f"MELD runtime starting (mode={mode})")

    config = _load_contract()
    df = pd.read_csv("/input/input.csv")
    print(f"Loaded {len(df)} rows from /input/input.csv")

    if mode == "train":
        run_train(df)
    else:
        run_inference(df, config)


if __name__ == "__main__":
    main()
