#!/usr/bin/env bash
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class RouteLLMPreferenceHead(nn.Module):
    """3-Layer Neural Pairwise Preference Head predicting P(Local SLM satisfies query >= Cloud LLM)."""
    def __init__(self, input_dim=16, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def generate_synthetic_training_data(num_samples=2000):
    """Generates synthetic preference training tuples grounded in empirical difficulty & capabilities."""
    np.random.seed(42)
    X = []
    y = []

    for _ in range(num_samples):
        difficulty = np.random.uniform(0.0, 1.0)
        code_density = np.random.uniform(0.0, 1.0) if np.random.rand() > 0.5 else 0.0
        math_density = np.random.uniform(0.0, 1.0) if np.random.rand() > 0.5 else 0.0
        length_norm = np.random.uniform(0.1, 1.0)
        
        # One-hot intent: [coding, math, reasoning, chat]
        intent_idx = np.random.randint(0, 4)
        intent_vec = [0.0, 0.0, 0.0, 0.0]
        intent_vec[intent_idx] = 1.0

        # Local SLM capability: [reasoning, code, math, speed]
        local_cap = [0.85, 0.82, 0.84, 0.96]
        # Cloud LLM capability: [reasoning, code, math, speed]
        cloud_cap = [0.98, 0.97, 0.96, 0.60]

        features = [
            difficulty, code_density, math_density, length_norm,
            intent_vec[0], intent_vec[1], intent_vec[2], intent_vec[3],
            local_cap[0], local_cap[1], local_cap[2], local_cap[3],
            cloud_cap[0], cloud_cap[1], cloud_cap[2], cloud_cap[3]
        ]

        # Ground truth rule: Local SLM wins when difficulty is low or speed priority is high without complex reasoning
        prob_win = 1.0 - (difficulty * 0.70) - (code_density * 0.15) - (math_density * 0.15)
        label = 1.0 if prob_win >= 0.50 else 0.0

        X.append(features)
        y.append([label])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_and_export():
    print("=========================================================================")
    print("  TRAINING & EXPORTING ROUTELLM ONNX NEURAL PREFERENCE HEAD              ")
    print("=========================================================================")

    X, y = generate_synthetic_training_data(num_samples=3000)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = RouteLLMPreferenceHead(input_dim=16, hidden_dim=64)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("[*] Training PyTorch Preference Network across 15 epochs...")
    for epoch in range(1, 16):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_y)
            predicted_labels = (preds >= 0.5).float()
            correct += (predicted_labels == batch_y).sum().item()
            total += len(batch_y)

        if epoch % 3 == 0 or epoch == 15:
            avg_loss = epoch_loss / total
            acc = (correct / total) * 100
            print(f"   Epoch {epoch:2d}/15 - BCE Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    # Save PyTorch Checkpoint
    pt_path = MODELS_DIR / "preference_head.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"\n[+] Saved PyTorch model: {pt_path}")

    # Export to ONNX
    onnx_path = MODELS_DIR / "preference_head.onnx"
    dummy_input = torch.randn(1, 16, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14
    )
    print(f"[+] Exported ONNX binary: {onnx_path}")

    # Verify numerical equivalence with ONNX Runtime
    print("\n[*] Verifying numerical consistency between PyTorch and ONNX Runtime...")
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    test_x = torch.randn(50, 16, dtype=torch.float32)
    with torch.no_grad():
        pt_preds = model(test_x).numpy()

    ort_inputs = {"input": test_x.numpy()}
    ort_preds = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_preds - ort_preds))
    print(f"   Max absolute difference across 50 test vectors: {max_diff:.8f}")
    assert max_diff < 1e-5, "ONNX output deviation exceeds tolerance!"
    print("[✔] Numerical equivalence verified with tolerance < 1e-5!")
    print("=========================================================================\n")

if __name__ == "__main__":
    train_and_export()
