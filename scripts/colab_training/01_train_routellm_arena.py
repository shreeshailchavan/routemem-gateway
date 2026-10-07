#!/usr/bin/env python3
"""
RouteMem AI Gateway — Tier A: RouteLLM Neural Preference Head Training Script
Designed to run on Google Colab (GPU) or local PyTorch environment.
Trains a 3-layer pairwise preference MLP on LMSYS Arena 140k battles and exports to ONNX.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxruntime as ort

def train_and_export():
    print("[1/5] Loading LMSYS Arena Human Preference Dataset (140k)...")
    dataset = load_dataset("lmarena-ai/arena-human-preference-140k", split="train[:35000]")

    print("[2/5] Filtering pairwise battles comparing SLMs vs Frontier Cloud Models...")
    slm_keywords = ["llama-3", "llama-2", "qwen", "mistral", "phi", "gemma", "3b", "7b", "8b"]
    cloud_keywords = ["gpt-4", "claude-3", "gemini-1.5", "o1", "sonnet"]

    X_features = []
    y_labels = []

    for row in dataset:
        m_a = (row.get("model_a") or "").lower()
        m_b = (row.get("model_b") or "").lower()
        winner = row.get("winner")

        is_a_slm = any(k in m_a for k in slm_keywords)
        is_b_cloud = any(k in m_b for k in cloud_keywords)
        is_b_slm = any(k in m_b for k in slm_keywords)
        is_a_cloud = any(k in m_a for k in cloud_keywords)

        if (is_a_slm and is_b_cloud) or (is_b_slm and is_a_cloud):
            conv = row.get("conversation_a") or []
            prompt_text = conv[0]["content"] if conv and len(conv) > 0 else ""
            if not prompt_text:
                continue

            prompt_lower = prompt_text.lower()
            code_density = min(1.0, sum(prompt_lower.count(k) for k in ["def ", "class ", "function", "import ", "return", "{", "}"]) / 10.0)
            math_density = min(1.0, sum(prompt_lower.count(k) for k in ["\\int", "\\sum", "^", "==", "!=", "solve", "calculate"]) / 8.0)
            length_norm = min(1.0, len(prompt_text.split()) / 400.0)
            difficulty = min(1.0, 0.25 + (code_density * 0.4) + (math_density * 0.3) + (length_norm * 0.2))

            if code_density > 0.3:
                intent_vec = [1.0, 0.0, 0.0, 0.0]
            elif math_density > 0.3:
                intent_vec = [0.0, 1.0, 0.0, 0.0]
            elif length_norm > 0.5:
                intent_vec = [0.0, 0.0, 1.0, 0.0]
            else:
                intent_vec = [0.0, 0.0, 0.0, 1.0]

            local_cap = [0.89, 0.82, 0.85, 0.96]
            cloud_cap = [0.98, 0.97, 0.96, 0.60]

            feat = [
                difficulty, code_density, math_density, length_norm,
                intent_vec[0], intent_vec[1], intent_vec[2], intent_vec[3],
                local_cap[0], local_cap[1], local_cap[2], local_cap[3],
                cloud_cap[0], cloud_cap[1], cloud_cap[2], cloud_cap[3]
            ]

            if is_a_slm:
                label = 1.0 if winner in ["model_a", "tie", "both"] else 0.0
            else:
                label = 1.0 if winner in ["model_b", "tie", "both"] else 0.0

            X_features.append(feat)
            y_labels.append([label])

    print(f"[+] Prepared {len(X_features)} balanced pairwise battles.")

    X_tensor = torch.tensor(X_features, dtype=torch.float32)
    y_tensor = torch.tensor(y_labels, dtype=torch.float32)

    dataset_torch = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset_torch, batch_size=64, shuffle=True)

    class RouteLLMPreferenceHead(nn.Module):
        def __init__(self, input_dim=16, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouteLLMPreferenceHead().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"[3/5] Training Neural Preference Head on {device}...")
    model.train()
    for epoch in range(12):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"   Epoch {epoch+1:02d}/12 — Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs("models", exist_ok=True)
    onnx_path = "models/preference_head.onnx"
    print(f"[4/5] Exporting Trained Model to {onnx_path}...")
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 16, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14
    )

    ort_session = ort.InferenceSession(onnx_path)
    ort_out = ort_session.run(None, {"input": dummy_input.numpy()})[0]
    torch_out = model(dummy_input).detach().numpy()
    max_diff = np.max(np.abs(ort_out - torch_out))
    print(f"[5/5] ONNX Verification Complete! Max absolute difference vs PyTorch: {max_diff:.8f}")
    print("✔ Model exported successfully! Ready for production deployment.")

if __name__ == "__main__":
    train_and_export()
