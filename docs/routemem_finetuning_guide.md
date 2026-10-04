# RouteMem AI Gateway — Model Fine-Tuning & Profiling Developer Guide

This guide provides complete implementation specifications, data pipelines, hyperparameter recipes, and execution code for fine-tuning, training, and profiling custom models within the RouteMem AI Gateway architecture.

---

## 1. Subsystems Needing Fine-Tuning or Profiling

RouteMem incorporates four distinct trainable or profilable machine learning components to optimize query routing, difficulty estimation, and capability mapping:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          ROUTEMEM TRAINABLE & PROFILABLE SUBSYSTEMS                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. DeBERTa-v3 ONNX Difficulty & Intent Profiler                                       │
│    • Small (<100M param) encoder fine-tuned for Task Intent + Difficulty Score D in [0,1]│
│    • Quantized to ONNX INT8 for <3 ms execution latency on CPU/GPU                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. RouteLLM Preference Classifier Head                                                 │
│    • Pairwise preference scoring head trained on Chatbot Arena / Nectar comparison logs │
│    • Predicts P(Weak SLM satisfies query Quality >= Strong Cloud LLM)                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. UniRoute / ICL-Router Capability Profiler                                          │
│    • Zero-retraining continuous vector mapper over K-cluster error probabilities       │
│    • Generates K-dimensional capability vector Psi(h) via 500 anchor probing queries   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Router-R1 RL Policy Fine-Tuning (Optional Advanced)                                │
│    • PPO/GRPO reinforcement learning fine-tuning on base LLM (e.g. Qwen2.5-3B)         │
│    • Multi-step reasoning with rule-based rewards for format, outcome, & token cost    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Fine-Tuning Subsystem 1: DeBERTa-v3 Difficulty & Intent Profiler

### Goal & SLA Requirements
The Profiler inspects incoming user prompts *before* model routing to compute two values in **<3 ms**:
1. **Task Intent Category**: `coding`, `math`, `reasoning`, `chat`, `extraction`.
2. **Difficulty Score ($D \in [0.0, 1.0]$)**: Calibrated estimation of query complexity.

### Dataset Preparation Pipeline
Combine open-source datasets with difficulty labels:
* **MixInstruct / LMSYS Crowd**: 100k instruction prompts labeled by length, AST code complexity, and multi-hop depth.
* **GSM8K & HumanEval**: Hard reasoning & code benchmarks ($D \ge 0.8$).
* **Alpaca Cleaned**: Low-to-medium difficulty instructions ($D \in [0.1, 0.4]$).

Data schema (`dataset.jsonl`):
```json
{"text": "Write a Python function to implement quicksort with in-place partitioning.", "intent": "coding", "difficulty": 0.65}
{"text": "What is the capital of France?", "intent": "chat", "difficulty": 0.05}
```

### PyTorch Training Setup (HuggingFace Transformers)
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

training_args = TrainingArguments(
    output_dir="./deberta_profiler_checkpoints",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch"
)
```

### ONNX Export & INT8 Quantization (<3 ms Latency)
To achieve the **<3 ms latency SLA**, export the trained PyTorch weights to INT8 ONNX:
```bash
# Export PyTorch checkpoint to ONNX
python -m transformers.onnx --model=deberta_profiler_checkpoints --feature=sequence-classification onnx/

# Quantize ONNX model to INT8
python -m onnxruntime.quantization.quantize --input onnx/model.onnx --output onnx/model_quant.onnx
```

---

## 3. Fine-Tuning Subsystem 2: RouteLLM Preference Classifier

### Mathematical Loss Formulation
Trains a scoring head $f_\theta(x)$ predicting whether a low-cost model $M_{\text{weak}}$ produces a response equivalent in human preference to a high-cost model $M_{\text{strong}}$:
$$\mathcal{L}_{\text{pairwise}}(\theta) = -\sum_{(x, y_{\text{weak}}, y_{\text{strong}})} \left[ y \log \sigma(f_\theta(x)) + (1-y) \log (1 - \sigma(f_\theta(x))) \right]$$
where $y = 1$ if $M_{\text{weak}}$ score $\ge M_{\text{strong}}$ score according to human or LLM-as-a-judge votes.

### Training Datasets
* **Chatbot Arena 80k Battle Dataset**: Pairwise human comparison votes.
* **Nectar Dataset**: 120k GPT-4 rated pairs across candidate outputs.

---

## 4. Subsystem 3: UniRoute & ICL-Router Continuous Capability Profiling

### Zero-Retraining Model Onboarding Protocol
When onboarded to the gateway pool, **a new candidate LLM does not require router retraining**. Instead, profile its capability vector $\Psi(h_{\text{new}})$ over $K$ task clusters using a fixed anchor set:

1. Execute the **500 Anchor Probing Query Set** (sampled from MMLU-Pro, HumanEval, GSM8K) against the new model.
2. Compute the cluster-level error probability vector:
   $$\Psi_k(h_{\text{new}}) = \frac{1}{|C_k|} \sum_{(x,y) \in C_k} \mathbb{I}[h_{\text{new}}(x) \neq y]$$
3. Register $\Psi(h_{\text{new}})$ in the RouteMem active routing registry (`models_capability_matrix.json`).

---

## 5. Subsystem 4: Router-R1 Reinforcement Learning Fine-Tuning

For multi-step reasoning models, fine-tune a small base model (e.g. `Qwen/Qwen2.5-3B-Instruct`) using **GRPO (Group Relative Policy Optimization)**:

### Hierarchical Rule-Based Reward Function
$$R(x, y) = R_{\text{format}}(y) + R_{\text{outcome}}(y) - \gamma \cdot \text{Cost}(y)$$
* $R_{\text{format}}$: $+1.0$ if the model emits valid `<think>...</think><route>model_name</route>` XML tags.
* $R_{\text{outcome}}$: $+2.0$ if the routed model produces correct ground-truth results.
* $\text{Cost}(y)$: Penalty proportional to routed API token expense.

---

## 6. Summary of Hardware & Compute Requirements

| Training Subsystem | GPU Hardware Required | Compute Time | Output Artifact |
| :--- | :--- | :--- | :--- |
| **DeBERTa Profiler** | 1x NVIDIA RTX 4090 / L4 (24GB) | ~2 hours | `profiler_model.onnx` (45MB) |
| **RouteLLM Head** | 1x NVIDIA T4 / Colab Free | ~45 minutes | `preference_head.pt` (12MB) |
| **UniRoute Profiling** | 500 API calls per new LLM | ~3 minutes | `capability_vector.json` (<1KB) |
| **Router-R1 RL (3B)** | 1x or 2x NVIDIA A100 (80GB) | ~6-10 hours | LoRA Weights / Quantized GGUF |
