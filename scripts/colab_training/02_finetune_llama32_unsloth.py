#!/usr/bin/env python3
"""
RouteMem AI Gateway — Tier B: Llama-3.2-3B Unsloth QLoRA Fine-Tuning Script
Designed to run on Google Colab (Free T4 GPU) or any NVIDIA GPU machine.
Fine-tunes Llama-3.2-3B in 4-bit NF4 using Unsloth and exports GGUF directly to Hugging Face Hub.
"""
import os
import torch

def run_finetuning():
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError:
        print("[!] Unsloth or dependencies missing. In Colab, run:")
        print("    !pip install --no-deps 'xformers<0.0.28' 'trl<0.9.0' peft accelerate bitsandbytes")
        print("    !pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        return

    max_seq_length = 2048
    dtype = None # Auto detection
    load_in_4bit = True

    print("[1/4] Loading unsloth/Llama-3.2-3B-Instruct in 4-bit NF4...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-3B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    print("[2/4] Applying LoRA Adapters (r=16, alpha=32)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("[3/4] Loading High-Density Reasoning Dataset (Bespoke-Stratos-17k)...")
    dataset = load_dataset("Bespoke-Labs/Bespoke-Stratos-17k", split="train[:3000]")

    def formatting_prompts_func(examples):
        instructions = examples["system"] if "system" in examples else ["You are a RouteMem specialist reasoning assistant."] * len(examples["conversations"])
        texts = []
        for conv in examples["conversations"]:
            text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("[4/4] Starting SFTTrainer on GPU...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 150,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 15,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    trainer.train()
    print("✔ Fine-tuning completed!")

    # Export instructions
    print("\n[Optional] To save locally or push to Hugging Face Hub:")
    print("  model.save_pretrained_merged('routemem-llama3.2-3b-merged', tokenizer, save_method='merged_16bit')")
    print("  model.push_to_hub_gguf('YOUR_USERNAME/routemem-llama3.2-3b', tokenizer, quantization_method='q4_k_m')")

if __name__ == "__main__":
    run_finetuning()
