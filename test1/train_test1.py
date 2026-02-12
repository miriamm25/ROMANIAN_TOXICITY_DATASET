#!/usr/bin/env python3
"""Script de antrenament pentru test1 - Qwen2.5 pentru română"""

import os
import sys

# Adaugă root-ul proiectului la path pentru a accesa torch_rar
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from torch_rar.training.config import GRPOTrainingConfig
from torch_rar.training.data_loader import load_training_dataset
from torch_rar.training.reward_function import RaRRewardFunction

# Configurare pentru test1
test1_dir = os.path.dirname(os.path.abspath(__file__))

# Dezactivează WandB
os.environ["WANDB_DISABLED"] = "true"

# Configurare antrenament
config = GRPOTrainingConfig(
    base_model="Qwen/Qwen2.5-7B-Instruct",  # Model recomandat pentru română
    dataset_path=os.path.join(test1_dir, "output/augmented_dataset.parquet"),
    output_dir=os.path.join(test1_dir, "checkpoints"),
    reward_mode="rule_based",  # Mai rapid, nu necesită judge
    num_train_epochs=3,
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    num_generations=4,
    judge_base_url="http://localhost:11434/v1",
    judge_model="deepseek-r1:70b",
    use_lora=True,
)

# Verifică și afișează GPU-urile disponibile
num_gpus = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]

print("=" * 60)
print("TORCH-RaR GRPO Training - TEST1")
print("=" * 60)
print(f"  Base model:     {config.base_model}")
print(f"  Reward mode:    {config.reward_mode}")
print(f"  LoRA:           {config.use_lora}")
print(f"  Epochs:         {config.num_train_epochs}")
print(f"  LR:             {config.learning_rate}")
print(f"  Batch size:     {config.per_device_train_batch_size} (per device)")
print(f"  Group size:     {config.num_generations}")
print(f"  Dataset:        {config.dataset_path}")
print(f"  Output:         {config.output_dir}")
print(f"  GPU-uri:        {num_gpus} GPU-uri detectate")
for i, name in enumerate(gpu_names):
    print(f"    GPU {i}: {name}")
print(f"  Total batch size: {config.per_device_train_batch_size * num_gpus * config.gradient_accumulation_steps}")
print("=" * 60)

# Încarcă dataset
print("\nLoading augmented dataset...")
dataset = load_training_dataset(config)
print(f"  Samples: {len(dataset)}")

# Creează reward function
print("\nInitializing reward function...")
reward_fn = RaRRewardFunction(config)

# Configurează TRL GRPO cu suport multi-GPU
training_args = GRPOConfig(
    output_dir=config.output_dir,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    max_grad_norm=config.max_grad_norm,
    warmup_ratio=config.warmup_ratio,
    bf16=config.bf16,
    gradient_checkpointing=config.gradient_checkpointing,
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    num_generations=config.num_generations,
    max_completion_length=config.max_new_tokens,
    temperature=config.temperature,
    report_to="none",
    # Configurare multi-GPU
    dataloader_num_workers=4,  # Workers pentru data loading
    ddp_find_unused_parameters=False,  # Optimizare pentru multi-GPU
)

# LoRA config
peft_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.lora_target_modules,
    task_type="CAUSAL_LM",
)
print(f"\nLoRA config: r={config.lora_r}, alpha={config.lora_alpha}")

# Creează trainer
print("\nInitializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=config.base_model,
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

# Antrenament
print("\nStarting GRPO training...")
print("-" * 60)
trainer.train()

# Salvează modelul
print("\nSaving model...")
final_path = os.path.join(config.output_dir, "final")
trainer.save_model(final_path)
print(f"Model saved to {final_path}")

# Evaluare post-antrenament
print("\nRunning post-training evaluation...")
from torch_rar.evaluation.evaluator import Evaluator

evaluator = Evaluator(final_path, config, is_baseline=False)
trained_metrics, results = evaluator.evaluate(max_samples=100)
evaluator.save_results(
    trained_metrics, results,
    os.path.join(config.output_dir, "eval_results.json"),
)

print(f"\n{'='*60}")
print("REZULTATE FINALE")
print(f"{'='*60}")
print(f"Accuracy:            {trained_metrics.accuracy:.1%}")
print(f"F1 Score:            {trained_metrics.f1:.1%}")
print(f"Precision:           {trained_metrics.precision:.1%}")
print(f"Recall:              {trained_metrics.recall:.1%}")
print(f"False Positive Rate: {trained_metrics.false_positive_rate:.1%}")
print(f"False Negative Rate: {trained_metrics.false_negative_rate:.1%}")
print(f"{'='*60}")

print("\nTraining complete!")
print(f"Model final salvat în: {final_path}")
print(f"Rezultate evaluare: {os.path.join(config.output_dir, 'eval_results.json')}")

