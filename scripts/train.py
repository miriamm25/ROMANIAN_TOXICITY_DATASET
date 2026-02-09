#!/usr/bin/env python3
"""GRPO training script for TORCH-RaR Romanian toxicity classification.

Usage:
    # Default settings (hybrid reward, 2 epochs)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py

    # Rule-based reward only (fastest)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --reward-mode rule_based

    # Custom settings
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
        --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --epochs 3 \
        --reward-mode hybrid \
        --lr 1e-5

GPU Layout:
    Terminal 1: Ollama is already running (judge on GPU 1)
    Terminal 2: CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO training with RaR rewards for Romanian toxicity"
    )
    parser.add_argument(
        "--base-model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        default="./output/augmented_dataset.parquet",
        help="Path to augmented dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["rule_based", "implicit", "hybrid"],
        default="hybrid",
        help="Reward computation mode",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of completions per prompt (GRPO group size)",
    )
    parser.add_argument(
        "--judge-url",
        default="http://localhost:11434/v1",
        help="Ollama judge API URL",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek-r1:70b",
        help="Ollama judge model name",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--eval-first",
        action="store_true",
        help="Run baseline evaluation before training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory (e.g., ./checkpoints/checkpoint-25)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Lazy imports (heavy dependencies)
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    from torch_rar.training.config import GRPOTrainingConfig
    from torch_rar.training.data_loader import load_training_dataset
    from torch_rar.training.reward_function import RaRRewardFunction

    # Build config
    config = GRPOTrainingConfig(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        reward_mode=args.reward_mode,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        judge_base_url=args.judge_url,
        judge_model=args.judge_model,
        use_lora=not args.no_lora,
    )

    # WandB
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    print("=" * 60)
    print("TORCH-RaR GRPO Training")
    print("=" * 60)
    print(f"  Base model:     {config.base_model}")
    print(f"  Reward mode:    {config.reward_mode}")
    print(f"  LoRA:           {config.use_lora}")
    print(f"  Epochs:         {config.num_train_epochs}")
    print(f"  LR:             {config.learning_rate}")
    print(f"  Batch size:     {config.per_device_train_batch_size}")
    print(f"  Group size:     {config.num_generations}")
    print(f"  Judge:          {config.judge_model} @ {config.judge_base_url}")
    print(f"  Dataset:        {config.dataset_path}")
    print(f"  Output:         {config.output_dir}")
    print(f"  Device:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # Optional baseline evaluation
    if args.eval_first:
        print("\nRunning baseline evaluation...")
        from torch_rar.evaluation.evaluator import Evaluator

        evaluator = Evaluator(config.base_model, config, is_baseline=True)
        baseline_metrics, _ = evaluator.evaluate(max_samples=100)
        print(f"Baseline accuracy: {baseline_metrics.accuracy:.1%}")
        print(f"Baseline F1:       {baseline_metrics.f1:.1%}")
        del evaluator
        torch.cuda.empty_cache()

    # Load dataset
    print("\nLoading augmented dataset...")
    dataset = load_training_dataset(config)
    print(f"  Samples: {len(dataset)}")

    # Create reward function
    print("\nInitializing reward function...")
    reward_fn = RaRRewardFunction(config)

    # Test judge connectivity (if using implicit/hybrid)
    if config.reward_mode in ("implicit", "hybrid"):
        print(f"  Testing judge at {config.judge_base_url}...")
        try:
            import requests
            resp = requests.get(
                f"{config.judge_base_url}/models",
                timeout=5,
            )
            models = resp.json()
            print(f"  Judge OK: {models}")
        except Exception as e:
            print(f"  WARNING: Judge not reachable: {e}")
            if config.reward_mode == "implicit":
                print("  Falling back to rule_based reward mode")
                config.reward_mode = "rule_based"
                reward_fn = RaRRewardFunction(config)

    # Configure TRL GRPO
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
        report_to="wandb" if not args.no_wandb else "none",
    )

    # LoRA config
    peft_config = None
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        print(f"\nLoRA config: r={config.lora_r}, alpha={config.lora_alpha}")

    # Create trainer
    print("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=config.base_model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Train
    print("\nStarting GRPO training...")
    print("-" * 60)
    trainer.train(resume_from_checkpoint=args.resume)

    # Save
    print("\nSaving model...")
    final_path = os.path.join(config.output_dir, "final")
    trainer.save_model(final_path)
    print(f"Model saved to {final_path}")

    # Post-training evaluation
    print("\nRunning post-training evaluation...")
    from torch_rar.evaluation.evaluator import Evaluator

    evaluator = Evaluator(final_path, config, is_baseline=False)
    trained_metrics, results = evaluator.evaluate(max_samples=100)
    evaluator.save_results(
        trained_metrics, results,
        os.path.join(config.output_dir, "eval_results.json"),
    )

    print(f"\nFinal accuracy: {trained_metrics.accuracy:.1%}")
    print(f"Final F1:       {trained_metrics.f1:.1%}")

    if args.eval_first:
        from torch_rar.evaluation.metrics import print_comparison
        print_comparison(baseline_metrics, trained_metrics)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
