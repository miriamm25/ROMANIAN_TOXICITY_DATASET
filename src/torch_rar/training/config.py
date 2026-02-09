"""Training configuration for GRPO fine-tuning with RaR rewards."""

from dataclasses import dataclass, field


@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training with RaR reward signals.

    GPU Layout (2x H200 NVL):
        GPU 0: Training (model + LoRA + optimizer)
        GPU 1: Judge (DeepSeek-R1:70b via Ollama)
    """

    # Model
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # GRPO
    num_generations: int = 4
    learning_rate: float = 5e-6
    kl_coeff: float = 0.1

    # Reward
    reward_mode: str = "hybrid"  # "implicit", "rule_based", "hybrid"
    hybrid_alpha: float = 0.5  # weight for rule-based component in hybrid

    # Training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.9
    top_p: float = 0.95

    # Precision
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Judge (Ollama)
    judge_base_url: str = "http://localhost:11434/v1"
    judge_model: str = "deepseek-r1:70b"
    judge_max_concurrent: int = 2
    judge_timeout: int = 600

    # Paths
    output_dir: str = "./checkpoints"
    dataset_path: str = "./output/augmented_dataset.parquet"

    # Logging
    wandb_project: str = "torch-rar-grpo"
    logging_steps: int = 1
    save_steps: int = 25
    eval_steps: int = 50
