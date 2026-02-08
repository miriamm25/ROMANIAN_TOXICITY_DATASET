"""Model evaluator for Romanian toxicity classification.

Runs a trained (or baseline) model on the test split and collects
predictions for metrics computation.
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from torch_rar.training.config import GRPOTrainingConfig
from torch_rar.training.utils import extract_classification, format_prompt, label_to_int
from torch_rar.evaluation.metrics import ClassificationMetrics, compute_metrics


class Evaluator:
    """Evaluate a model on Romanian toxicity classification.

    Loads either a base model or a LoRA-adapted model and runs inference
    on the test split of the toxicity dataset.
    """

    def __init__(
        self,
        model_path: str,
        config: GRPOTrainingConfig | None = None,
        is_baseline: bool = False,
    ):
        """Initialize evaluator.

        Args:
            model_path: Path to model or LoRA adapter checkpoint.
            config: Training config (for base_model name if loading adapter).
            is_baseline: If True, load model_path as a standalone model
                (no LoRA adapter).
        """
        self.config = config or GRPOTrainingConfig()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if is_baseline:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                device_map={"": self.device},
            )
        else:
            # Load base model + LoRA adapter
            base_model = self.config.base_model
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                device_map={"": self.device},
            )
            self.model = PeftModel.from_pretrained(self.model, model_path)

        self.model.eval()

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(
        self,
        dataset_name: str = "olimpia20/toxicity-dataset-ro-master",
        split: str = "test",
        max_samples: int | None = None,
    ) -> tuple[ClassificationMetrics, list[dict]]:
        """Run evaluation on test split.

        Args:
            dataset_name: HuggingFace dataset name.
            split: Dataset split to evaluate on.
            max_samples: Limit number of samples (for quick testing).

        Returns:
            Tuple of (metrics, detailed_results).
        """
        # Load test data
        dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Find text and label columns
        text_col = self._find_column(dataset.column_names, ["text", "content", "comment"])
        label_col = self._find_column(dataset.column_names, ["label", "labels", "toxic", "toxicity"])

        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not find text/label columns in {dataset.column_names}"
            )

        predictions = []
        labels = []
        detailed_results = []

        for i, sample in enumerate(dataset):
            text = str(sample[text_col]).strip()
            label = label_to_int(sample[label_col])

            if not text or label is None:
                continue

            # Generate prediction
            completion = self._generate(text)
            predicted_class = extract_classification(completion)
            predicted_label = (
                1 if predicted_class == "TOXIC"
                else 0 if predicted_class == "NON-TOXIC"
                else None
            )

            predictions.append(predicted_label)
            labels.append(label)

            detailed_results.append({
                "index": i,
                "text": text[:200],  # Truncate for readability
                "label": label,
                "predicted": predicted_label,
                "predicted_class": predicted_class,
                "completion": completion[:500],
                "correct": predicted_label == label,
            })

            if (i + 1) % 10 == 0:
                current_correct = sum(
                    1 for r in detailed_results if r["correct"]
                )
                print(
                    f"  [{i + 1}/{len(dataset)}] "
                    f"Running accuracy: {current_correct / len(detailed_results):.1%}"
                )

        metrics = compute_metrics(predictions, labels)
        return metrics, detailed_results

    def _generate(self, text: str) -> str:
        """Generate a classification response for a Romanian text.

        Uses greedy decoding (temperature=0.1) for deterministic evaluation.

        Args:
            text: Romanian text to classify.

        Returns:
            Model-generated completion string.
        """
        messages = format_prompt(text)

        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = messages[0]["content"]

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens (not the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _find_column(columns: list[str], candidates: list[str]) -> str | None:
        """Find matching column name (case-insensitive)."""
        cols_lower = {c.lower(): c for c in columns}
        for candidate in candidates:
            if candidate.lower() in cols_lower:
                return cols_lower[candidate.lower()]
        return None

    def save_results(
        self,
        metrics: ClassificationMetrics,
        detailed_results: list[dict],
        output_path: str,
    ) -> None:
        """Save evaluation results to JSON.

        Args:
            metrics: Computed metrics.
            detailed_results: Per-sample results.
            output_path: Path to save JSON file.
        """
        output = {
            "metrics": {
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "false_positive_rate": metrics.false_positive_rate,
                "false_negative_rate": metrics.false_negative_rate,
                "total": metrics.total,
                "correct": metrics.correct,
                "no_classification": metrics.no_classification,
            },
            "results": detailed_results,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")
