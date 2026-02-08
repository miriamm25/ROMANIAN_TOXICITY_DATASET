"""Dataset preparation for GRPO training with TRL.

Loads the augmented parquet from Phase 2 and formats it for TRL's
GRPOTrainer, which expects a dataset with a "prompt" column.
"""

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset

from torch_rar.training.config import GRPOTrainingConfig
from torch_rar.training.utils import format_prompt, label_to_int


def load_training_dataset(config: GRPOTrainingConfig) -> Dataset:
    """Load augmented dataset and prepare for TRL GRPOTrainer.

    Reads the augmented parquet file produced by Phase 2, formats prompts,
    and returns a HuggingFace Dataset with the columns TRL expects.

    Required columns in parquet:
        - text: Romanian text content
        - label: Ground truth (0/1 or toxic/non-toxic)

    Optional columns (passed to reward function as kwargs):
        - rubrics: Pre-generated rubric list (JSON)
        - reward_explicit: Pre-computed explicit reward
        - reward_implicit: Pre-computed implicit reward

    Args:
        config: Training configuration with dataset_path.

    Returns:
        HuggingFace Dataset with 'prompt' column and metadata columns.

    Raises:
        FileNotFoundError: If dataset_path doesn't exist.
        ValueError: If required columns are missing.
    """
    path = Path(config.dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Augmented dataset not found: {path}")

    # Load based on file extension
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Find text column
    text_col = _find_column(df, ["text", "content", "comment", "sentence"])
    if text_col is None:
        raise ValueError(
            f"No text column found. Available columns: {list(df.columns)}"
        )

    # Find label column
    label_col = _find_column(df, ["label", "labels", "toxic", "toxicity", "is_toxic"])
    if label_col is None:
        raise ValueError(
            f"No label column found. Available columns: {list(df.columns)}"
        )

    # Build training records
    records = []
    skipped = 0
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        label = label_to_int(row[label_col])

        if not text or label is None:
            skipped += 1
            continue

        record = {
            "prompt": format_prompt(text),
            "original_text": text,
            "label": label,
        }

        # Include rubrics if available (for reward function)
        if "rubrics" in df.columns:
            rubrics = row["rubrics"]
            if isinstance(rubrics, str):
                try:
                    rubrics = json.loads(rubrics)
                except json.JSONDecodeError:
                    rubrics = []
            record["rubrics"] = rubrics if rubrics else []

        records.append(record)

    if not records:
        raise ValueError(
            f"No valid training samples found. Total rows: {len(df)}, "
            f"skipped: {skipped}"
        )

    if skipped > 0:
        print(f"Warning: Skipped {skipped}/{len(df)} samples (missing text or label)")

    print(f"Loaded {len(records)} training samples from {path}")

    return Dataset.from_list(records)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name (case-insensitive).

    Args:
        df: DataFrame to search.
        candidates: List of possible column names.

    Returns:
        Matching column name or None.
    """
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None
