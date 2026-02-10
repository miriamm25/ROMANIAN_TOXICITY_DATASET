#!/usr/bin/env python3
"""Augment paraphrased samples with rubrics and rewards using the existing pipeline.

Runs each paraphrased text through the same RubricGenerator + RewardCalculator
that the original 400 samples went through, then merges everything into one dataset.

Usage:
    export OPENROUTER_API_KEY="your-key"
    uv run python scripts/augment_paraphrases.py
"""

import asyncio
import os
import socket
import sys

# Force IPv4 globally — must happen before any imports that open connections
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from torch_rar.config import load_settings
from torch_rar.data_loader import AugmentedSample, ToxicitySample
from torch_rar.pipeline import AugmentationPipeline


async def main():
    # Load settings (needs openrouter config)
    settings = load_settings()

    # Override to use openrouter
    from torch_rar.config import LLMProvider
    settings.llm_provider = LLMProvider.OPENROUTER

    pipeline = AugmentationPipeline(settings)

    # Load paraphrased samples (retry file if exists, otherwise full)
    retry_path = "output/paraphrases_to_retry.parquet"
    if os.path.exists(retry_path):
        para_df = pd.read_parquet(retry_path)
        print(f"Loaded {len(para_df)} paraphrases to RETRY from {retry_path}", flush=True)
    else:
        para_df = pd.read_parquet("output/paraphrased_samples.parquet")
        print(f"Loaded {len(para_df)} paraphrased samples", flush=True)

    # Process each paraphrase through the pipeline
    augmented = []
    failed = 0
    total = len(para_df)

    batch_size = settings.batch_size
    for i in range(0, total, batch_size):
        batch = para_df.iloc[i:i+batch_size]

        tasks = []
        for _, row in batch.iterrows():
            sample = ToxicitySample(
                id=f"para_{i}_{_}",
                text=str(row["text"]),
                label=int(row["label"]),
                metadata=None,
            )
            tasks.append(pipeline.process_sample(sample, reward_method="both"))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for j, result in enumerate(results):
            row = batch.iloc[j]
            if isinstance(result, Exception) or result is None:
                failed += 1
                continue
            augmented.append({
                "text": str(row["text"]),
                "label": int(row["label"]),
                "source": "paraphrase",
                "original_text": str(row["original_text"]),
                "rubrics": result.rubrics,
                "reward_explicit": result.reward_explicit,
                "reward_implicit": result.reward_implicit,
            })

        done = min(i + batch_size, total)
        print(f"  [{done}/{total}] Augmented {len(augmented)} paraphrases ({failed} failed)", flush=True)

    print(f"\nAugmentation complete: {len(augmented)} paraphrases augmented ({failed} failed)", flush=True)

    # Load existing augmented dataset (already has originals + previous paraphrases)
    existing_df = pd.read_parquet("output/augmented_dataset.parquet")
    print(f"Existing dataset: {len(existing_df)} samples", flush=True)

    # Merge new augmented paraphrases with existing
    para_aug_df = pd.DataFrame(augmented)
    merged = pd.concat([existing_df, para_aug_df], ignore_index=True)

    print(f"Merged dataset: {len(merged)} samples", flush=True)
    print(f"  Toxic: {sum(merged['label']==1)}, Non-toxic: {sum(merged['label']==0)}", flush=True)
    print(f"  Sources: {merged['source'].value_counts().to_dict()}", flush=True)

    merged.to_parquet("output/augmented_dataset.parquet", index=False)
    print(f"Saved to output/augmented_dataset.parquet", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
