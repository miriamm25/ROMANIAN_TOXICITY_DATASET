#!/usr/bin/env python3
"""Generate paraphrased versions of the augmented dataset via OpenRouter.

Creates 2 paraphrases per original text, preserving labels and toxicity levels.
Outputs a parquet file with the new samples.

Usage:
    export OPENROUTER_API_KEY="your-key"
    uv run python scripts/paraphrase.py
    uv run python scripts/paraphrase.py --limit 10  # test with 10 samples
"""

import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import urllib3

# Force IPv4 to avoid IPv6 connectivity issues
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    return [r for r in _orig_getaddrinfo(*args, **kwargs) if r[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_getaddrinfo

SYSTEM_PROMPT = (
    "Ești un expert lingvistic în limba română. Primești un text românesc "
    "și trebuie să creezi EXACT 2 parafraze diferite ale textului. "
    "Reguli stricte: "
    "1) Păstrează EXACT același sens și aceeași intenție. "
    "2) Păstrează același nivel de toxicitate/agresivitate. "
    "3) Folosește cuvinte și structuri diferite. "
    "4) Textul TREBUIE să fie în limba română corectă. "
    "5) Nu adăuga și nu elimina informații. "
    'Răspunde DOAR cu JSON: {"paraphrase_1": "...", "paraphrase_2": "..."}'
)


def paraphrase_text(text: str, api_key: str, retries: int = 3) -> dict | None:
    """Send a text to GPT-4o-mini for paraphrasing."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f'Text original: "{text}"'},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            result = resp.json()

            if "error" in result:
                print(f"  API error: {result['error']}")
                time.sleep(2 ** attempt)
                continue

            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response (handle markdown code blocks)
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean
                clean = clean.rsplit("```", 1)[0]
                clean = clean.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()

            data = json.loads(clean)
            if "paraphrase_1" in data and "paraphrase_2" in data:
                return data

            print(f"  Missing keys in response: {list(data.keys())}")

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt + 1}): {e}")
            print(f"  Raw response: {content[:200]}")
        except Exception as e:
            print(f"  Error (attempt {attempt + 1}): {e}")

        time.sleep(1)

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate paraphrases via OpenRouter")
    parser.add_argument(
        "--input", default="output/augmented_dataset.parquet", help="Input parquet"
    )
    parser.add_argument(
        "--output", default="output/paraphrased_samples.parquet", help="Output parquet"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    # Load dataset
    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} samples from {args.input}", flush=True)

    # Generate paraphrases
    results = []
    failed = 0
    total = len(df)

    def process_row(idx, row):
        text = str(row["text"])
        label = int(row["label"])
        data = paraphrase_text(text, api_key)
        return idx, text, label, data

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_row, i, row): i
            for i, row in df.iterrows()
        }

        done = 0
        for future in as_completed(futures):
            idx, text, label, data = future.result()
            done += 1

            if data is None:
                failed += 1
                print(f"  [{done}/{total}] FAILED: {text[:60]}...")
                continue

            # Add both paraphrases
            for key in ["paraphrase_1", "paraphrase_2"]:
                para_text = data[key].strip()
                if para_text and len(para_text) > 10:
                    results.append({
                        "text": para_text,
                        "label": label,
                        "source": "paraphrase",
                        "original_text": text,
                    })

            if done % 50 == 0:
                print(f"  [{done}/{total}] Generated {len(results)} paraphrases so far...", flush=True)

    print(f"\nDone: {len(results)} paraphrases from {total} originals ({failed} failed)")

    # Save
    out_df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

    # Quick stats
    toxic = out_df[out_df["label"] == 1]
    nontoxic = out_df[out_df["label"] == 0]
    print(f"  Toxic: {len(toxic)}, Non-toxic: {len(nontoxic)}")


if __name__ == "__main__":
    main()
