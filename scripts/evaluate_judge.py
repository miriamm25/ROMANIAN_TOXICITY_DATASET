#!/usr/bin/env python3
"""Evaluate the judge model (Qwen3-32B via Ollama) with different prompt strategies.

Strategies:
  zero-shot       — Generic prompt, no definition (already ran = 65%)
  definition      — Zero-shot with toxicity definition
  few-shot        — Definition + examples from train set
  cot             — Definition + structured chain-of-thought steps

Usage (on server where Ollama is running):
    uv run python scripts/evaluate_judge.py --strategy definition
    uv run python scripts/evaluate_judge.py --strategy few-shot
    uv run python scripts/evaluate_judge.py --strategy cot
    uv run python scripts/evaluate_judge.py --strategy zero-shot  # redo baseline
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from datasets import load_dataset

from torch_rar.training.utils import extract_classification, label_to_int
from torch_rar.evaluation.metrics import ClassificationMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Toxicity definition (shared by definition, few-shot, cot)
# ---------------------------------------------------------------------------
TOXICITY_DEFINITION = """\
Definiția toxicității pentru această sarcină:

TOXIC include:
- Insulte directe sau indirecte (porecle, diminutive pejorative, atacuri la persoană)
- Sarcasm și ironie agresivă îndreptate spre o persoană sau un grup
- Limbaj dismissiv sau denigrator (minimizarea, ridiculizarea cuiva)
- Agresiune pasivă (întrebări retorice menite să rănească, amenințări voalate)
- Derogare bazată pe aspect fizic, inteligență, competență sau etnie
- Expresii de dispreț sau ură, chiar dacă nu conțin cuvinte vulgare explicite
- Apeluri la violență, chiar și metaforice ("la pușcărie", "marș")

NON-TOXIC include:
- Critică constructivă sau opinii politice exprimate civilizat
- Dezacord sau nemulțumire fără atacuri personale
- Constatări factuale, chiar dacă sunt negative
- Expresii religioase, rugăciuni, sau îndemnuri morale
- Analize sau comentarii neutre despre situații politice"""


# ---------------------------------------------------------------------------
# Few-shot examples (from TRAIN set, not test)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """\
Exemple:

Text: "mucușor handicapatul"
Clasificare: TOXIC
(Poreclă denigrătoare + termen peiorativ referitor la dizabilitate)

Text: "diseară gâdea face infarct miocardic"
Clasificare: TOXIC
(Dorință de rău mascată ca predicție, agresivitate pasivă)

Text: "e delulu omul nostru... la momentul asta nici el nu mai stie ce vrea sa vorbeasca!"
Clasificare: TOXIC
(Atacuri la competență și ridicularizare a persoanei)

Text: "mulțumesc pentru știri în direct super digi 24 tv felicitări"
Clasificare: NON-TOXIC
(Apreciere sinceră, fără elemente agresive)

Text: "gadea nu mai minți omule te vede dumnezeu!"
Clasificare: NON-TOXIC
(Critică directă dar fără insulte, apel moral/religios)

Text: "dragi romani, vedeti cum psd si pnl isi bage iar joc de voi? nu sunt in stare sa apara la dezbatere, dar sa conduca o tara?"
Clasificare: NON-TOXIC
(Critică politică civilizată, fără atacuri la persoană)"""


# ---------------------------------------------------------------------------
# Chain-of-thought structured steps
# ---------------------------------------------------------------------------
COT_STEPS = """\
Analizează textul urmând acești pași:

Pasul 1: Identifică dacă textul conține insulte directe, porecle, sau termeni peiorativi.
Pasul 2: Verifică dacă există sarcasm sau ironie agresivă îndreptată spre o persoană.
Pasul 3: Caută limbaj dismissiv, denigrator, sau ridiculizant.
Pasul 4: Verifică dacă există agresiune pasivă, amenințări voalate, sau dorințe de rău.
Pasul 5: Evaluează contextul cultural românesc — expresii sau referințe care pot fi ofensatoare.
Pasul 6: Pe baza analizei de mai sus, oferă clasificarea finală."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_prompt(text: str, strategy: str) -> str:
    """Build the classification prompt based on strategy."""

    if strategy == "zero-shot":
        return (
            "Analizează următorul text românesc și stabilește dacă este "
            "toxic sau non-toxic.\n"
            "Explică raționamentul tău pas cu pas, apoi oferă "
            "clasificarea finală.\n\n"
            f"Text: {text}\n\n"
            "Răspunde cu raționamentul tău urmat de clasificarea finală: "
            "TOXIC sau NON-TOXIC."
        )

    elif strategy == "definition":
        return (
            "Ești un expert în detectarea toxicității în texte românești, "
            "în special în contextul discursului politic și social media.\n\n"
            f"{TOXICITY_DEFINITION}\n\n"
            f"Text de analizat: {text}\n\n"
            "Clasificare finală: TOXIC sau NON-TOXIC."
        )

    elif strategy == "few-shot":
        return (
            "Ești un expert în detectarea toxicității în texte românești, "
            "în special în contextul discursului politic și social media.\n\n"
            f"{TOXICITY_DEFINITION}\n\n"
            f"{FEW_SHOT_EXAMPLES}\n\n"
            "---\n\n"
            f"Text de analizat: {text}\n\n"
            "Clasificare finală: TOXIC sau NON-TOXIC."
        )

    elif strategy == "cot":
        return (
            "Ești un expert în detectarea toxicității în texte românești, "
            "în special în contextul discursului politic și social media.\n\n"
            f"{TOXICITY_DEFINITION}\n\n"
            f"{COT_STEPS}\n\n"
            f"Text de analizat: {text}\n\n"
            "Oferă analiza pas cu pas și clasificarea finală: TOXIC sau NON-TOXIC."
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate judge model with different prompt strategies"
    )
    parser.add_argument(
        "--strategy",
        choices=["zero-shot", "definition", "few-shot", "cot"],
        required=True,
        help="Prompt strategy to use",
    )
    parser.add_argument(
        "--judge-url",
        default="http://localhost:11434/v1",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen3:32b",
    )
    parser.add_argument(
        "--dataset",
        default="olimpia20/toxicity-dataset-ro-master",
    )
    parser.add_argument(
        "--split",
        default="test",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: ./output/eval_judge_{strategy}.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
    )
    return parser.parse_args()


def classify_with_judge(
    text: str,
    strategy: str,
    session: requests.Session,
    judge_url: str,
    judge_model: str,
    timeout: int,
) -> str:
    prompt = build_prompt(text, strategy)

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 2048,
    }

    response = session.post(
        f"{judge_url}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def main():
    args = parse_args()
    output_path = args.output or f"./output/eval_judge_{args.strategy}.json"

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    # Verify Ollama
    try:
        session.get(f"{args.judge_url}/models", timeout=5)
        print(f"Ollama OK: {args.judge_url}")
    except Exception as e:
        print(f"Cannot reach Ollama at {args.judge_url}: {e}")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Find columns
    text_col = label_col = None
    for col in dataset.column_names:
        if col.lower() in ("text", "content", "comment"):
            text_col = col
        if col.lower() in ("label", "labels", "toxic", "toxicity"):
            label_col = col
    if not text_col or not label_col:
        print(f"Cannot find text/label columns in {dataset.column_names}")
        sys.exit(1)

    strategy_name = {
        "zero-shot": "Zero-Shot Generic",
        "definition": "Zero-Shot + Definition",
        "few-shot": "Few-Shot (6 examples)",
        "cot": "Chain-of-Thought (structured)",
    }[args.strategy]

    print(f"Strategy: {strategy_name}")
    print(f"Evaluating {len(dataset)} samples with {args.judge_model}...")
    print("-" * 50)

    predictions = []
    labels = []
    detailed_results = []

    for i, sample in enumerate(dataset):
        text = str(sample[text_col]).strip()
        label = label_to_int(sample[label_col])

        if not text or label is None:
            continue

        try:
            completion = classify_with_judge(
                text, args.strategy, session,
                args.judge_url, args.judge_model, args.timeout,
            )
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            completion = ""

        # Strip <think> blocks before extracting classification
        clean = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
        predicted_class = extract_classification(clean or completion)
        predicted_label = (
            1 if predicted_class == "TOXIC"
            else 0 if predicted_class == "NON-TOXIC"
            else None
        )

        predictions.append(predicted_label)
        labels.append(label)

        detailed_results.append({
            "index": i,
            "text": text[:200],
            "label": label,
            "predicted": predicted_label,
            "predicted_class": predicted_class,
            "completion": completion[:2000],
            "correct": predicted_label == label,
        })

        if (i + 1) % 10 == 0:
            correct = sum(1 for r in detailed_results if r["correct"])
            print(
                f"  [{i+1}/{len(dataset)}] "
                f"Running accuracy: {correct / len(detailed_results):.1%}"
            )

    # Compute & save
    metrics = compute_metrics(predictions, labels)

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
        "model": args.judge_model,
        "strategy": args.strategy,
        "eval_type": "judge_as_classifier",
        "results": detailed_results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_evaluated = metrics.total + metrics.no_classification
    print(f"\nResults saved to {output_path}")
    print(f"\n{'='*55}")
    print(f"  {strategy_name} — {args.judge_model}")
    print(f"{'='*55}")
    print(f"  Accuracy:            {metrics.accuracy:.1%}")
    print(f"  F1 Score:            {metrics.f1:.1%}")
    print(f"  Precision:           {metrics.precision:.1%}")
    print(f"  Recall:              {metrics.recall:.1%}")
    print(f"  False Positive Rate: {metrics.false_positive_rate:.1%}")
    print(f"  False Negative Rate: {metrics.false_negative_rate:.1%}")
    print(f"  Classified:          {metrics.total}/{total_evaluated}")
    print(f"  No Classification:   {metrics.no_classification}/{total_evaluated}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
