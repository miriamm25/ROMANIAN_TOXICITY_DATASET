"""Utility functions for GRPO training."""

import re


def extract_classification(text: str) -> str | None:
    """Extract TOXIC/NON-TOXIC classification from model output.

    Handles various output formats:
    - Direct: "TOXIC" or "NON-TOXIC"
    - After reasoning: "... therefore the text is TOXIC"
    - Romanian: "TOXIC" / "NON-TOXIC" (same in Romanian context)
    - With tags: "<think>...</think> NON-TOXIC"

    Args:
        text: Model-generated completion text.

    Returns:
        "TOXIC", "NON-TOXIC", or None if no classification found.
    """
    if not text:
        return None

    # Normalize whitespace
    text_clean = text.strip()

    # Strip <think>...</think> reasoning blocks — Qwen3 produces these
    # and the reasoning often mentions "toxic" which causes false matches
    text_clean = re.sub(r'<think>.*?</think>', '', text_clean, flags=re.DOTALL).strip()

    # If stripping removed everything, fall back to original text
    if not text_clean:
        text_clean = text.strip()

    # Check for NON-TOXIC first (it contains "TOXIC" as substring)
    non_toxic_patterns = [
        r"\bNON[-_\s]?TOXIC\b",
        r"\bnon[-_\s]?toxic\b",
        r"\bne[-_\s]?toxic\b",  # Romanian
    ]
    toxic_patterns = [
        r"\bTOXIC\b",
        r"\btoxic\b",
    ]

    # Search from the end of the text (classification usually comes last)
    # Split into last 200 chars for final classification
    tail = text_clean[-200:] if len(text_clean) > 200 else text_clean

    for pattern in non_toxic_patterns:
        if re.search(pattern, tail, re.IGNORECASE):
            return "NON-TOXIC"

    for pattern in toxic_patterns:
        if re.search(pattern, tail, re.IGNORECASE):
            # Make sure it's not part of "NON-TOXIC"
            match = re.search(pattern, tail, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 4)
                prefix = tail[start:match.start()].lower()
                if "non" not in prefix:
                    return "TOXIC"

    # Fallback: search entire text
    for pattern in non_toxic_patterns:
        if re.search(pattern, text_clean, re.IGNORECASE):
            return "NON-TOXIC"

    for pattern in toxic_patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 4)
            prefix = text_clean[start:match.start()].lower()
            if "non" not in prefix:
                return "TOXIC"

    return None


def format_prompt(text: str) -> list[dict[str, str]]:
    """Format a Romanian text into a chat prompt for toxicity classification.

    Uses a reasoning-first prompt that encourages diverse completions
    for GRPO (different reasoning paths → different conclusions →
    meaningful reward variance).

    Args:
        text: Romanian text to classify.

    Returns:
        List of message dicts in chat format.
    """
    return [
        {
            "role": "user",
            "content": (
                "Analizează următorul text românesc și stabilește dacă este "
                "toxic sau non-toxic.\n"
                "Explică raționamentul tău pas cu pas, apoi oferă "
                "clasificarea finală.\n\n"
                f"Text: {text}\n\n"
                "Răspunde cu raționamentul tău urmat de clasificarea finală: "
                "TOXIC sau NON-TOXIC."
            ),
        }
    ]


def label_to_int(label) -> int | None:
    """Convert various label formats to integer 0/1.

    Handles: int, float, str ("toxic"/"non-toxic"/"0"/"1"), bool.

    Args:
        label: Label value in any supported format.

    Returns:
        0 (non-toxic) or 1 (toxic), or None if unparseable.
    """
    if label is None:
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return int(label)
    if isinstance(label, str):
        label_lower = label.strip().lower()
        if label_lower in ("1", "toxic", "true"):
            return 1
        if label_lower in ("0", "non-toxic", "nontoxic", "false"):
            return 0
    return None
