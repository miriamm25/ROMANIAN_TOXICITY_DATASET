"""RaR reward function bridge for TRL's GRPOTrainer.

This module bridges the TORCH-RaR reward calculation system into TRL's
expected reward function signature. It supports three reward modes:

1. rule_based: Fast binary check — does the classification match ground truth?
2. implicit: LLM judge (DeepSeek-R1:70b via Ollama) gives holistic 1-10 score
3. hybrid: Weighted combination of rule_based + implicit

The hybrid mode is recommended: rule_based provides a strong gradient signal
for correctness, while implicit captures reasoning quality via RaR rubrics.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from torch_rar.training.config import GRPOTrainingConfig
from torch_rar.training.utils import extract_classification


class RaRRewardFunction:
    """Reward function bridging RaR methodology to TRL's GRPOTrainer.

    TRL calls this with generated completions and dataset kwargs.
    We compute rewards using rule-based classification checking
    and/or LLM judge evaluation via Ollama.

    GPU Layout:
        - Training model runs on GPU 0
        - Ollama judge (DeepSeek-R1:70b) runs on GPU 1
        - Communication via HTTP (localhost:11434)
    """

    __name__ = "rar_reward"

    def __init__(self, config: GRPOTrainingConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._executor = ThreadPoolExecutor(
            max_workers=config.judge_max_concurrent
        )

    def __call__(self, completions, **kwargs) -> list[float]:
        """Compute rewards for a batch of completions.

        This is called by TRL's GRPOTrainer after generating completions.

        Args:
            completions: Model-generated completions. Format depends on dataset:
                - Standard format: list[str]
                - Conversational format: list[list[dict]] (message dicts)
            **kwargs: Additional dataset columns including:
                - label: Ground truth labels (list[int])
                - rubrics: Pre-generated rubrics (list[list[dict]])
                - original_text: Original Romanian texts (list[str])

        Returns:
            List of float rewards, one per completion.
        """
        # Handle both conversational and standard formats from TRL
        texts = []
        for c in completions:
            if isinstance(c, list):  # conversational: [{"role": "assistant", "content": "..."}]
                texts.append(c[0]["content"] if c else "")
            elif isinstance(c, str):
                texts.append(c)
            else:
                texts.append(str(c))

        labels = kwargs.get("label", [None] * len(texts))
        original_texts = kwargs.get("original_text", [""] * len(texts))
        rubrics = kwargs.get("rubrics", [[] for _ in texts])

        rewards = []
        for completion, label, text, rubric_list in zip(
            texts, labels, original_texts, rubrics
        ):
            reward = self._compute_single_reward(
                completion, label, text, rubric_list
            )
            rewards.append(reward)

        return rewards

    def _compute_single_reward(
        self,
        completion: str,
        label: int | None,
        original_text: str,
        rubrics: list[dict],
    ) -> float:
        """Compute reward for a single completion.

        Args:
            completion: Model-generated text (reasoning + classification).
            label: Ground truth label (0 or 1).
            original_text: The original Romanian text being classified.
            rubrics: Pre-generated rubric criteria.

        Returns:
            Float reward value.
        """
        if self.config.reward_mode == "rule_based":
            return self._rule_based_reward(completion, label)
        elif self.config.reward_mode == "implicit":
            return self._implicit_reward(completion, original_text, rubrics)
        else:  # hybrid
            rule = self._rule_based_reward(completion, label)
            implicit = self._implicit_reward(
                completion, original_text, rubrics
            )
            alpha = self.config.hybrid_alpha
            return alpha * rule + (1 - alpha) * implicit

    def _rule_based_reward(self, completion: str, label: int | None) -> float:
        """Fast reward based on classification correctness.

        Extracts TOXIC/NON-TOXIC from completion and compares with
        ground truth label. Also gives partial credit for having
        a clear classification even if wrong.

        Args:
            completion: Model output text.
            label: Ground truth (0=non-toxic, 1=toxic).

        Returns:
            1.0 if correct, 0.1 if wrong but has classification, 0.0 if no classification.
        """
        if label is None:
            return 0.5  # No ground truth available

        predicted = extract_classification(completion)
        if predicted is None:
            return 0.0  # No classification found — penalize

        predicted_label = 1 if predicted == "TOXIC" else 0
        if predicted_label == label:
            return 1.0  # Correct classification
        else:
            return 0.1  # Wrong but at least produced a classification

    def _implicit_reward(
        self,
        completion: str,
        original_text: str,
        rubrics: list[dict],
    ) -> float:
        """LLM judge reward using implicit aggregation.

        Sends all rubrics + the completion to DeepSeek-R1:70b via Ollama
        for a holistic 1-10 score, then normalizes to [0, 1].

        Args:
            completion: Model output to evaluate.
            original_text: Original Romanian text.
            rubrics: List of rubric criterion dicts.

        Returns:
            Normalized reward [0, 1].
        """
        # Format rubrics by category
        essential = []
        important = []
        pitfall = []
        for r in rubrics:
            cat = r.get("category", "").lower()
            line = f"- [{r.get('rubric_id', '?')}] {r.get('title', '')}: {r.get('description', '')}"
            if cat == "essential":
                essential.append(line)
            elif cat == "important":
                important.append(line)
            elif cat == "pitfall":
                pitfall.append(line)

        essential_text = "\n".join(essential) or "None specified"
        important_text = "\n".join(important) or "None specified"
        pitfall_text = "\n".join(pitfall) or "None specified"

        system_prompt = (
            "You are an expert evaluator for toxicity detection in Romanian "
            "political discourse.\n\n"
            "Score the following model response on a scale of 1-10 based on "
            "how well it analyzes the text and reaches a correct classification.\n\n"
            "Scoring Guide:\n"
            "- 1-2: Completely wrong classification with poor reasoning\n"
            "- 3-4: Wrong classification but shows some understanding\n"
            "- 5-6: Borderline - partially correct reasoning\n"
            "- 7-8: Correct classification with reasonable analysis\n"
            "- 9-10: Correct classification with thorough, nuanced analysis\n\n"
            'Respond with ONLY a JSON object: {"rating": <1-10>}'
        )

        user_prompt = (
            f"**Original Text (Romanian):**\n{original_text}\n\n"
            f"**Model Response:**\n{completion}\n\n"
            f"**Evaluation Criteria:**\n\n"
            f"ESSENTIAL:\n{essential_text}\n\n"
            f"IMPORTANT:\n{important_text}\n\n"
            f"PITFALL:\n{pitfall_text}\n\n"
            f"Rate this response (1-10):"
        )

        try:
            rating = self._call_judge(system_prompt, user_prompt)
            # Normalize from 1-10 to 0-1
            return max(0.0, min(1.0, (rating - 1) / 9.0))
        except Exception:
            return 0.5  # Fallback on judge failure

    def _call_judge(self, system_prompt: str, user_prompt: str) -> int:
        """Call Ollama judge via OpenAI-compatible API.

        Args:
            system_prompt: System message for the judge.
            user_prompt: User message with content to evaluate.

        Returns:
            Integer rating 1-10.
        """
        url = f"{self.config.judge_base_url}/chat/completions"

        payload = {
            "model": self.config.judge_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 64,
        }

        response = self.session.post(
            url, json=payload, timeout=self.config.judge_timeout
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return self._parse_rating(content)

    @staticmethod
    def _parse_rating(text: str) -> int:
        """Extract integer rating from judge response.

        Tries JSON parsing first, then regex fallback.

        Args:
            text: Judge response text.

        Returns:
            Integer rating clamped to 1-10.
        """
        # Try JSON
        try:
            # Handle potential <think>...</think> tags from R1
            clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            # Find JSON object
            json_match = re.search(r"\{[^}]+\}", clean)
            if json_match:
                data = json.loads(json_match.group())
                rating = int(data.get("rating", 5))
                return max(1, min(10, rating))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Regex fallback: find a number 1-10
        numbers = re.findall(r"\b(\d{1,2})\b", text)
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= 10:
                return num

        return 5  # Default middle score
