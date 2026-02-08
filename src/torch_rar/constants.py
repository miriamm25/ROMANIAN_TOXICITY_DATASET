"""Default constants for TORCH-RaR configuration.

These constants provide default values for all configuration fields in Settings.
They are organized into logical groups matching the configuration structure.

Values are traced from config/settings.yaml defaults and the TORCH-RaR
methodology (RaR paper Section 5.1 for rubric weights).
"""


class APIEndpoints:
    """Default API endpoint URLs for LLM providers."""

    OPENROUTER = "https://openrouter.ai/api/v1"
    VLLM_LOCAL = "http://localhost:8000/v1"
    LITELLM_PROXY = "http://localhost:4000"


class ModelDefaults:
    """Default model names for different tasks."""

    RUBRIC_GENERATOR = "openrouter/openai/gpt-4o"
    JUDGE = "openrouter/openai/gpt-4o-mini"
    VLLM_MODEL = "/models/qwen2.5-7b-instruct-q3_k_m.gguf"


class DatasetDefaults:
    """Default dataset configuration values."""

    NAME = "olimpia20/toxicity-dataset-ro-master"
    SPLIT = "train"
    OUTPUT_DIR = "./output"


class ProcessingLimits:
    """Default processing and API limits."""

    MIN_RUBRIC_ITEMS = 7
    MAX_RUBRIC_ITEMS = 20
    DEFAULT_BATCH_SIZE = 10
    MAX_CONCURRENT_REQUESTS = 5
    REQUEST_TIMEOUT_SECONDS = 120
    MAX_RETRIES = 3
    MAX_TOKENS = 8192


class CategoryDefaults:
    """Default weights for rubric categories (used for dynamically generated rubrics)."""

    ESSENTIAL = 1.0
    IMPORTANT = 0.7
    OPTIONAL = 0.3
    PITFALL = -0.6


class LoggingDefaults:
    """Default logging configuration values."""

    LEVEL = "INFO"
    DIRECTORY = "logs"
    ROTATION = "10 MB"
    RETENTION = "7 days"


class CacheDefaults:
    """Default cache configuration values."""

    DIRECTORY = ".cache/rubrics"
    TTL_SECONDS = 2592000  # 30 days
    SIZE_LIMIT_GB = 1.0


class PromptDefaults:
    """Default prompt template configuration values."""

    DIRECTORY = "prompts/"
    DEFAULT_DOMAIN = "toxicity"


class EssentialWeights:
    """Weights for Essential rubric criteria (E1-E4).

    From TORCH-RaR guide Section 5.1:
    - E1 (Correct Label): 1.0 - most critical
    - E2 (Personal Attack): 0.95
    - E3 (Threat Detection): 0.90
    - E4 (Group Hatred): 0.90
    """

    E1_CORRECT_LABEL = 1.0
    E2_PERSONAL_ATTACK = 0.95
    E3_THREAT_DETECTION = 0.90
    E4_GROUP_HATRED = 0.90


class ImportantWeights:
    """Weights for Important rubric criteria (I1-I4).

    From TORCH-RaR guide Section 5.1:
    - I1 (Contextual): 0.70
    - I2 (Emotional): 0.65
    - I3 (Sarcasm): 0.60
    - I4 (Political): 0.60
    """

    I1_CONTEXTUAL = 0.70
    I2_EMOTIONAL = 0.65
    I3_SARCASM = 0.60
    I4_POLITICAL = 0.60


class PitfallWeights:
    """Weights for Pitfall rubric criteria (P1-P3) - negative for penalties.

    From TORCH-RaR guide Section 5.1:
    - P1 (False Positive): -0.60
    - P2 (False Negative): -0.65
    - P3 (Context-Free): -0.50
    """

    P1_FALSE_POSITIVE = -0.60
    P2_FALSE_NEGATIVE = -0.65
    P3_CONTEXT_FREE = -0.50
