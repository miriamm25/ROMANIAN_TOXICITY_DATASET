"""Configuration settings for TORCH-RaR loaded from settings.yaml."""

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from torch_rar.constants import (
    APIEndpoints,
    CacheDefaults,
    CategoryDefaults,
    LoggingDefaults,
    ModelDefaults,
    DatasetDefaults,
    ProcessingLimits,
    PromptDefaults,
)


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""

    level: str = Field(
        default=LoggingDefaults.LEVEL,
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    directory: str = Field(
        default=LoggingDefaults.DIRECTORY,
        description="Directory for log files",
    )
    json_format: bool = Field(
        default=True,
        description="Whether to output JSON to files",
    )
    rotation: str = Field(
        default=LoggingDefaults.ROTATION,
        description="When to rotate log files",
    )
    retention: str = Field(
        default=LoggingDefaults.RETENTION,
        description="How long to keep old logs",
    )


class CacheConfig(BaseModel):
    """Configuration for rubric caching."""

    enabled: bool = Field(
        default=True,
        description="Whether caching is enabled",
    )
    directory: str = Field(
        default=CacheDefaults.DIRECTORY,
        description="Directory for cache files",
    )
    ttl_seconds: int = Field(
        default=CacheDefaults.TTL_SECONDS,
        description="Cache entry time-to-live in seconds",
    )
    size_limit_gb: float = Field(
        default=CacheDefaults.SIZE_LIMIT_GB,
        description="Maximum cache size in GB",
    )


class DomainConfig(BaseModel):
    """Configuration for a specific prompt template domain."""

    tasks: list[str] = Field(
        default_factory=list,
        description="List of task descriptions for the system prompt",
    )
    context: Optional[str] = Field(
        default=None,
        description="Important context for the domain",
    )


class PromptTemplatesConfig(BaseModel):
    """Configuration for prompt template system."""

    directory: str = Field(
        default=PromptDefaults.DIRECTORY,
        description="Path to the templates directory",
    )
    default_domain: str = Field(
        default=PromptDefaults.DEFAULT_DOMAIN,
        description="Default domain to use when not specified",
    )
    domains: dict[str, DomainConfig] = Field(
        default_factory=dict,
        description="Domain-specific configurations",
    )


class RubricWeights:
    """Standard weights for TORCH-RaR rubric categories.

    These weights follow the RaR paper Section 5.1 methodology for
    Romanian toxicity detection. Weights are used in explicit aggregation:

        r(x, ŷ) = Σⱼ(wⱼ · cⱼ(x, ŷ)) / Σⱼ|wⱼ|

    Essential criteria have highest weights (critical indicators).
    Important criteria have moderate weights (contextual factors).
    Pitfall criteria have negative weights (penalties for errors).

    Note: Individual rubric weights are now defined in torch_rar.constants
    (EssentialWeights, ImportantWeights, PitfallWeights). This class is
    kept for backward compatibility.
    """

    # Import from constants for backward compatibility
    from torch_rar.constants import EssentialWeights, ImportantWeights, PitfallWeights

    # Essential criteria weights (E1-E4)
    E1_CORRECT_LABEL = EssentialWeights.E1_CORRECT_LABEL
    E2_PERSONAL_ATTACK = EssentialWeights.E2_PERSONAL_ATTACK
    E3_THREAT_DETECTION = EssentialWeights.E3_THREAT_DETECTION
    E4_GROUP_HATRED = EssentialWeights.E4_GROUP_HATRED

    # Important criteria weights (I1-I4)
    I1_CONTEXTUAL = ImportantWeights.I1_CONTEXTUAL
    I2_EMOTIONAL = ImportantWeights.I2_EMOTIONAL
    I3_SARCASM = ImportantWeights.I3_SARCASM
    I4_POLITICAL = ImportantWeights.I4_POLITICAL

    # Pitfall criteria weights (P1-P3) - negative for penalties
    P1_FALSE_POSITIVE = PitfallWeights.P1_FALSE_POSITIVE
    P2_FALSE_NEGATIVE = PitfallWeights.P2_FALSE_NEGATIVE
    P3_CONTEXT_FREE = PitfallWeights.P3_CONTEXT_FREE

    # Default category weights for prompt-based rubrics
    ESSENTIAL_DEFAULT = CategoryDefaults.ESSENTIAL
    IMPORTANT_DEFAULT = CategoryDefaults.IMPORTANT
    OPTIONAL_DEFAULT = CategoryDefaults.OPTIONAL
    PITFALL_DEFAULT = CategoryDefaults.PITFALL


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENROUTER = "openrouter"
    VLLM = "vllm"
    LITELLM_PROXY = "litellm_proxy"


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in values.

    Supports ${VAR_NAME} syntax for environment variable substitution.
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} pattern
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.environ.get(var_name, "")
            value = value.replace(f"${{{var_name}}}", env_value)
        return value if value else None
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _find_settings_file() -> Path:
    """Find the settings.yaml file.

    Searches in order:
    1. Current working directory (config/settings.yaml)
    2. Project root (where pyproject.toml is, config/settings.yaml)
    3. Package directory (config/settings.yaml)
    """
    # Current directory
    cwd_path = Path.cwd() / "config" / "settings.yaml"
    if cwd_path.exists():
        return cwd_path

    # Look for project root by finding pyproject.toml
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            settings_path = parent / "config" / "settings.yaml"
            if settings_path.exists():
                return settings_path

    # Fallback to package directory
    package_dir = Path(__file__).parent.parent.parent
    settings_path = package_dir / "config" / "settings.yaml"
    if settings_path.exists():
        return settings_path

    raise FileNotFoundError(
        "config/settings.yaml not found. Please create one from config/settings.yaml.example"
    )


class Settings(BaseModel):
    """Application settings loaded from settings.yaml."""

    # LLM Provider Settings
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENROUTER,
        description="LLM provider to use (openrouter, vllm, litellm_proxy)",
    )

    # OpenRouter Settings
    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key",
    )
    openrouter_base_url: str = Field(
        default=APIEndpoints.OPENROUTER,
        description="OpenRouter API base URL",
    )

    # vLLM Settings (for local Docker deployment)
    vllm_base_url: str = Field(
        default=APIEndpoints.VLLM_LOCAL,
        description="vLLM server base URL",
    )
    vllm_model_name: str = Field(
        default=ModelDefaults.VLLM_MODEL,
        description="Model name for vLLM server",
    )

    # LiteLLM Proxy Settings
    litellm_proxy_url: str = Field(
        default=APIEndpoints.LITELLM_PROXY,
        description="LiteLLM proxy server URL",
    )
    litellm_api_key: Optional[str] = Field(
        default=None,
        description="LiteLLM proxy API key (if configured)",
    )

    # Model Selection
    rubric_generator_model: str = Field(
        default=ModelDefaults.RUBRIC_GENERATOR,
        description="Model for generating rubrics",
    )
    judge_model: str = Field(
        default=ModelDefaults.JUDGE,
        description="Model for evaluating responses (LLM-as-Judge)",
    )

    # Dataset Settings
    dataset_name: str = Field(
        default=DatasetDefaults.NAME,
        description="HuggingFace dataset to augment",
    )
    dataset_split: str = Field(
        default=DatasetDefaults.SPLIT,
        description="Dataset split to use",
    )
    output_dir: str = Field(
        default=DatasetDefaults.OUTPUT_DIR,
        description="Directory for output files",
    )

    # Rubric Settings
    min_rubric_items: int = Field(
        default=ProcessingLimits.MIN_RUBRIC_ITEMS,
        description="Minimum rubric items per prompt",
    )
    max_rubric_items: int = Field(
        default=ProcessingLimits.MAX_RUBRIC_ITEMS,
        description="Maximum rubric items per prompt",
    )

    # Processing Settings
    batch_size: int = Field(
        default=ProcessingLimits.DEFAULT_BATCH_SIZE,
        description="Batch size for processing",
    )
    max_concurrent_requests: int = Field(
        default=ProcessingLimits.MAX_CONCURRENT_REQUESTS,
        description="Max concurrent LLM API requests",
    )
    request_timeout: int = Field(
        default=ProcessingLimits.REQUEST_TIMEOUT_SECONDS,
        description="API request timeout in seconds",
    )
    max_retries: int = Field(
        default=ProcessingLimits.MAX_RETRIES,
        description="Max retries for failed requests",
    )
    max_tokens: int = Field(
        default=ProcessingLimits.MAX_TOKENS,
        description="Maximum tokens in LLM response",
    )

    # Reward Weights (for explicit aggregation)
    weight_essential: float = Field(
        default=CategoryDefaults.ESSENTIAL,
        description="Weight for Essential criteria",
    )
    weight_important: float = Field(
        default=CategoryDefaults.IMPORTANT,
        description="Weight for Important criteria",
    )
    weight_optional: float = Field(
        default=CategoryDefaults.OPTIONAL,
        description="Weight for Optional criteria",
    )
    weight_pitfall: float = Field(
        default=abs(CategoryDefaults.PITFALL),
        description="Weight for Pitfall criteria (absolute value)",
    )

    # Prompt Templates Configuration
    prompt_templates: PromptTemplatesConfig = Field(
        default_factory=PromptTemplatesConfig,
        description="Configuration for prompt template system",
    )

    # Logging Configuration
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Configuration for logging",
    )

    # Cache Configuration
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Configuration for rubric caching",
    )

    # Progress Display
    show_progress: bool = Field(
        default=True,
        description="Whether to show progress bars",
    )

    model_config = {"extra": "ignore"}

    def get_litellm_model_name(self, model_type: str = "judge") -> str:
        """Get the appropriate model name for LiteLLM based on provider."""
        if model_type == "rubric":
            model = self.rubric_generator_model
        else:
            model = self.judge_model

        if self.llm_provider == LLMProvider.OPENROUTER:
            # OpenRouter models are prefixed with "openrouter/"
            if not model.startswith("openrouter/"):
                return f"openrouter/{model}"
            return model
        elif self.llm_provider == LLMProvider.VLLM:
            # vLLM uses the model name directly
            return f"hosted_vllm/{self.vllm_model_name}"
        else:
            # LiteLLM proxy - use model name as-is
            return model

    def get_api_base(self) -> Optional[str]:
        """Get the API base URL based on provider."""
        if self.llm_provider == LLMProvider.OPENROUTER:
            return self.openrouter_base_url
        elif self.llm_provider == LLMProvider.VLLM:
            return self.vllm_base_url
        elif self.llm_provider == LLMProvider.LITELLM_PROXY:
            return self.litellm_proxy_url
        return None

    def get_api_key(self) -> Optional[str]:
        """Get the API key based on provider."""
        if self.llm_provider == LLMProvider.OPENROUTER:
            return self.openrouter_api_key
        elif self.llm_provider == LLMProvider.LITELLM_PROXY:
            return self.litellm_api_key
        return None


def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Load settings from YAML file.

    Args:
        config_path: Path to settings.yaml. If None, searches default locations.

    Returns:
        Settings object with loaded configuration.
    """
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Settings file not found: {path}")
    else:
        path = _find_settings_file()

    with open(path) as f:
        raw_config = yaml.safe_load(f) or {}

    # Expand environment variables
    config = _expand_env_vars(raw_config)

    return Settings(**config)


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings(
    config_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> Settings:
    """Get the global settings instance.

    Args:
        config_path: Optional path to settings.yaml for first load.
        force_reload: If True, reload settings even if already loaded.

    Returns:
        Settings instance.
    """
    global _settings
    if _settings is None or force_reload:
        _settings = load_settings(config_path)
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance.

    Useful for testing to ensure clean state between tests.
    After calling this, the next call to get_settings() will
    reload settings from the configuration file.
    """
    global _settings
    _settings = None
