"""Load and validate project configuration from config.yaml + .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .errors import ConfigError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config.yaml"
_ENV_PATH = _REPO_ROOT / ".env"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hf_id: str
    quantization: str
    num_layers: int
    target_layer: int
    hook_point: str
    tensor_parallel_size: int


@dataclass(frozen=True, slots=True)
class VllmServerConfig:
    base_url: str
    api_key: str | None
    steering_api_key: str | None
    max_model_len: int
    max_num_seqs: int
    max_steering_configs: int
    enable_steering: bool


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    root: Path
    writer_threads: int
    queue_size: int


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    provider: str
    model: str
    api_key: str
    concurrency: int


@dataclass(frozen=True, slots=True)
class CorpusSliceConfig:
    num_emotions: int
    stories_per_emotion: int
    topics_per_emotion: int


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    shakedown: CorpusSliceConfig
    full: CorpusSliceConfig


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    skip_tokens_before: int
    pca_subspace_dim: int


@dataclass(frozen=True, slots=True)
class ManifoldConfig:
    primary: str
    secondary: str


@dataclass(frozen=True, slots=True)
class PathsConfig:
    emotion_words: Path
    story_corpus: Path
    emotion_vectors: Path
    manifold_h: Path
    manifold_y: Path


@dataclass(frozen=True, slots=True)
class Config:
    model: ModelConfig
    vllm_server: VllmServerConfig
    capture: CaptureConfig
    judge: JudgeConfig
    corpus: CorpusConfig
    extraction: ExtractionConfig
    manifold: ManifoldConfig
    paths: PathsConfig


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise ConfigError(f"environment variable {name} is required but unset")
    return value


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def load_config(config_path: Path | None = None) -> Config:
    """Read config.yaml and resolve env-var-referenced secrets via .env."""
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)

    path = config_path or _CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text())

    try:
        model = ModelConfig(**raw["model"])

        vllm_raw = raw["vllm_server"]
        # VLLM_BASE_URL env var overrides config.yaml; lets the same script run
        # against localhost or node1 without code edits.
        base_url_override = os.environ.get("VLLM_BASE_URL")
        vllm = VllmServerConfig(
            base_url=base_url_override or vllm_raw["base_url"],
            api_key=_optional_env(vllm_raw["api_key_env"]),
            steering_api_key=_optional_env(vllm_raw["steering_api_key_env"]),
            max_model_len=vllm_raw["max_model_len"],
            max_num_seqs=vllm_raw["max_num_seqs"],
            max_steering_configs=vllm_raw["max_steering_configs"],
            enable_steering=vllm_raw["enable_steering"],
        )

        capture_raw = raw["capture"]
        capture = CaptureConfig(
            root=_resolve_path(capture_raw["root"]),
            writer_threads=capture_raw["writer_threads"],
            queue_size=capture_raw["queue_size"],
        )

        judge_raw = raw["judge"]
        judge = JudgeConfig(
            provider=judge_raw["provider"],
            model=judge_raw["model"],
            api_key=_require_env(judge_raw["api_key_env"]),
            concurrency=judge_raw["concurrency"],
        )

        corpus = CorpusConfig(
            shakedown=CorpusSliceConfig(**raw["corpus"]["shakedown"]),
            full=CorpusSliceConfig(**raw["corpus"]["full"]),
        )

        extraction = ExtractionConfig(**raw["extraction"])
        manifold = ManifoldConfig(**raw["manifold"])

        paths_raw = raw["paths"]
        paths = PathsConfig(
            emotion_words=_resolve_path(paths_raw["emotion_words"]),
            story_corpus=_resolve_path(paths_raw["story_corpus"]),
            emotion_vectors=_resolve_path(paths_raw["emotion_vectors"]),
            manifold_h=_resolve_path(paths_raw["manifold_h"]),
            manifold_y=_resolve_path(paths_raw["manifold_y"]),
        )

    except KeyError as exc:
        raise ConfigError(f"missing key in config.yaml: {exc.args[0]}") from exc

    return Config(
        model=model,
        vllm_server=vllm,
        capture=capture,
        judge=judge,
        corpus=corpus,
        extraction=extraction,
        manifold=manifold,
        paths=paths,
    )


def load_emotion_words(path: Path) -> list[str]:
    """Read the emotion words file, one word per line."""
    if not path.exists():
        raise ConfigError(f"emotion words file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
