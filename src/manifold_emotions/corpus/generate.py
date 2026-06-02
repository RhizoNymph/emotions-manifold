"""Generate emotion-conditioned stories via the vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from pathlib import Path

import httpx
import structlog

from ..config import Config
from ..errors import CorpusError
from ..types import EmotionLabel, Story, Topic

log = structlog.get_logger(__name__)


_STORY_PROMPT = (
    "Write a short paragraph (about 120 words) about a character in this setting: "
    "{topic}. The character is experiencing {emotion}. Show the emotion vividly "
    "through the character's actions, thoughts, and physical reactions. Do not "
    'use the word "{emotion}" itself. Write only the paragraph, no preamble or commentary.'
)

# Stop strings sent to vLLM. The GGUF chat template appears malformed and
# the model hallucinates these special-token markers in lieu of a clean EOS.
# Halting on them prevents degenerate tail-ends in the output.
_STOP_SEQUENCES = [
    "<end_of_turn>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|channel>",
    "<turn|>",
    "<|channel|>",
]

# Reject stories that still contain template artifacts in the body.
_TEMPLATE_ARTIFACT_RE = re.compile(
    r"<\|[A-Za-z_]*\|?>|<[A-Za-z_]*_of_turn>|<turn\|?>|<\|channel"
)

# Reject stories that fall into degenerate token loops (same short fragment
# repeated many times).
_REPETITION_RE = re.compile(r"(.{1,15}?)\1{6,}", re.DOTALL)

_MIN_WORD_COUNT = 40


class InvalidStoryError(CorpusError):
    """Raised when a generated story fails content validation.

    Triggered by template artifacts, degenerate loops, or insufficient content.
    """


def _build_prompt(emotion: EmotionLabel, topic: Topic) -> str:
    return _STORY_PROMPT.format(emotion=emotion, topic=topic)


def _validate_story_text(emotion: EmotionLabel, story_idx: int, text: str) -> None:
    """Raise InvalidStoryError if the story shows signs of template leakage,
    degenerate repetition, or insufficient content.
    """
    if not text:
        raise InvalidStoryError(f"empty story for {emotion}/{story_idx}")

    if _TEMPLATE_ARTIFACT_RE.search(text):
        raise InvalidStoryError(
            f"template artifact in {emotion}/{story_idx}: "
            f"...{text[-120:]!r}"
        )

    if _REPETITION_RE.search(text):
        raise InvalidStoryError(
            f"degenerate repetition in {emotion}/{story_idx}: "
            f"...{text[-120:]!r}"
        )

    word_count = len(text.split())
    if word_count < _MIN_WORD_COUNT:
        raise InvalidStoryError(
            f"story too short ({word_count} words) for {emotion}/{story_idx}"
        )


async def _generate_one(
    client: httpx.AsyncClient,
    config: Config,
    emotion: EmotionLabel,
    topic: Topic,
    story_idx: int,
    semaphore: asyncio.Semaphore,
) -> Story:
    payload = {
        "model": config.model.hf_id,
        "messages": [{"role": "user", "content": _build_prompt(emotion, topic)}],
        "max_tokens": 512,
        "temperature": 0.9,
        "top_p": 0.95,
        "stop": _STOP_SEQUENCES,
    }

    headers = {}
    if config.vllm_server.api_key:
        headers["Authorization"] = f"Bearer {config.vllm_server.api_key}"

    async with semaphore:
        try:
            response = await client.post(
                f"{config.vllm_server.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            raise CorpusError(f"generation HTTP error for {emotion}/{story_idx}: {exc}") from exc

        if response.status_code >= 400:
            raise CorpusError(
                f"generation HTTP {response.status_code} for {emotion}/{story_idx} "
                f"at {response.request.url}: {response.text}"
            )

    body = response.json()
    try:
        text = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        raise CorpusError(f"malformed response body for {emotion}/{story_idx}: {body}") from exc

    # Strip any trailing stop-sequence-adjacent garbage that may have slipped in.
    for marker in _STOP_SEQUENCES:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()

    _validate_story_text(emotion, story_idx, text)
    return Story(emotion=emotion, topic=topic, story_idx=story_idx, text=text)


async def generate_corpus(
    config: Config,
    emotions: Iterable[EmotionLabel],
    topics: Iterable[Topic],
    stories_per_pair: int,
    output_path: Path,
    concurrency: int = 32,
) -> tuple[int, int]:
    """Generate the cross product of (emotions × topics × stories_per_pair) and write JSONL.

    Returns (written, rejected). Stories that fail validation (template
    artifacts, degenerate loops, too short) are logged and skipped.
    HTTP-level failures are also logged and skipped.
    Appends to output_path if it exists; caller is responsible for clearing
    it first if a fresh run is wanted.
    """
    emotions_list = list(emotions)
    topics_list = list(topics)
    total = len(emotions_list) * len(topics_list) * stories_per_pair

    log.info(
        "corpus.generate.start",
        num_emotions=len(emotions_list),
        num_topics=len(topics_list),
        stories_per_pair=stories_per_pair,
        total=total,
        concurrency=concurrency,
        output=str(output_path),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    written = 0
    invalid: list[InvalidStoryError] = []
    http_errors: list[CorpusError] = []
    results: list[Story | CorpusError] = []

    async with httpx.AsyncClient(http2=False) as client:
        async def run_one(
            emotion: EmotionLabel, topic: Topic, story_idx: int
        ) -> Story | CorpusError:
            try:
                return await _generate_one(client, config, emotion, topic, story_idx, semaphore)
            except CorpusError as exc:
                return exc

        async with asyncio.TaskGroup() as tg:
            tasks: list[asyncio.Task[Story | CorpusError]] = []
            for emotion in emotions_list:
                for topic in topics_list:
                    for story_idx in range(stories_per_pair):
                        tasks.append(tg.create_task(run_one(emotion, topic, story_idx)))

        for task in tasks:
            results.append(task.result())

    with output_path.open("a") as f:
        for result in results:
            match result:
                case Story() as story:
                    f.write(
                        json.dumps(
                            {
                                "emotion": story.emotion,
                                "topic": story.topic,
                                "story_idx": story.story_idx,
                                "text": story.text,
                            }
                        )
                        + "\n"
                    )
                    written += 1
                case InvalidStoryError() as exc:
                    invalid.append(exc)
                case CorpusError() as exc:
                    http_errors.append(exc)

    log.info(
        "corpus.generate.done",
        written=written,
        invalid=len(invalid),
        http_errors=len(http_errors),
        first_invalid=[str(e) for e in invalid[:3]],
        first_http_errors=[str(e) for e in http_errors[:3]],
    )
    return written, len(invalid) + len(http_errors)
