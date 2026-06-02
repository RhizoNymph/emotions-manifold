"""LLM judge: rate emotion words on (valence, arousal) using Claude.

Per Anthropic's paper, valence + arousal are the primary affective
dimensions (the "circumplex"). They used Claude to score each of the
171 emotion words on 1-7 scales, then validated against established
human PAD norms (r=0.92 for valence, r=0.90 for arousal).

We do the same so we can correlate our extracted vectors' principal
components with valence/arousal axes. Cached to disk so repeated runs
don't re-hit the API.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog

from ..config import Config
from ..errors import JudgeError

log = structlog.get_logger(__name__)


_JUDGE_PROMPT = (
    "Rate the emotion concept '{emotion}' on two scales used in affective psychology:\n\n"
    "Valence: 1 (very negative / unpleasant) to 7 (very positive / pleasant).\n"
    "Arousal: 1 (very low energy / calm) to 7 (very high energy / activated).\n\n"
    "Use the standard Russell circumplex / PAD interpretation. Respond with "
    "exactly two numbers separated by a space, in the format `<valence> <arousal>`. "
    "Decimals are fine. No other text."
)


# Strict two-number response: "5.5 6", "3 2.5", etc.
_RESPONSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True, slots=True)
class EmotionRating:
    emotion: str
    valence: float  # 1-7
    arousal: float  # 1-7


async def _judge_one(
    client: httpx.AsyncClient,
    config: Config,
    emotion: str,
    semaphore: asyncio.Semaphore,
) -> EmotionRating:
    payload = {
        "model": config.judge.model,
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": _JUDGE_PROMPT.format(emotion=emotion)}],
    }
    headers = {
        "x-api-key": config.judge.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with semaphore:
        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
        except httpx.HTTPError as exc:
            raise JudgeError(f"judge HTTP error for {emotion!r}: {exc}") from exc

    if response.status_code >= 400:
        raise JudgeError(
            f"judge HTTP {response.status_code} for {emotion!r}: {response.text}"
        )

    body = response.json()
    try:
        text = body["content"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeError(f"unexpected judge response shape for {emotion!r}: {body}") from exc

    match = _RESPONSE_RE.match(text)
    if match is None:
        raise JudgeError(f"could not parse judge response for {emotion!r}: {text!r}")

    valence = float(match.group(1))
    arousal = float(match.group(2))
    if not (1.0 <= valence <= 7.0) or not (1.0 <= arousal <= 7.0):
        raise JudgeError(
            f"judge response for {emotion!r} out of [1,7] range: "
            f"valence={valence}, arousal={arousal}"
        )
    return EmotionRating(emotion=emotion, valence=valence, arousal=arousal)


async def judge_emotions(
    config: Config,
    emotions: list[str],
    cache_path: Path | None = None,
) -> dict[str, EmotionRating]:
    """Score every emotion in ``emotions`` via the configured judge.

    If ``cache_path`` is given and exists, ratings already present in
    the cache are loaded directly and only missing emotions hit the API.
    The cache is rewritten with the merged ratings after each batch
    completes so partial runs are recoverable.
    """
    cache: dict[str, EmotionRating] = {}
    if cache_path is not None and cache_path.exists():
        for row in json.loads(cache_path.read_text()):
            cache[row["emotion"]] = EmotionRating(**row)

    missing = [e for e in emotions if e not in cache]
    log.info(
        "judge.start",
        total=len(emotions),
        cached=len(emotions) - len(missing),
        to_fetch=len(missing),
        concurrency=config.judge.concurrency,
    )
    if not missing:
        return {e: cache[e] for e in emotions}

    semaphore = asyncio.Semaphore(config.judge.concurrency)
    errors: list[JudgeError] = []

    async with httpx.AsyncClient(http2=False) as client:

        async def run_one(emotion: str) -> EmotionRating | JudgeError:
            try:
                return await _judge_one(client, config, emotion, semaphore)
            except JudgeError as exc:
                return exc

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run_one(e)) for e in missing]

        for task in tasks:
            result = task.result()
            if isinstance(result, JudgeError):
                errors.append(result)
            else:
                cache[result.emotion] = result

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                [
                    {"emotion": r.emotion, "valence": r.valence, "arousal": r.arousal}
                    for r in cache.values()
                ],
                indent=2,
            )
        )

    log.info(
        "judge.done",
        rated=len([e for e in emotions if e in cache]),
        errors=len(errors),
        first_errors=[str(e) for e in errors[:3]],
    )

    # Return only the requested emotions in their original order.
    return {e: cache[e] for e in emotions if e in cache}
