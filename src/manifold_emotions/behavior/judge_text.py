"""Score generated text for (valence, arousal) using Claude.

Similar to ``vectors.judge`` but takes a passage of generated text
instead of an emotion concept word. Used to build the behavior manifold
M_y: each story in the corpus becomes a (valence, arousal) point;
we aggregate per emotion to get the behavior centroid.

We hit the API at higher concurrency than the concept-word judge
because passages are longer and total batch is larger (1500 stories
vs. 30 emotion words).
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


_TEXT_JUDGE_PROMPT = (
    "Read the following passage and rate its emotional content on two scales "
    "used in affective psychology (Russell circumplex):\n\n"
    "Valence: 1 (very negative / unpleasant) to 7 (very positive / pleasant).\n"
    "Arousal: 1 (very low energy / calm) to 7 (very high energy / activated).\n\n"
    "Rate the affective tone of the passage as a whole. Respond with exactly two "
    "numbers separated by a space in the form `<valence> <arousal>`. Decimals are "
    "fine. No other text.\n\n"
    "Passage:\n{text}"
)

_RESPONSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True, slots=True)
class TextRating:
    """Per-passage emotional rating."""

    # Stable identifier for the rated text (e.g. Story.request_id).
    text_id: str
    valence: float  # 1-7
    arousal: float  # 1-7


async def _judge_one(
    client: httpx.AsyncClient,
    config: Config,
    text_id: str,
    text: str,
    semaphore: asyncio.Semaphore,
) -> TextRating:
    payload = {
        "model": config.judge.model,
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": _TEXT_JUDGE_PROMPT.format(text=text)},
        ],
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
                timeout=45.0,
            )
        except httpx.HTTPError as exc:
            raise JudgeError(f"text judge HTTP error for {text_id!r}: {exc}") from exc

    if response.status_code >= 400:
        raise JudgeError(
            f"text judge HTTP {response.status_code} for {text_id!r}: {response.text}"
        )

    body = response.json()
    try:
        reply = body["content"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeError(
            f"unexpected text judge response shape for {text_id!r}: {body}"
        ) from exc

    match = _RESPONSE_RE.match(reply)
    if match is None:
        raise JudgeError(f"could not parse text judge response for {text_id!r}: {reply!r}")

    valence = float(match.group(1))
    arousal = float(match.group(2))
    if not (1.0 <= valence <= 7.0) or not (1.0 <= arousal <= 7.0):
        raise JudgeError(
            f"text judge for {text_id!r} out of [1,7] range: "
            f"valence={valence}, arousal={arousal}"
        )
    return TextRating(text_id=text_id, valence=valence, arousal=arousal)


async def judge_texts(
    config: Config,
    passages: list[tuple[str, str]],  # [(text_id, text), ...]
    cache_path: Path | None = None,
) -> dict[str, TextRating]:
    """Score every passage; cache resumable per ``text_id``.

    Idempotent: re-running merges with the on-disk cache and only
    hits the API for passages not already rated.
    """
    cache: dict[str, TextRating] = {}
    if cache_path is not None and cache_path.exists():
        for row in json.loads(cache_path.read_text()):
            cache[row["text_id"]] = TextRating(**row)

    missing = [(tid, text) for tid, text in passages if tid not in cache]
    log.info(
        "behavior.judge.start",
        total=len(passages),
        cached=len(passages) - len(missing),
        to_fetch=len(missing),
        concurrency=config.judge.concurrency,
    )
    if not missing:
        return {tid: cache[tid] for tid, _ in passages}

    semaphore = asyncio.Semaphore(config.judge.concurrency)
    errors: list[JudgeError] = []

    async with httpx.AsyncClient(http2=False) as client:

        async def run_one(text_id: str, text: str) -> TextRating | JudgeError:
            try:
                return await _judge_one(client, config, text_id, text, semaphore)
            except JudgeError as exc:
                return exc

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run_one(tid, text)) for tid, text in missing]

        for task in tasks:
            result = task.result()
            if isinstance(result, JudgeError):
                errors.append(result)
            else:
                cache[result.text_id] = result

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                [
                    {
                        "text_id": r.text_id,
                        "valence": r.valence,
                        "arousal": r.arousal,
                    }
                    for r in cache.values()
                ],
                indent=2,
            )
        )

    log.info(
        "behavior.judge.done",
        rated=len([tid for tid, _ in passages if tid in cache]),
        errors=len(errors),
        first_errors=[str(e) for e in errors[:3]],
    )
    return {tid: cache[tid] for tid, _ in passages if tid in cache}
