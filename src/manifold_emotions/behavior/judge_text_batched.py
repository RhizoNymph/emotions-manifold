"""Batched-API drop-in for ``judge_text.judge_texts``.

Uses Anthropic's Message Batches API (``/v1/messages/batches``) instead of
sequential ``/v1/messages`` calls. 50% cost reduction; wall time of
0–60 min per batch (vs ~0.5 s per individual call). Use when the
total per-pair latency hit is acceptable — for a 900-call pair the
batch returns in well under the 60-min cap and the math comes out
ahead in cost.

Same ``TextRating`` schema, same ``cache_path`` JSON shape, same
return type as ``judge_text.judge_texts`` — swap one import to A/B
test the two paths against each other.

Wire shapes (raw HTTP — no SDK dependency):

  POST  /v1/messages/batches
    body: {"requests": [{"custom_id": "...", "params": {...}}, ...]}
    → {"id": "msgbatch_...", "processing_status": "in_progress", ...}

  GET   /v1/messages/batches/{batch_id}
    → {"processing_status": "in_progress" | "canceling" | "ended",
       "request_counts": {"processing": N, "succeeded": N, "errored": N,
                           "canceled": N, "expired": N},
       "results_url": "https://..." (only when ended)}

  GET   {results_url}
    → JSONL stream, each line:
      {"custom_id": "...", "result": {"type": "succeeded"|"errored"|...,
                                       "message"|"error": {...}}}

Limits:
  - 100,000 requests / 256 MB per batch (chunk above that)
  - Most batches complete in <1 h; max 24 h
  - Results retained 29 days
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog

from ..config import Config
from ..errors import JudgeError
from .judge_text import TextRating, _RESPONSE_RE, _TEXT_JUDGE_PROMPT

log = structlog.get_logger(__name__)


# ---- Batch limits & polling tunables -----------------------------------

# Conservative — actual cap is 100,000 / 256 MB. Smaller chunks recover
# from transient errors faster and let us start fetching the first chunk's
# results while later chunks are still processing.
MAX_REQUESTS_PER_BATCH = 25_000

# Backoff schedule for status polling: tight near submission, loose later.
# Most batches finish in 10–30 min; the schedule below caps any wait at
# ~60 min before we start exponential backoff.
POLL_INTERVALS_SEC = (10, 15, 30, 60, 60, 60, 60, 60, 90, 120, 120, 180)


# ---- HTTP helpers ------------------------------------------------------


def _headers(config: Config) -> dict[str, str]:
    return {
        "x-api-key": config.judge.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _build_request(custom_id: str, text: str, model: str) -> dict:
    """One entry in the batch ``requests`` array.

    Note: ``custom_id`` here is the API-safe ID (regex
    ``^[a-zA-Z0-9_-]{1,64}$``), not the project's text_id (which can
    contain spaces in multi-word labels). The caller is responsible
    for maintaining the safe_id ↔ text_id mapping.
    """
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 32,
            "temperature": 0.0,
            "messages": [
                {"role": "user", "content": _TEXT_JUDGE_PROMPT.format(text=text)},
            ],
        },
    }


async def _submit_batch(
    client: httpx.AsyncClient,
    config: Config,
    requests: list[dict],
) -> str:
    """Submit a single batch; return the batch ID."""
    response = await client.post(
        "https://api.anthropic.com/v1/messages/batches",
        json={"requests": requests},
        headers=_headers(config),
        timeout=120.0,
    )
    if response.status_code >= 400:
        raise JudgeError(
            f"batches submit HTTP {response.status_code}: {response.text[:400]}"
        )
    body = response.json()
    batch_id = body.get("id")
    if not batch_id:
        raise JudgeError(f"batches submit missing id: {body}")
    log.info(
        "judge.batch.submitted",
        batch_id=batch_id,
        n_requests=len(requests),
        processing_status=body.get("processing_status"),
    )
    return batch_id


async def _poll_batch(
    client: httpx.AsyncClient,
    config: Config,
    batch_id: str,
    max_wait_sec: float = 24 * 3600,
) -> dict:
    """Poll until the batch ends; return the final batch status dict."""
    started = time.monotonic()
    poll_idx = 0
    while True:
        response = await client.get(
            f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
            headers=_headers(config),
            timeout=60.0,
        )
        if response.status_code >= 400:
            raise JudgeError(
                f"batches poll HTTP {response.status_code} for {batch_id}: "
                f"{response.text[:300]}"
            )
        body = response.json()
        status = body.get("processing_status")
        counts = body.get("request_counts", {})
        elapsed = time.monotonic() - started

        if status == "ended":
            log.info(
                "judge.batch.ended",
                batch_id=batch_id,
                elapsed_sec=round(elapsed, 1),
                succeeded=counts.get("succeeded", 0),
                errored=counts.get("errored", 0),
                expired=counts.get("expired", 0),
            )
            return body
        if status == "canceling":
            raise JudgeError(f"batches {batch_id} was canceled")

        if elapsed > max_wait_sec:
            raise JudgeError(
                f"batches {batch_id} did not end within {max_wait_sec}s "
                f"(last status: {status}, counts: {counts})"
            )

        interval = POLL_INTERVALS_SEC[min(poll_idx, len(POLL_INTERVALS_SEC) - 1)]
        if poll_idx == 0 or poll_idx % 4 == 0:
            log.info(
                "judge.batch.polling",
                batch_id=batch_id,
                elapsed_sec=round(elapsed, 1),
                processing=counts.get("processing"),
                succeeded=counts.get("succeeded"),
                errored=counts.get("errored"),
                next_poll_sec=interval,
            )
        await asyncio.sleep(interval)
        poll_idx += 1


async def _fetch_results(
    client: httpx.AsyncClient,
    config: Config,
    batch: dict,
) -> list[dict]:
    """Stream the JSONL results endpoint and return one dict per line."""
    results_url = batch.get("results_url")
    if not results_url:
        raise JudgeError(f"batch ended without results_url: {batch}")

    rows: list[dict] = []
    async with client.stream(
        "GET", results_url, headers=_headers(config), timeout=300.0,
    ) as response:
        if response.status_code >= 400:
            text = await response.aread()
            raise JudgeError(
                f"batches results HTTP {response.status_code}: {text[:300]!r}"
            )
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise JudgeError(f"malformed results JSONL line: {line[:200]!r}") from exc
    return rows


# ---- Result parsing ----------------------------------------------------


def _parse_succeeded(custom_id: str, message: dict) -> TextRating:
    """Pull V/A out of a succeeded message; raise JudgeError on bad shape."""
    try:
        reply = message["content"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeError(
            f"unexpected batched judge response shape for {custom_id!r}: {message}"
        ) from exc

    match = _RESPONSE_RE.match(reply)
    if match is None:
        raise JudgeError(
            f"could not parse batched judge response for {custom_id!r}: {reply!r}"
        )

    valence = float(match.group(1))
    arousal = float(match.group(2))
    if not (1.0 <= valence <= 7.0) or not (1.0 <= arousal <= 7.0):
        raise JudgeError(
            f"batched judge for {custom_id!r} out of [1,7] range: "
            f"valence={valence}, arousal={arousal}"
        )
    return TextRating(text_id=custom_id, valence=valence, arousal=arousal)


def _process_result_row(row: dict) -> tuple[str, TextRating | JudgeError]:
    """One JSONL row → (custom_id, TextRating | JudgeError)."""
    custom_id = row.get("custom_id", "?")
    result = row.get("result") or {}
    result_type = result.get("type")

    if result_type == "succeeded":
        message = result.get("message") or {}
        try:
            return custom_id, _parse_succeeded(custom_id, message)
        except JudgeError as exc:
            return custom_id, exc
    if result_type == "errored":
        err = result.get("error") or {}
        # Permanent vs transient: invalid_request_error is permanent; other
        # types may be retryable, but for now surface as an error in both
        # cases — the caller's cache layer will keep them missing for retry.
        return custom_id, JudgeError(
            f"batched judge errored for {custom_id!r}: "
            f"{err.get('type', '?')}: {(err.get('message') or '')[:200]}"
        )
    if result_type in ("canceled", "expired"):
        return custom_id, JudgeError(
            f"batched judge {result_type} for {custom_id!r}"
        )
    return custom_id, JudgeError(
        f"batched judge unknown result type {result_type!r} for {custom_id!r}"
    )


# ---- Public API --------------------------------------------------------


async def judge_texts_batched(
    config: Config,
    passages: list[tuple[str, str]],  # [(text_id, text), ...]
    cache_path: Path | None = None,
) -> dict[str, TextRating]:
    """Drop-in replacement for ``judge_text.judge_texts`` using the Batches API.

    Same signature, same cache JSON shape, same return semantics.
    Resumable: passages already in ``cache_path`` are not re-submitted.

    For ``len(passages) > MAX_REQUESTS_PER_BATCH`` we chunk into multiple
    sequential batches and merge results — keeps any single batch's wait
    bounded and lets us write incremental cache updates per chunk.
    """
    cache: dict[str, TextRating] = {}
    if cache_path is not None and cache_path.exists():
        for row in json.loads(cache_path.read_text()):
            cache[row["text_id"]] = TextRating(**row)

    missing = [(tid, text) for tid, text in passages if tid not in cache]
    log.info(
        "behavior.judge.batched.start",
        total=len(passages),
        cached=len(passages) - len(missing),
        to_submit=len(missing),
        chunk_size=MAX_REQUESTS_PER_BATCH,
    )
    if not missing:
        return {tid: cache[tid] for tid, _ in passages if tid in cache}

    errors: list[JudgeError] = []
    model = config.judge.model

    async with httpx.AsyncClient(http2=False) as client:
        for chunk_start in range(0, len(missing), MAX_REQUESTS_PER_BATCH):
            chunk = missing[chunk_start:chunk_start + MAX_REQUESTS_PER_BATCH]
            # Anthropic's batches API enforces custom_id regex
            # ^[a-zA-Z0-9_-]{1,64}$ — text_ids can contain spaces (multi-word
            # labels like 'at ease') and may exceed 64 chars, so we substitute
            # a positional safe_id and keep a mapping back to the real text_id.
            safe_id_to_text_id: dict[str, str] = {}
            requests: list[dict] = []
            for i, (text_id, text) in enumerate(chunk):
                safe_id = f"req{chunk_start + i:07d}"
                safe_id_to_text_id[safe_id] = text_id
                requests.append(_build_request(safe_id, text, model))
            batch_id = await _submit_batch(client, config, requests)
            batch = await _poll_batch(client, config, batch_id)
            rows = await _fetch_results(client, config, batch)

            for row in rows:
                safe_id, parsed = _process_result_row(row)
                real_text_id = safe_id_to_text_id.get(safe_id)
                if real_text_id is None:
                    errors.append(JudgeError(
                        f"batched judge returned unknown custom_id {safe_id!r}"
                    ))
                    continue
                if isinstance(parsed, JudgeError):
                    # Re-wrap with real text_id for clearer error message
                    errors.append(JudgeError(
                        str(parsed).replace(repr(safe_id), repr(real_text_id))
                    ))
                else:
                    # Rebind TextRating.text_id from safe_id back to real_text_id
                    cache[real_text_id] = TextRating(
                        text_id=real_text_id,
                        valence=parsed.valence,
                        arousal=parsed.arousal,
                    )

            # Incremental cache write per chunk so a long chain can resume
            # if interrupted mid-run.
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        [
                            {"text_id": r.text_id,
                             "valence": r.valence,
                             "arousal": r.arousal}
                            for r in cache.values()
                        ],
                        indent=2,
                    )
                )

    log.info(
        "behavior.judge.batched.done",
        rated=len([tid for tid, _ in passages if tid in cache]),
        errors=len(errors),
        first_errors=[str(e) for e in errors[:3]],
    )
    return {tid: cache[tid] for tid, _ in passages if tid in cache}


# Alias so a caller can do `from ...judge_text_batched import judge_texts`
# and have the same import name as the sync version.
judge_texts = judge_texts_batched
