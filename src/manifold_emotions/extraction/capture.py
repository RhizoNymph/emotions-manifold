"""Send stories through vLLM with a per-request capture spec.

Each request triggers the filesystem capture consumer to write a
(num_positions, hidden_size) bf16 .bin file plus a sidecar JSON to
`{capture.root}/{tag_slug}/{request_id_slug}/{layer}_{hook}.bin`.

We set `max_tokens=1` because we only need the prompt forward pass —
captured residuals come from prompt positions.

Known fork limitation: the HTTP response body's ``capture_results`` is
empty even when the disk write succeeds — capture finalize on the
worker happens at step N+1, but the request's ``EngineCoreOutput`` was
emitted at step N and the request state is freed before the results
can be attached. We treat HTTP 200 as "admission + generation succeeded"
and verify the capture by checking the expected disk path afterward,
with a brief retry for the writer's async drain.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from pathlib import Path

import httpx
import structlog

from ..config import Config
from ..errors import CaptureError
from ..types import Story

log = structlog.get_logger(__name__)


# Mirrors the consumer's slugging at
# ``vllm.v1.capture.consumers.filesystem.validation._slug`` — characters
# outside [a-zA-Z0-9._-] are replaced with '_'. We slug client-side too
# so we can predict the on-disk path without parsing the response.
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _slug(name: str) -> str:
    return _SLUG_RE.sub("_", name)


def _expected_bin_path(config: Config, story: Story) -> Path:
    return (
        config.capture.root
        / _slug(story.emotion)
        / _slug(story.request_id)
        / f"{config.model.target_layer}_{config.model.hook_point}.bin"
    )


def _build_capture_payload(config: Config, story: Story) -> dict:
    """Build the OpenAI-compatible request body with the filesystem capture spec.

    `capture` lives at the top level of the request body — it's declared as a
    field on ``ChatCompletionRequest`` in the fork's protocol module. The
    docs' ``extra_body`` example is OpenAI-SDK ergonomics; the SDK flattens
    that key before sending, but raw HTTP needs the field at the top level.
    """
    return {
        "model": config.model.hf_id,
        "messages": [{"role": "user", "content": story.text}],
        "max_tokens": 1,
        "temperature": 0.0,
        "capture": {
            "filesystem": {
                "request_id": story.request_id,
                "tag": story.emotion,
                "hooks": {config.model.hook_point: [config.model.target_layer]},
                "positions": "all_prompt",
            },
        },
    }


async def _capture_one(
    client: httpx.AsyncClient,
    config: Config,
    story: Story,
    semaphore: asyncio.Semaphore,
) -> Path:
    """Send one story and return the path to its captured .bin file."""
    payload = _build_capture_payload(config, story)
    headers = {}
    if config.vllm_server.api_key:
        headers["Authorization"] = f"Bearer {config.vllm_server.api_key}"

    async with semaphore:
        try:
            response = await client.post(
                f"{config.vllm_server.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=180.0,
            )
        except httpx.HTTPError as exc:
            raise CaptureError(
                f"capture HTTP error for {story.request_id}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise CaptureError(
                f"capture HTTP {response.status_code} for {story.request_id} "
                f"at {response.request.url}: {response.text}"
            )

    body = response.json()
    capture_results = body.get("capture_results") or {}
    result = capture_results.get("filesystem")

    # Preferred path: response carries an explicit per-consumer result.
    # When the fork closes the EngineCoreOutput-vs-capture-finalize timing
    # gap we'll always land here.
    if result is not None:
        if result.get("status") != "ok":
            raise CaptureError(
                f"capture status {result.get('status')!r} for {story.request_id}: "
                f"{result.get('error')}"
            )
        paths = result.get("payload") or []
        if paths:
            return Path(paths[0])

    # Fallback path: response omits or empties capture_results because the
    # writer's finalize lands one engine step after the terminal token.
    # We know the expected disk path from our own slugs; the writer is
    # async so we briefly wait for the .bin to appear.
    expected = _expected_bin_path(config, story)
    deadline = asyncio.get_event_loop().time() + 5.0
    while not expected.exists():
        if asyncio.get_event_loop().time() > deadline:
            raise CaptureError(
                f"no capture_results.filesystem in response for {story.request_id} "
                f"and expected .bin file did not appear at {expected}. "
                f"body keys: {list(body.keys())}; capture_results: {capture_results!r}"
            )
        await asyncio.sleep(0.05)

    return expected


async def capture_corpus(
    config: Config,
    stories: Iterable[Story],
    concurrency: int = 16,
) -> tuple[int, int]:
    """Capture activations for every story in the iterable.

    Returns (succeeded, failed). Failed requests are logged and skipped.
    """
    stories_list = list(stories)
    total = len(stories_list)

    log.info(
        "capture.start",
        total=total,
        concurrency=concurrency,
        layer=config.model.target_layer,
        hook=config.model.hook_point,
        capture_root=str(config.capture.root),
    )

    semaphore = asyncio.Semaphore(concurrency)
    succeeded = 0
    errors: list[CaptureError] = []
    results: list[Path | CaptureError] = []

    async with httpx.AsyncClient(http2=False) as client:
        async def run_one(story: Story) -> Path | CaptureError:
            try:
                return await _capture_one(client, config, story, semaphore)
            except CaptureError as exc:
                return exc

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run_one(s)) for s in stories_list]

        for task in tasks:
            results.append(task.result())

    for result in results:
        if isinstance(result, CaptureError):
            errors.append(result)
        else:
            succeeded += 1

    log.info(
        "capture.done",
        succeeded=succeeded,
        failed=len(errors),
        first_errors=[str(e) for e in errors[:3]],
    )
    return succeeded, len(errors)


def load_corpus(jsonl_path: Path) -> list[Story]:
    """Read a stories JSONL file produced by corpus.generate into a list of Story."""
    from ..types import EmotionLabel, Topic

    stories: list[Story] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stories.append(
            Story(
                emotion=EmotionLabel(row["emotion"]),
                topic=Topic(row["topic"]),
                story_idx=row["story_idx"],
                text=row["text"],
            )
        )
    return stories
