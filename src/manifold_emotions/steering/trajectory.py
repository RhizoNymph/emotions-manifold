"""Generate continuations along a path of steering waypoints.

For each (waypoint, prompt) pair, send a chat completion to vLLM with
the waypoint as a steering vector at the configured hook+layer. Returns
the generated continuations, keyed by (waypoint_index, prompt_index).

Used by Phase 8 to compare manifold steering vs. linear steering: same
prompts, same K, same vLLM config — only the waypoint path differs.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Iterable
from dataclasses import dataclass

import httpx
import numpy as np
import structlog

from ..config import Config
from ..errors import CaptureError

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SteeredContinuation:
    """One continuation generated under a specific steering waypoint."""

    waypoint_index: int  # 0..K-1, position along the path
    prompt_index: int  # 0..N-1, which prompt in the eval set
    text: str
    finish_reason: str | None


def _build_payload(
    config: Config,
    prompt: str,
    waypoint: np.ndarray,
    max_tokens: int,
) -> dict:
    """Build a chat-completion request with the waypoint as a steering vector.

    Uses the fork's binary wire format (see docs/steering.md): each hook
    carries one base64-encoded ``(num_layers, hidden_size)`` blob plus a
    sibling ``layer_indices`` list. The server decodes via zero-copy
    ``np.frombuffer``, avoiding the ~10-15 ms-per-request cost a JSON
    ``list[float]`` payload would incur on the API-server event loop.
    Inline JSON-list steering is only supported on the in-process Python
    API, not over HTTP.
    """
    layer = config.model.target_layer
    hook = config.model.hook_point
    # Reshape to (num_layers=1, hidden_size) per the binary format.
    stacked = waypoint.astype(np.float32).reshape(1, -1)
    return {
        "model": config.model.hf_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "steering_vectors": {
            hook: {
                "dtype": str(stacked.dtype),
                "shape": list(stacked.shape),
                "layer_indices": [layer],
                "data": base64.b64encode(stacked.tobytes()).decode("ascii"),
            },
        },
    }


async def _generate_one(
    client: httpx.AsyncClient,
    config: Config,
    prompt_idx: int,
    prompt: str,
    waypoint_idx: int,
    waypoint: np.ndarray,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> SteeredContinuation:
    payload = _build_payload(config, prompt, waypoint, max_tokens)
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
                f"steered generation HTTP error for wp={waypoint_idx} p={prompt_idx}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise CaptureError(
                f"steered generation HTTP {response.status_code} for "
                f"wp={waypoint_idx} p={prompt_idx}: {response.text}"
            )

    body = response.json()
    try:
        choice = body["choices"][0]
        text = choice["message"]["content"].strip()
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, AttributeError) as exc:
        raise CaptureError(
            f"malformed steered response for wp={waypoint_idx} p={prompt_idx}: {body}"
        ) from exc

    return SteeredContinuation(
        waypoint_index=waypoint_idx,
        prompt_index=prompt_idx,
        text=text,
        finish_reason=finish_reason,
    )


async def generate_along_path(
    config: Config,
    waypoints: np.ndarray,  # (K, hidden_size)
    prompts: Iterable[str],
    max_tokens: int = 128,
    concurrency: int = 16,
) -> list[SteeredContinuation]:
    """For each (waypoint, prompt) pair, generate one steered continuation.

    Returns K * len(prompts) continuations as a flat list. Caller can
    reshape via ``(c.waypoint_index, c.prompt_index)``.
    """
    prompts_list = list(prompts)
    k = waypoints.shape[0]
    total = k * len(prompts_list)
    log.info(
        "steering.trajectory.start",
        num_waypoints=k,
        num_prompts=len(prompts_list),
        total_generations=total,
        concurrency=concurrency,
        max_tokens=max_tokens,
    )

    semaphore = asyncio.Semaphore(concurrency)
    results: list[SteeredContinuation | CaptureError] = []

    async with httpx.AsyncClient(http2=False) as client:

        async def run_one(
            wi: int, pi: int, wp: np.ndarray, p: str
        ) -> SteeredContinuation | CaptureError:
            try:
                return await _generate_one(
                    client, config, pi, p, wi, wp, max_tokens, semaphore
                )
            except CaptureError as exc:
                return exc

        async with asyncio.TaskGroup() as tg:
            tasks = []
            for wi in range(k):
                for pi, prompt in enumerate(prompts_list):
                    tasks.append(
                        tg.create_task(run_one(wi, pi, waypoints[wi], prompt))
                    )

        for task in tasks:
            results.append(task.result())

    succeeded = [r for r in results if isinstance(r, SteeredContinuation)]
    errors = [r for r in results if isinstance(r, CaptureError)]
    log.info(
        "steering.trajectory.done",
        succeeded=len(succeeded),
        errors=len(errors),
        first_errors=[str(e) for e in errors[:3]],
    )
    return succeeded
