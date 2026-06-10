"""Batched judge tests against a mocked Batches API. No network access.

Exercises the full submit → poll → fetch-results path via
httpx.MockTransport, including the safe_id ↔ text_id mapping for
multi-word labels, error-row handling, range validation, and cache
resumability.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import manifold_emotions.behavior.judge_text_batched as jtb
from manifold_emotions.config import Config, load_config


@pytest.fixture()
def config(monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return load_config()


class _FakeBatchesApi:
    """Stateful handler for httpx.MockTransport simulating one batch round."""

    def __init__(self, replies: dict[int, dict]) -> None:
        # replies: positional request index -> result row template. The
        # custom_id is filled in from the actual submitted request so the
        # test exercises the real safe_id mapping rather than assuming it.
        self.replies = replies
        self.submits: list[dict] = []
        self.polls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages/batches":
            self.submits.append(json.loads(request.content))
            return httpx.Response(
                200, json={"id": "msgbatch_test", "processing_status": "in_progress"}
            )
        if request.method == "GET" and path == "/v1/messages/batches/msgbatch_test":
            self.polls += 1
            return httpx.Response(
                200,
                json={
                    "processing_status": "ended",
                    "request_counts": {"succeeded": len(self.replies)},
                    "results_url": "https://api.anthropic.com/fake_results",
                },
            )
        if request.method == "GET" and path == "/fake_results":
            submitted = self.submits[-1]["requests"]
            lines = []
            for i, req in enumerate(submitted):
                row = dict(self.replies[i])
                row["custom_id"] = req["custom_id"]
                lines.append(json.dumps(row))
            return httpx.Response(200, text="\n".join(lines) + "\n")
        raise AssertionError(f"unexpected request: {request.method} {request.url}")


def _succeeded(text: str) -> dict:
    return {
        "result": {
            "type": "succeeded",
            "message": {"content": [{"type": "text", "text": text}]},
        }
    }


def _install(monkeypatch: pytest.MonkeyPatch, api: _FakeBatchesApi) -> None:
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(api)
    monkeypatch.setattr(
        jtb.httpx, "AsyncClient", lambda **kw: real_client(transport=transport)
    )


async def test_judge_texts_batched_round_trip(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeBatchesApi(
        replies={
            0: _succeeded("6.0 4.5"),
            1: _succeeded("2 5"),
            2: {
                "result": {
                    "type": "errored",
                    "error": {"type": "api_error", "message": "boom"},
                }
            },
            3: _succeeded("9 9"),  # out of [1,7] range -> rejected
        }
    )
    _install(monkeypatch, api)

    passages = [
        ("at ease_0", "a calm passage"),  # multi-word label: unsafe as custom_id
        ("happy_1", "a happy passage"),
        ("errored_2", "a passage the API fails on"),
        ("out_of_range_3", "a passage with a bad rating"),
    ]
    cache_path = tmp_path / "ratings.json"
    ratings = await jtb.judge_texts_batched(config, passages, cache_path=cache_path)

    assert set(ratings) == {"at ease_0", "happy_1"}
    assert ratings["at ease_0"].valence == 6.0
    assert ratings["at ease_0"].arousal == 4.5
    assert ratings["at ease_0"].text_id == "at ease_0"  # mapped back from safe_id
    assert ratings["happy_1"].valence == 2.0

    # The submitted custom_ids must satisfy the API regex even though the
    # text_ids do not.
    submitted_ids = [r["custom_id"] for r in api.submits[0]["requests"]]
    assert all(cid.isidentifier() or cid.replace("-", "_").isidentifier() for cid in submitted_ids)
    assert "at ease_0" not in submitted_ids

    # Cache holds exactly the successful ratings, keyed by real text_id.
    cached = {row["text_id"] for row in json.loads(cache_path.read_text())}
    assert cached == {"at ease_0", "happy_1"}


async def test_judge_texts_batched_resumes_from_cache(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "ratings.json"
    cache_path.write_text(
        json.dumps(
            [
                {"text_id": "happy_0", "valence": 6.5, "arousal": 5.0},
                {"text_id": "sad_0", "valence": 1.5, "arousal": 2.0},
            ]
        )
    )
    api = _FakeBatchesApi(replies={})
    _install(monkeypatch, api)

    passages = [("happy_0", "text"), ("sad_0", "text")]
    ratings = await jtb.judge_texts_batched(config, passages, cache_path=cache_path)

    assert api.submits == []  # fully cached: no batch submitted
    assert ratings["happy_0"].valence == 6.5
    assert ratings["sad_0"].arousal == 2.0


async def test_judge_texts_batched_submits_only_missing(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "ratings.json"
    cache_path.write_text(
        json.dumps([{"text_id": "happy_0", "valence": 6.5, "arousal": 5.0}])
    )
    api = _FakeBatchesApi(replies={0: _succeeded("3.0 3.0")})
    _install(monkeypatch, api)

    passages = [("happy_0", "cached text"), ("new_0", "uncached text")]
    ratings = await jtb.judge_texts_batched(config, passages, cache_path=cache_path)

    assert len(api.submits) == 1
    assert len(api.submits[0]["requests"]) == 1  # only the uncached passage
    assert set(ratings) == {"happy_0", "new_0"}
    assert ratings["new_0"].valence == 3.0
    # Cache merged old + new.
    cached = {row["text_id"] for row in json.loads(cache_path.read_text())}
    assert cached == {"happy_0", "new_0"}
