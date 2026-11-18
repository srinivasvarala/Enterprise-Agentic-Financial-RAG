"""POST /query through create_app with in-memory fakes: a Server-Sent Events stream (ADR-0008)."""

import json
from typing import Any

from harness import Env
from httpx import Response

CASH = "Cash and cash equivalents were $1,234 million at year end."
QUESTION = "how much cash and cash equivalents"


def ingest(env: Env, text: str = CASH) -> str:
    response = env.client.post("/documents", files={"file": ("f.txt", text.encode(), "text/plain")})
    env.run_jobs()
    document_id: str = response.json()["id"]
    return document_id


def events(response: Response) -> list[tuple[str, Any]]:
    """The (event name, JSON data) pairs of an SSE body, in order."""
    parsed: list[tuple[str, Any]] = []
    for frame in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines())
        parsed.append((fields["event"], json.loads(fields["data"])))
    return parsed


def ask(env: Env, question: str = QUESTION) -> Response:
    return env.client.post("/query", json={"question": question})


def test_a_question_streams_progress_then_one_answer_with_resolved_citations(env: Env) -> None:
    document_id = ingest(env)

    response = ask(env)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    stream = events(response)
    assert [name for name, _ in stream] == ["progress"] * 3 + ["answer"]
    assert [data["stage"] for name, data in stream if name == "progress"] == [
        "retrieving",
        "generating",
        "validating",
    ]
    _, answer = stream[-1]
    assert answer["refusal_reason"] is None
    assert answer["text"] == "Cash and cash equivalents were $1,234 million at year end [1]."
    (citation,) = answer["citations"]
    assert citation["marker"] == 1
    assert citation["document_id"] == document_id
    assert citation["page_start"] == citation["page_end"] == 1
    assert citation["quoted_span"] == "Cash and cash equivalents were $1,234 million at year end"


def test_nothing_relevant_indexed_is_a_refusal_answer_not_an_error(env: Env) -> None:
    response = ask(env)  # nothing ingested

    assert response.status_code == 200
    stream = events(response)
    assert [name for name, _ in stream] == ["progress", "answer"]
    _, answer = stream[-1]
    assert answer["refusal_reason"] == "insufficient_evidence"
    assert answer["citations"] == []


class FailingGenerator:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate(self, question: str, evidence: object) -> object:
        raise self.error


def test_a_provider_failure_ends_the_stream_with_an_error_event_and_no_answer(env: Env) -> None:
    from finrag.core.errors import TransientError

    ingest(env)
    env.container.generator = FailingGenerator(  # type: ignore[assignment]
        TransientError("https://api.anthropic.com/v1/messages returned 503")
    )

    response = ask(env)

    assert response.status_code == 200  # the stream had already started
    stream = events(response)
    assert [name for name, _ in stream][-1] == "error"
    assert "answer" not in [name for name, _ in stream]
    message = stream[-1][1]["message"]
    assert "temporarily unavailable" in message
    assert "anthropic" not in message and "503" not in message


def test_an_unexpected_failure_is_a_generic_error_event_without_internals(env: Env) -> None:
    ingest(env)
    env.container.generator = FailingGenerator(  # type: ignore[assignment]
        RuntimeError("password=hunter2 at postgres://db.internal")
    )

    stream = events(ask(env))

    assert stream[-1][0] == "error"
    assert "hunter2" not in json.dumps(stream) and "internal" not in json.dumps(stream)


def test_an_invalid_question_is_rejected_before_any_stream_starts(env: Env) -> None:
    too_long = "x" * (env.settings.max_question_chars + 1)

    assert ask(env, "").status_code == 422
    assert ask(env, "   \n ").status_code == 422
    assert ask(env, too_long).status_code == 422
    assert env.client.post("/query", json={}).status_code == 422
    assert ask(env, "x" * env.settings.max_question_chars).status_code == 200
