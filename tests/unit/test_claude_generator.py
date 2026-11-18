"""The Claude answer-generator adapter, against a fake client: no network, no API key."""

import json
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import anthropic
import httpx2
import pytest

from finrag.core.domain import DraftAnswer, DraftCitation, EvidenceChunk
from finrag.core.errors import ModelOutputError, TransientError
from finrag.generation.claude import ClaudeAnswerGenerator

CHUNK = EvidenceChunk(uuid4(), uuid4(), "Cash was $1,234 million.", ("Part II",), 41, 41)


class FakeClient:
    """Stands in for anthropic.Anthropic: records the request, returns or raises a script."""

    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> object:
        self.requests.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def reply(text: str, stop_reason: str = "end_turn") -> object:
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], stop_reason=stop_reason)


def generator(client: FakeClient) -> ClaudeAnswerGenerator:
    return ClaudeAnswerGenerator(client, model="claude-sonnet-5-5", max_tokens=1000)  # type: ignore[arg-type]


GOOD = json.dumps(
    {
        "answer": "Cash was $1,234 million [1].",
        "citations": [
            {"marker": 1, "evidence_number": 1, "quoted_span": "Cash was $1,234 million"}
        ],
    }
)


def test_it_asks_for_structured_output_from_the_configured_model_and_parses_the_reply() -> None:
    client = FakeClient(reply(GOOD))

    draft = generator(client).generate("How much cash?", (CHUNK,))

    assert draft == DraftAnswer(
        text="Cash was $1,234 million [1].",
        citations=(DraftCitation(1, CHUNK.chunk_id, "Cash was $1,234 million"),),
    )
    (request,) = client.requests
    assert request["model"] == "claude-sonnet-5-5"
    assert request["max_tokens"] == 1000
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "Cash was $1,234 million." in request["messages"][0]["content"]  # the Evidence
    assert "How much cash?" in request["messages"][0]["content"]
    assert "Cash was $1,234 million." not in request["system"]  # document text is never a rule
    assert "temperature" not in request  # sampling parameters are rejected by this model


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_a_refused_or_cut_off_reply_is_a_model_output_error(stop_reason: str) -> None:
    client = FakeClient(reply(GOOD, stop_reason=stop_reason))

    with pytest.raises(ModelOutputError):
        generator(client).generate("How much cash?", (CHUNK,))


def test_a_reply_that_is_not_json_is_a_model_output_error() -> None:
    with pytest.raises(ModelOutputError):
        generator(FakeClient(reply("Sorry, I cannot help."))).generate("q", (CHUNK,))


def test_a_reply_with_no_text_is_a_model_output_error() -> None:
    empty = SimpleNamespace(content=[], stop_reason="end_turn")

    with pytest.raises(ModelOutputError):
        generator(FakeClient(empty)).generate("q", (CHUNK,))


def request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.test/v1/messages")


def status_error(code: int) -> anthropic.APIStatusError:
    response = httpx2.Response(code, request=request())
    return anthropic.APIStatusError("boom", response=response, body=None)


@pytest.mark.parametrize(
    "failure",
    [
        anthropic.APITimeoutError(request=request()),
        anthropic.APIConnectionError(request=request()),
        status_error(429),
        status_error(503),
    ],
    ids=["timeout", "connection", "rate-limited", "server-error"],
)
def test_retryable_provider_failures_become_transient_errors_without_provider_detail(
    failure: Exception,
) -> None:
    with pytest.raises(TransientError) as caught:
        generator(FakeClient(failure)).generate("q", (CHUNK,))

    assert "api.anthropic.test" not in str(caught.value)


def test_a_non_retryable_provider_error_is_not_swallowed() -> None:
    with pytest.raises(anthropic.APIStatusError):
        generator(FakeClient(status_error(401))).generate("q", (CHUNK,))
