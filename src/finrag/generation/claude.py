"""Claude adapter for the AnswerGenerator Protocol (ADR-0002, ADR-0005)."""

import json
import logging
import threading
from collections.abc import Sequence

import anthropic

from finrag.core.config import Settings
from finrag.core.domain import DraftAnswer, EvidenceChunk, TokenUsage
from finrag.core.errors import ModelOutputError, TransientError
from finrag.generation.draft import parse_draft
from finrag.generation.prompt import build_prompt

log = logging.getLogger(__name__)

# Structured output: the shape parse_draft expects. The model sees evidence numbers, not ids.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "marker": {"type": "integer"},
                    "evidence_number": {"type": "integer"},
                    "quoted_span": {"type": "string"},
                },
                "required": ["marker", "evidence_number", "quoted_span"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}

# A safety classifier may decline; the API then re-runs the request on its fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeAnswerGenerator:
    def __init__(self, client: anthropic.Anthropic, model: str, max_tokens: int) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._usage = TokenUsage()
        self._usage_lock = threading.Lock()  # one generator serves concurrent API requests

    def usage(self) -> TokenUsage:
        """Tokens the API has reported for this generator so far (read by the eval dry run)."""
        with self._usage_lock:
            return self._usage

    @classmethod
    def from_settings(cls, settings: Settings) -> "ClaudeAnswerGenerator":
        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value() or None,
            timeout=settings.answer_timeout_seconds,
        )
        return cls(client, settings.answer_model, settings.answer_max_tokens)

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> DraftAnswer:
        prompt = build_prompt(question, evidence)
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                output_config={"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIConnectionError as exc:  # includes timeouts
            log.warning("answer model unreachable: %s", exc)
            raise TransientError("the answer model could not be reached") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                log.warning("answer model returned %s", exc.status_code)
                raise TransientError("the answer model is unavailable") from exc
            raise  # a rejected request or bad credentials is a configuration fault

        with self._usage_lock:
            self._usage += TokenUsage(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ModelOutputError(f"model stopped with {response.stop_reason}")
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelOutputError("model reply is not JSON") from exc
        return parse_draft(raw, evidence)
