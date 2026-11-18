"""POST /query: answers a question as a Server-Sent Events stream (ADR-0008).

Events, in order: `progress` {stage} for retrieving, generating and validating, then exactly one
`answer` {text, citations, refusal_reason}. A Refusal is an `answer` with a refusal_reason. If
answering fails after the stream has started, a single `error` {message} ends it instead. Text is
never sent before it has been validated.
"""

import json
import logging
import queue
import threading
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from finrag.api.deps import AnswerServiceDep, PrincipalDep
from finrag.core.domain import Answer, AnswerStage
from finrag.core.errors import TransientError

log = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

UNAVAILABLE = "The answer service is temporarily unavailable. Try again."
FAILED = "The question could not be answered. Try again later."


class QueryRequest(BaseModel):
    question: str


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _answer_data(answer: Answer) -> dict[str, Any]:
    return {
        "text": answer.text,
        "refusal_reason": answer.refusal_reason.value if answer.refusal_reason else None,
        "citations": [
            {
                "marker": c.marker,
                "document_id": str(c.document_id),
                "section_path": list(c.section_path),
                "page_start": c.page_start,
                "page_end": c.page_end,
                "quoted_span": c.quoted_span,
            }
            for c in answer.citations
        ],
    }


@router.post("/query")
def query(
    body: QueryRequest, request: Request, principal: PrincipalDep, service: AnswerServiceDep
) -> StreamingResponse:
    question = body.question.strip()
    limit = request.app.state.settings.max_question_chars
    if not question:
        raise HTTPException(422, "The question is empty.")
    if len(question) > limit:
        raise HTTPException(422, f"The question is longer than {limit} characters.")

    events: queue.Queue[str | None] = queue.Queue()

    def work() -> None:
        def stage(s: AnswerStage) -> None:
            events.put(_event("progress", {"stage": s.value}))

        try:
            answer = service.answer(principal, question, on_stage=stage)
            events.put(_event("answer", _answer_data(answer)))
            # ids, reason and counts only: a question about a position can itself be sensitive
            log.info(
                "question answered",
                extra={
                    "tenant_id": str(principal.tenant_id),
                    "refusal_reason": answer.refusal_reason,
                    "citations": len(answer.citations),
                    "question_chars": len(question),
                },
            )
        except TransientError:
            log.warning(
                "answer provider unavailable", extra={"tenant_id": str(principal.tenant_id)}
            )
            events.put(_event("error", {"message": UNAVAILABLE}))
        except Exception:
            log.exception("answering failed", extra={"tenant_id": str(principal.tenant_id)})
            events.put(_event("error", {"message": FAILED}))
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream() -> Iterator[str]:
        while (item := events.get()) is not None:
            yield item

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )
