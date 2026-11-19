"""Seam: the eval index through the real ingestion pipeline (text parser stands in for Docling)."""

from pathlib import Path

import pytest
import yaml
from harness import Env

from finrag.core.config import Environment, Settings
from finrag.core.domain import ParsedDocument, Principal
from finrag.core.errors import ParseError
from finrag.evaluation.environment import (
    EVAL_COLLECTION_ID,
    EVAL_QDRANT_COLLECTION,
    EVAL_S3_BUCKET,
    EVAL_TENANT_ID,
    EvalEnvironmentError,
    PipelineFilingIndexer,
    eval_principal,
    require_eval_settings,
    seed_eval_tenant,
)
from finrag.evaluation.index import IndexBuildError
from finrag.ingestion.text_parser import TextParser

QUESTION = "supply chains disrupted shortages"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def indexer(env: Env) -> PipelineFilingIndexer:
    env.container.parsers["application/pdf"] = TextParser()  # type: ignore[index]
    seed_eval_tenant(env.container.session_factory)
    return PipelineFilingIndexer(env.container, env.pipeline())


def eval_documents(env: Env) -> int:
    service = env.container.document_service()
    return len(service.list_documents(eval_principal()))


def test_a_filing_is_ingested_into_the_eval_tenant_and_a_second_run_adds_nothing(
    env: Env, indexer: PipelineFilingIndexer
) -> None:
    pdf = b"Item 1A. Risk Factors\n\nSupply chains may be disrupted by shortages."

    assert indexer.index("A.pdf", pdf) is True
    assert indexer.index("A.pdf", pdf) is False

    assert eval_documents(env) == 1
    assert env.container.retriever().retrieve(eval_principal(), QUESTION)


def test_eval_documents_are_invisible_to_the_default_tenant(
    env: Env, indexer: PipelineFilingIndexer
) -> None:
    indexer.index("A.pdf", b"Item 1A. Risk Factors\n\nSupply chains may be disrupted.")

    default = Principal(env.settings.default_tenant_id, env.settings.default_collection_id)
    assert env.container.document_service().list_documents(default) == []
    assert env.client.get("/documents").json() == []
    assert env.container.retriever().retrieve(default, QUESTION) == ()
    assert env.container.retriever().retrieve(eval_principal(), QUESTION)
    assert eval_documents(env) == 1


def test_a_filing_that_fails_ingestion_is_reported_and_retried_on_the_next_run(
    env: Env, indexer: PipelineFilingIndexer
) -> None:
    class Unparseable:
        def parse(self, data: bytes) -> ParsedDocument:
            raise ParseError("The PDF could not be read.")

    good = env.container.parsers["application/pdf"]
    env.container.parsers["application/pdf"] = Unparseable()  # type: ignore[index]
    pdf = b"Item 1A. Risk Factors\n\nSupply chains may be disrupted."

    with pytest.raises(IndexBuildError, match=r"A\.pdf.*could not be read"):
        indexer.index("A.pdf", pdf)

    env.container.parsers["application/pdf"] = good  # type: ignore[index]
    assert indexer.index("A.pdf", pdf) is True
    assert eval_documents(env) == 1


def test_require_eval_settings_refuses_the_dev_stack_settings() -> None:
    with pytest.raises(EvalEnvironmentError, match="eval"):
        require_eval_settings(Settings(env=Environment.TEST))


def test_require_eval_settings_accepts_the_eval_profile_settings() -> None:
    require_eval_settings(
        Settings(
            env=Environment.TEST,
            default_tenant_id=EVAL_TENANT_ID,
            default_collection_id=EVAL_COLLECTION_ID,
            qdrant_collection=EVAL_QDRANT_COLLECTION,
            s3_bucket=EVAL_S3_BUCKET,
        )
    )


EVAL_SERVICES = ("eval-index", "eval-retrieval")


@pytest.mark.parametrize("name", EVAL_SERVICES)
def test_the_compose_eval_profile_matches_the_eval_environment_and_is_memory_capped(
    name: str,
) -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    service = compose["services"][name]

    assert service["profiles"] == ["eval"]
    assert service["build"]["target"] == "eval"
    assert service["mem_limit"] == "6g"
    env = service["environment"]
    assert env["FINRAG_DEFAULT_TENANT_ID"] == str(EVAL_TENANT_ID)
    assert env["FINRAG_DEFAULT_COLLECTION_ID"] == str(EVAL_COLLECTION_ID)
    assert env["FINRAG_QDRANT_COLLECTION"] == EVAL_QDRANT_COLLECTION
    assert env["FINRAG_S3_BUCKET"] == EVAL_S3_BUCKET
    # no other service is part of the profile: the dev stack is untouched by `--profile eval`
    others = [n for n, s in compose["services"].items() if n not in EVAL_SERVICES]
    assert all("profiles" not in compose["services"][n] for n in others)
