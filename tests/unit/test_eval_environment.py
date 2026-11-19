"""Seam: the eval index through the real ingestion pipeline (text parser stands in for Docling)."""

import dataclasses
from pathlib import Path

import pytest
import yaml
from harness import Env

from finrag.core.config import Environment, Settings
from finrag.core.container import Container
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
    wipe_eval_index,
)
from finrag.evaluation.index import IndexBuildError
from finrag.ingestion.text_parser import TextParser
from finrag.storage.models import Chunk, DocumentVersion

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


@pytest.mark.parametrize("name", ["eval-index", "eval-retrieval", "eval-full"])
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
    eval_services = ("eval-index", "eval-dry-run", "eval-retrieval", "eval-full")
    others = [n for n in compose["services"] if n not in eval_services]
    assert all("profiles" not in compose["services"][n] for n in others)
    # the dry run reads and writes only the eval Tenant, Collection and collections too
    dry_run = compose["services"]["eval-dry-run"]
    assert dry_run["profiles"] == ["eval"]
    assert dry_run["environment"]["FINRAG_DEFAULT_TENANT_ID"] == str(EVAL_TENANT_ID)
    assert dry_run["environment"]["FINRAG_QDRANT_COLLECTION"] == EVAL_QDRANT_COLLECTION


def test_the_full_eval_service_passes_the_keys_through_and_mounts_results_writable() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    service = compose["services"]["eval-full"]

    env = service["environment"]
    assert (
        env["FINRAG_ANTHROPIC_API_KEY"] == "${FINRAG_ANTHROPIC_API_KEY:-}"
    )  # host env, no literal
    assert env["FINRAG_VOYAGE_API_KEY"] == "${FINRAG_VOYAGE_API_KEY:-}"
    assert env["FINRAG_EVAL_COST_CAP_USD"] == "${FINRAG_EVAL_COST_CAP_USD:-}"  # no default cap
    assert "./evals/results:/evals/results" in service["volumes"]
    assert "finrag.evaluation.full_run" in service["command"]


def eval_container(env: Env) -> Container:
    settings = env.settings.model_copy(
        update={
            "default_tenant_id": EVAL_TENANT_ID,
            "default_collection_id": EVAL_COLLECTION_ID,
            "qdrant_collection": EVAL_QDRANT_COLLECTION,
            "s3_bucket": EVAL_S3_BUCKET,
        }
    )
    return dataclasses.replace(env.container, settings=settings)


def test_a_re_rendered_filing_with_different_bytes_is_not_ingested_twice(
    env: Env, indexer: PipelineFilingIndexer
) -> None:
    first = b"Item 1A. Risk Factors\n\nSupply chains may be disrupted by shortages."
    re_rendered = first + b"\n"  # wkhtmltopdf output is not byte-stable: same Filing, new hash

    assert indexer.index("0000000000-26-000001.pdf", first) is True
    points = len(env.vectors._points)  # type: ignore[attr-defined]
    assert indexer.index("0000000000-26-000001.pdf", re_rendered) is False

    assert eval_documents(env) == 1
    assert len(env.vectors._points) == points  # type: ignore[attr-defined]


def test_job_for_never_reads_another_tenants_document_version(
    env: Env, indexer: PipelineFilingIndexer
) -> None:
    indexer.index("A.pdf", b"Item 1A. Risk Factors\n\nSupply chains may be disrupted.")
    (document,) = env.container.document_service().list_documents(eval_principal())
    other = env.other_tenant()
    foreign = PipelineFilingIndexer(env.container, env.pipeline())
    foreign._principal = other  # type: ignore[attr-defined]

    with pytest.raises(LookupError):
        foreign._job_for(document.id)  # type: ignore[attr-defined]


def test_wiping_the_eval_index_removes_only_the_eval_namespace(
    env: Env, indexer: PipelineFilingIndexer
) -> None:
    indexer.index("A.pdf", b"Item 1A. Risk Factors\n\nSupply chains may be disrupted.")
    other = env.other_tenant()
    env.act_as(other)
    env.client.post(
        "/documents", files={"file": ("o.txt", b"Item 1. Other tenant text.", "text/plain")}
    )
    env.run_jobs()
    other_documents = len(env.container.document_service().list_documents(other))
    other_points = [
        k
        for k in env.vectors._points
        if k[0] == other.tenant_id  # type: ignore[attr-defined]
    ]
    other_blobs = {k for k in env.blobs.objects if k.startswith(f"{other.tenant_id}/")}
    assert other_documents == 1 and other_points and other_blobs

    wipe_eval_index(eval_container(env))

    assert eval_documents(env) == 0
    assert not [k for k in env.vectors._points if k[0] == EVAL_TENANT_ID]  # type: ignore[attr-defined]
    assert not [k for k in env.blobs.objects if k.startswith(f"{EVAL_TENANT_ID}/")]
    with env.container.session_factory() as session:
        for model in (Chunk, DocumentVersion):
            assert not session.query(model).filter(model.tenant_id == EVAL_TENANT_ID).count()
    assert len(env.container.document_service().list_documents(other)) == other_documents
    assert [
        k
        for k in env.vectors._points
        if k[0] == other.tenant_id  # type: ignore[attr-defined]
    ] == other_points
    assert {k for k in env.blobs.objects if k.startswith(f"{other.tenant_id}/")} == other_blobs
    # and the Filing can be ingested again afterwards
    assert indexer.index("A.pdf", b"Item 1A. Risk Factors\n\nSupply chains may be disrupted.")
    assert eval_documents(env) == 1


def test_wiping_refuses_outside_the_eval_environment(env: Env) -> None:
    with pytest.raises(EvalEnvironmentError):
        wipe_eval_index(env.container)
