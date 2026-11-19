.DEFAULT_GOAL := help
.PHONY: help install build up down logs migrate reconcile dev lint fmt test test-int ci eval eval-fetch eval-render eval-index eval-retrieval eval-dry-run eval-compare eval-bless eval-image check-eval-image check-docker-memory check-image

help: ## list targets
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: ## uv sync + install the pre-commit hooks (shared by all git worktrees)
	uv sync
	uv run pre-commit install

build: ## build the app image
	docker compose build

check-image: ## prove the built image loads every model offline (run `make build` first)
	docker run --rm -i finrag-app:local python - < infra/docker/check_offline_models.py

up: ## build and start the whole stack (needs FINRAG_VOYAGE_API_KEY, or FINRAG_EMBEDDER_BACKEND=hash FINRAG_SPARSE_BACKEND=hash)
	docker compose up -d --build --wait

down: ## stop the stack
	docker compose down

logs: ## follow api and worker logs
	docker compose logs -f --tail=100 api worker

migrate: ## apply Alembic migrations + Procrastinate schema (needs `make up`)
	uv run alembic upgrade head

reconcile: ## re-defer ingestion jobs of Documents stuck by a crash (also runs at worker start)
	uv run python -m finrag.jobs.reconcile

dev: ## run the API with reload on :8000
	uv run uvicorn finrag.api.app:app --reload --port 8000

lint: ## ruff check + format check + mypy + import layering
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run lint-imports

fmt: ## auto-fix lint and format
	uv run ruff check --fix .
	uv run ruff format .

test: ## unit tests (no network, no docker)
	uv run pytest

test-int: ## integration tests incl. real Docling (needs `make up`)
	uv run pytest -m integration

ci: ## wait for GitHub CI on the pushed HEAD; non-zero if any job fails (Linux results differ from macOS)
	@sha=$$(git rev-parse HEAD); id=""; \
	for i in 1 2 3 4 5 6; do \
	  id=$$(gh run list --commit $$sha --limit 1 --json databaseId --jq '.[0].databaseId // empty'); \
	  [ -n "$$id" ] && break; sleep 5; \
	done; \
	[ -n "$$id" ] || { echo "No CI run for $$sha: push first"; exit 1; }; \
	gh run watch $$id --interval 20 --exit-status

eval-fetch: ## download the pinned 10-K filings into evals/cache/ and verify checksums (SEC EDGAR)
	uv run python -m finrag.evaluation.fetch

check-docker-memory: ## refuse when the Docker VM has under 8 GB (eval-index needs it)
	@uv run python -m finrag.evaluation.resources

eval-image: ## build the Linux `eval` image (app image plus the PDF renderer)
	docker build --target eval -t finrag-eval:dev .

check-eval-image: eval-image ## prove the renderer works offline and re-renders to the same text
	docker run --rm --network none -i finrag-eval:dev python - < infra/docker/check_render.py

eval-render: eval-image ## render cached filings to PDF in the eval image, offline; up-to-date PDFs are skipped
	docker run --rm --network none --user $$(id -u):$$(id -g) -v "$(CURDIR)/evals:/evals" finrag-eval:dev \
	  python -m finrag.evaluation.render --manifest /evals/manifest.json --cache-dir /evals/cache

eval-index: check-docker-memory eval-image ## ingest the rendered filings into the eval tenant; REBUILD=1 first wipes only the eval namespace (needs `make up`, `make eval-render`; real filings need FINRAG_VOYAGE_API_KEY)
	mkdir -p evals/results
	docker compose --profile eval run --rm --no-deps eval-index python -m finrag.evaluation.index \
	  --manifest /evals/manifest.json --cache-dir /evals/cache $(if $(REBUILD),--rebuild)

eval-retrieval: eval-image ## retrieval-only metrics over the real eval index (needs `make up`, `make eval-index`; embeddings only, no paid answer calls)
	mkdir -p evals/results
	FINRAG_EVAL_IMAGE_DIGEST=$$(docker image inspect --format '{{.Id}}' finrag-eval:dev) \
	FINRAG_EVAL_GIT_SHA=$$(git rev-parse --short HEAD) \
	docker compose --profile eval run --rm --no-deps eval-retrieval

eval-dry-run: eval-image ## PAID: answer+judge up to 5 questions, print measured tokens and cost (ARGS="--limit 8 --cost-cap-usd 2"; more than 5 needs a cap)
	docker compose --profile eval run --rm --no-deps eval-dry-run python -m finrag.evaluation.dry_run --golden-dir /evals/golden \
	  --manifest /evals/manifest.json --cache-dir /evals/cache --results-dir /evals/results $(ARGS)

eval: eval-image ## PAID: full run (retrieval, answers, judge); a cost cap is mandatory: make eval COST_CAP_USD=20
	@test -n "$(COST_CAP_USD)" || { echo "make eval refused: set a per-run cost cap, e.g. make eval COST_CAP_USD=20 (no default)" >&2; exit 1; }
	mkdir -p evals/results
	FINRAG_EVAL_IMAGE_DIGEST=$$(docker image inspect --format '{{.Id}}' finrag-eval:dev) \
	FINRAG_EVAL_GIT_SHA=$$(git rev-parse --short HEAD) \
	FINRAG_EVAL_COST_CAP_USD=$(COST_CAP_USD) \
	docker compose --profile eval run --rm --no-deps eval-full

eval-compare: ## delta of a results file (RESULTS=<file>, default newest) vs evals/baseline.json as a Markdown table; always exits 0
	uv run python -m finrag.evaluation.compare $(if $(RESULTS),--results $(RESULTS))

eval-bless: ## explicitly make RESULTS=<file> the new evals/baseline.json (ALLOW_RETRIEVAL_ONLY=1 permits a retrieval-only file)
	@test -n "$(RESULTS)" || { echo "make eval-bless refused: say which run, e.g. RESULTS=evals/results/full-<time>-<sha>.json" >&2; exit 1; }
	uv run python -m finrag.evaluation.bless $(RESULTS) $(if $(ALLOW_RETRIEVAL_ONLY),--allow-retrieval-only)
