# Contribution Guide

## Team

| Member | ERP ID | Role |
|--------|--------|------|
| Muhammad Jawad Maqsood | `27080` | Data preparation, model training, MLflow; Milestone 2 Docker/CI, AWS deployment and model improvement |
| Sudharth Kumar | `26925` | FastAPI service, tests, documentation; Milestone 2 linting, monitoring, web UI and documentation |

## Milestone 1: Model & API (`v1.0-milestone1`)

| Task | Owner |
|------|-------|
| Data preparation and validation (`prepare_data.py`, `validate_signal`) | Muhammad Jawad Maqsood |
| Feature extraction (467 features) and FLAML AutoML training (`train.py`) | Muhammad Jawad Maqsood |
| MLflow tracking and model registry | Muhammad Jawad Maqsood |
| FastAPI service (`/predict`, `/explain`, `/health`) | Sudharth Kumar |
| Tests (pytest) | Sudharth Kumar |
| Documentation (README, slides) | Sudharth Kumar |

## Milestone 2: Containerization, CI/CD & Deployment (`v2.0-milestone2`)

| Task | Branch / PR | Owner |
|------|-------------|-------|
| Pinned, split requirements; model loads from the committed file when MLflow is absent | `fix/model-fallback` | Muhammad Jawad Maqsood |
| Ruff configuration and lint fixes | `infra/lint` | Sudharth Kumar |
| Monitoring: `/stats` request count and latency, structured request logs | `feat/stats` | Sudharth Kumar |
| Dockerfile with `/health` healthcheck; GitHub Actions CI (lint, test, Docker build, GHCR push); `pytest.ini` fix for CI | `infra/docker-ci` | Muhammad Jawad Maqsood |
| AWS EC2 deployment scripts (provision, setup, deploy with automatic rollback, destroy), DEPLOYMENT.md, RUNBOOK.md, live deployment and rollback rehearsal | `infra/deploy` | Muhammad Jawad Maqsood |
| README and AI usage updates for Milestone 2 | `docs/milestone2` | Sudharth Kumar |
| Web UI and `/model` endpoint (accuracy and training date) | `feat/web-ui` | Sudharth Kumar |
| Labelled held-out test ECGs for the web UI | `feat/web-ui` | Muhammad Jawad Maqsood |
| Project focus: validated ensemble and promotion gate in training; retrained model (72.85% accuracy, macro-F1 0.640) | `feat/retrain-ensemble` | Muhammad Jawad Maqsood |
| Project Focus section and video walkthrough in README | `feat/retrain-ensemble` | Muhammad Jawad Maqsood, Sudharth Kumar |
| Peer review of the other member's pull requests (one approval per PR) | all branches | Both |

AI assistance is disclosed in [AI_USAGE.md](AI_USAGE.md).

### Reflections (Milestone 2)

**Muhammad Jawad Maqsood:** I containerised the API, set up CI and deployed it to AWS EC2 with a health-checked
deploy script that rolls back by itself, which I rehearsed on the live server. I learned that a pipeline needs gates
as much as automation: CI only caught real problems once its test command matched what runs locally, and retraining
needed a promotion gate so a worse model could never replace a better one. I also learned how account-level policies
(the single-Region rule of our AWS project) can block a deployment that looks correct.

**Sudharth Kumar:** I added Ruff linting, a `/stats` endpoint with request count, latency and structured request logs,
and a web UI with a `/model` endpoint that shows the model's accuracy and training date. I also updated the README and
AI usage notes. I learned that monitoring and linting are worth setting up early: structured logs made it easy to see
what the API was doing, and the lint check in CI caught style problems before they reached `main`.

## Branching

- `main` is always runnable. Work happens on feature branches:
  - `feat/<topic>` new functionality
  - `fix/<topic>` bug fixes
  - `infra/<topic>` CI, Docker, deployment
  - `docs/<topic>` documentation
- Every pull request to `main` needs **one peer approval** (from the other team member) and a green CI run (lint, test, build).
- Pull requests are merged with **"Create a merge commit"** (the Milestone 2 branches are stacked on each other).
- Commits are small and spread across the semester.

## Workflow

1. `make setup` (or `pip install -r requirements-dev.txt`) to install dependencies.
2. `python manage.py data` then `python manage.py train` to rebuild features and retrain. A new model is promoted to
   `models/` only if it beats the served one (see README, Pipeline).
3. `make test` (plain `pytest`, same as CI) and `make lint` (`ruff check .`) before opening a pull request.
4. `make docker-build docker-run` to check the container locally (`docker ps` shows `healthy`).
5. After merging to `main`, CI pushes `ghcr.io/jawad1569/ecganomaly:<short-sha>`; deploy with `~/deploy.sh <short-sha>`
   on the server (see [RUNBOOK.md](RUNBOOK.md)).
