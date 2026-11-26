# AI Usage

We used **Claude Code** (an AI coding assistant) during this project, as the course AI policy allows.

## What we did

- Chose the problem, dataset (PTB-XL), task, metric (macro-F1) and the overall pipeline design.
- Wrote the data preparation, feature extraction, AutoML training, ensemble and promotion gate, MLflow tracking, FastAPI service, `/stats` monitoring, `/model` endpoint, linting setup and tests.
- Made the Milestone 2 decisions: Docker on a single AWS EC2 instance, GHCR for images, rollback by image tag.
- Set up and operated the AWS project and the EC2 deployment, and checked the live service (`/health`, `/predict`, `/explain`, `/stats`), including a rollback rehearsal.

## What AI assisted with

- **Milestone 1:** we did most of the work ourselves (data preparation, features, AutoML training, MLflow, API). Claude Code helped with parts of the feature code, tests and documentation drafts, and reviewed our work.
- **Milestone 2:**
  - Dockerfile
  - Web UI
  - EC2 deploy and rollback scripts
  - Markdown documentation (README, DEPLOYMENT.md, RUNBOOK.md)

## Review

We read, ran and tested every AI-assisted change before keeping it.
