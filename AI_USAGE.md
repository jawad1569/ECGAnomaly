# AI Usage

We used **Claude Code** (an AI coding assistant) during this project, as the course AI policy allows.
This file lists what it was used for and what we did ourselves.

## What we did

- Chose the problem, dataset (PTB-XL), task, metric (macro-F1) and the overall pipeline design.
- Ran data preparation and the AutoML training runs, compared them in MLflow and picked the best model.
- Made the Milestone 2 decisions: Docker on a single AWS EC2 instance, GHCR for images, rollback by image tag,
  `/stats` for monitoring.
- Set up and operated the infrastructure: development environment, AWS project and access, the EC2 deployment,
  and checking the live service (`/health`, `/predict`, `/explain`, `/stats`).
- Reviewed, ran and tested all generated code, including a rollback rehearsal on the live server.

## What AI assisted with

- **Milestone 1:** the extended 467-feature set and FLAML AutoML training code, `validate_signal`, tests, and
  documentation drafts.
- **Milestone 2:** drafting the Dockerfile, the GitHub Actions workflow, the `/stats` monitoring middleware, the
  model-file fallback in the loader, the EC2 deploy and rollback scripts, the web UI (`src/api/static/`), the `/model` endpoint, the test-sample export script, and drafts
  of README, DEPLOYMENT.md and RUNBOOK.md.
- Debugging environment issues (Git ownership after an OS reinstall, AWS Region restrictions) and lint fixes.

## Review

We read and ran every AI-suggested change before keeping it. The test suite passes, metrics come from the held-out
PTB-XL fold 10, and the deployment was verified on the live endpoint. Every team member can explain the code and
the design decisions behind it.
