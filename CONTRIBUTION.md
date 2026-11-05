# Contribution Guide

## Team

| Member | Role |
|--------|------|
| Muhammad Jawad Maqsood | Data preparation, model training, MLflow |
| Sudharth Kumar | FastAPI service, tests, documentation |

## Branching

- `main` is always runnable. Work happens on feature branches:
  - `feat/<topic>` new functionality
  - `fix/<topic>` bug fixes
  - `infra/<topic>` CI, Docker, deployment
- Every pull request to `main` needs **one peer approval**.
- Commits are small and spread across the semester.

## Workflow

1. `make setup` to install dependencies.
2. `python manage.py train` to train and register the model (FLAML AutoML + MLflow).
3. `python manage.py test` (or `make test`) before opening a pull request.
4. `make lint` / `make format` keep ruff and black clean.
