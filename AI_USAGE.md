# AI Usage Disclosure

AI assistance (Claude Code) was used in this project for:

- **Model improvement:** replacing the 108 basic features with a 467-feature set (spectral,
  heart-rate, median-beat template, lead correlations), training on the full PTB-XL dataset and
  adding FLAML AutoML (`src/pipeline/train.py`, `src/utils/features.py`).
- **Input validation:** `validate_signal` used by the API and the training pipeline.
- **Tests and documentation:** `tests/test_pipeline.py`, `CONTRIBUTION.md`, this file and the
  Milestone 1 presentation.

All generated code was run and checked by the team: the pytest suite passes and the reported
metrics come from the held-out PTB-XL fold 10, not from the training data.
