# Repository Guidelines

## Project Structure & Module Organization
Runtime code lives directly under `src/`. Use `src/board.py` for game mechanics, `src/match.py` for match orchestration, `src/train.py` for training pipelines, and `src/evaluate.py` for submission checks. Keep policy implementations inside `src/agents/` (multiple strategies welcome). Store reusable configs in `configs/`, light assets in `assets/`, and commit metadata—not large checkpoints—under `artifacts/<date>-<run>/`. Align `tests/` with the module names (`tests/unit/test_board.py`, `tests/integration/test_random_agent_match.py`) so coverage stays close to the code.

## Build, Test, and Development Commands
`python -m venv venv && source venv/bin/activate` prepares the environment, followed by `pip install -r requirements.txt` to install tooling. Run `python -m pytest` for the default suite or add `--maxfail=1 --ff` for quicker iteration. Launch training with `python -m src.train --config configs/base.yaml --episodes 1000`, and sanity-check agents via `python -m src.evaluate --checkpoint artifacts/sample.ckpt --games 50` before freezing a build.

## Coding Style & Naming Conventions
Target Python 3.11+, use 4-space indentation, and export clean module-level APIs from each script. Format with `black src tests` and lint with `ruff check src tests` prior to commits; tweak settings in `pyproject.toml` if rules evolve. Stick to snake_case for functions/modules, PascalCase for classes, SCREAMING_SNAKE_CASE for constants, and kebab-case filenames for config variants (for example `baseline-aggressive.yaml`).

## Testing Guidelines
Pytest drives validation: place fast unit specs under `tests/unit/` and full games or integration flows under `tests/integration/`. Name tests `test_<unit>_<behavior>`, seed randomness in fixtures, and run `pytest --cov=src --cov-report=term-missing` when gauging coverage (target ≥85%). Add smoke tests that load any new checkpoint format and execute at least one full match.

## Commit & Pull Request Guidelines
Prefer Conventional Commits (`feat(agent): add epsilon-greedy policy`) with tight scopes. Reference issues or experiment notebooks in commit bodies, and summarize training time, seeds, and win-rate deltas in PR descriptions. Pull requests should list reproduction commands, new configs or assets, and links to externally hosted checkpoints; respond to review feedback with follow-up commits instead of force-pushes.

## Security & Configuration Tips
Keep API keys, opponent artifacts, and raw checkpoints out of Git; rely on `.env` + `python-dotenv` for local secrets. Archive released agents in `artifacts/released/` and include a README noting config, seed, and evaluation metrics. Validate any third-party resources against competition rules before adding them.
