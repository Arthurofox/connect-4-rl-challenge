# Repository Guidelines

## Project Structure & Module Organization
Runtime code lives directly under `src/`. Core files: `src/board.py` (game mechanics), `src/env.py` (Gymnasium wrapper), `src/match.py` (match orchestration), `src/train.py` (training pipelines), and `src/evaluate.py` (checkpoint checks). Keep policy implementations inside `src/agents/` (for example `random_policy.py`, `sb3_agent.py`). Store reusable configs in `configs/`—use `configs/base.yaml` for baselines and `configs/ppo.yaml` to kick off PPO with reward shaping or checkpoint resume settings. Place light assets in `assets/`, and commit only metadata or logs under `artifacts/<date>-<run>/`. Align `tests/` with module names (`tests/unit/test_board.py`, `tests/unit/test_env.py`, `tests/integration/test_random_agent_match.py`) so coverage tracks the code.

## Build, Test, and Development Commands
`python -m venv venv && source venv/bin/activate` prepares the environment; follow with `uv pip install -r requirements.txt` to sync dependencies. Run `python -m pytest` for the default suite or add `--maxfail=1 --ff` for quicker iteration. Launch random baselines with `python -m src.train --config configs/base.yaml --render`. For PPO, use `python -m src.train --config configs/ppo.yaml --algo ppo --timesteps 50000 --save-checkpoint artifacts/released/ppo_latest.zip`. Try the curriculum profile with `python -m src.train --config configs/ppo_curriculum.yaml --algo ppo --tensorboard-log artifacts/tensorboard/curriculum`. Resume from a saved model with `--load-checkpoint artifacts/released/ppo_latest.zip --resume`. Evaluate checkpoints via `python -m src.evaluate --checkpoint artifacts/released/ppo_latest.zip --algo ppo --games 50 --render` (append `--stochastic` to sample actions). Reward shaping lives in the `rewards` block of each config; override on the CLI with flags like `--reward-step -0.05`.

## Coding Style & Naming Conventions
Target Python 3.11+, use 4-space indentation, and export clean module-level APIs from each script. Format with `black src tests` and lint with `ruff check src tests` before committing; keep shared settings in `pyproject.toml`. Stick to snake_case for functions/modules, PascalCase for classes, SCREAMING_SNAKE_CASE for constants, and kebab-case filenames for config variants like `baseline-aggressive.yaml`.

## Testing Guidelines
Pytest drives validation: place fast unit specs under `tests/unit/` and full games or integration flows under `tests/integration/`. Name tests `test_<unit>_<behavior>`, seed randomness in fixtures, and run `pytest --cov=src --cov-report=term-missing` when gauging coverage (target ≥85%). Add smoke tests that load any new checkpoint format and execute at least one full match.

## Commit & Pull Request Guidelines
Prefer Conventional Commits (`feat(agent): add epsilon-greedy policy`) with tight scopes. Reference issues or experiment notebooks in commit bodies, and summarize training time, seeds, and win-rate deltas in PR descriptions. Pull requests should list reproduction commands, new configs or assets, and links to externally hosted checkpoints; respond to review feedback with follow-up commits instead of force-pushes.

## Security & Configuration Tips
Keep API keys, opponent artifacts, and raw checkpoints out of Git; rely on `.env` + `python-dotenv` for local secrets. Archive released agents in `artifacts/released/` and include a README noting config, seed, and evaluation metrics. Validate any third-party resources against competition rules before adding them.
