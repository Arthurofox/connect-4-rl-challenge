# Repository Guidelines

## Project Structure & Module Organization
Runtime code lives directly under `src/`. Core files: `src/board.py` (game mechanics + bitboards), `src/env.py` (Gymnasium wrapper), `src/match.py` (match orchestration), `src/train.py` (training CLI for PPO/MuZero), and `src/evaluate.py` (checkpoint evaluation). Keep policy implementations under `src/agents/` (`random_policy.py`, `sb3_agent.py`, `muzero/`, `alphabeta.py`). Store configs in `configs/`—baseline (`base.yaml`), PPO variants (`ppo.yaml`, `ppo_curriculum.yaml`, `ppo_selfplay.yaml`), plus MuZero profiles (`muzero.yaml` for full runs, `muzero_sanity.yaml`/`muzero_debug.yaml` for quick checks). Tests mirror modules (`tests/unit/test_board.py`, `tests/unit/test_env.py`, `tests/unit/test_network_shapes.py`, `tests/integration/test_selfplay_smoke.py`, etc.).

## Build, Test, and Development Commands
`python -m venv venv && source venv/bin/activate` prepares the environment; follow with `uv pip install -r requirements.txt` to sync dependencies. Run `python -m pytest` for the default suite or add `--maxfail=1 --ff` for quicker iteration. Launch random baselines with `python -m src.train --config configs/base.yaml --render`. For PPO, use `python -m src.train --config configs/ppo.yaml --algo ppo --timesteps 50000 --save-checkpoint artifacts/released/ppo_latest.zip`. Try `configs/ppo_curriculum.yaml` for curriculum runs, enable PPO self-play with `configs/ppo_selfplay.yaml --self-play`, and drive MuZero via `python -m src.train --config configs/muzero.yaml --algo muzero --iters 300 --mcts-simulations 800 --save-checkpoint artifacts/released/muzero_latest.pt`. Evaluate checkpoints with `python -m src.evaluate --checkpoint <path> --algo <ppo|muzero> --games 200 --opponent alphabeta --opponent-depth 9 --render` (append `--stochastic` to sample actions). Reward shaping lives in the `rewards` block; override via CLI flags like `--reward-step -0.05`. Illegal moves auto-forfeit for the acting agent.

## Coding Style & Naming Conventions
Target Python 3.11+, use 4-space indentation, and export clean module-level APIs from each script. Format with `black src tests` and lint with `ruff check src tests` before committing; keep shared settings in `pyproject.toml`. Stick to snake_case for functions/modules, PascalCase for classes, SCREAMING_SNAKE_CASE for constants, and kebab-case filenames for config variants like `baseline-aggressive.yaml`.

## Testing Guidelines
Pytest drives validation: place fast unit specs under `tests/unit/` and full games or integration flows under `tests/integration/`. Name tests `test_<unit>_<behavior>`, seed randomness in fixtures, and run `pytest --cov=src --cov-report=term-missing` when gauging coverage (target ≥85%). Add smoke tests that load any new checkpoint format and execute at least one full match.

## Commit & Pull Request Guidelines
Prefer Conventional Commits (`feat(agent): add epsilon-greedy policy`) with tight scopes. Reference issues or experiment notebooks in commit bodies, and summarize training time, seeds, and win-rate deltas in PR descriptions. Pull requests should list reproduction commands, new configs or assets, and links to externally hosted checkpoints; respond to review feedback with follow-up commits instead of force-pushes.

## Security & Configuration Tips
Keep API keys, opponent artifacts, and raw checkpoints out of Git; rely on `.env` + `python-dotenv` for local secrets. Archive released agents in `artifacts/released/` and include a README noting config, seed, and evaluation metrics. Validate any third-party resources against competition rules before adding them.

## TODO
- Parallelize MuZero self-play (`--num-workers`) using torch.multiprocessing.
- Implement temperature annealing schedules (`--temperature-anneal`).
- Add gating to keep the best MuZero checkpoint vs. AlphaBeta depth 7.
- Expand AlphaBeta opponent options in configs (depth presets).
- Refine replay sampling (prioritized or balanced terminal states).
- Instrument replay/value targets to ensure correct perspective and improve training stability.

## Current Status
- Latest MuZero long run (400–800 sims) remains effectively random: 0/200 vs alpha-beta depth 9 and 1/50 vs random.
- Quick debug config (`muzero_debug.yaml`) also fails to score, confirming further training logic review is required.
