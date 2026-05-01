# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

CollusionLab — research platform for studying onset and concealment of collusion in repeated LLM multi-agent games. Default environment is a Calvano-style repeated pricing game; design is meant to extend to other repeated games. Read `README.md` for the research framing and `docs/plan.md` for the phased implementation roadmap (Phases 0–4 done, Phase 5 done, Phase 6+ pending as of this writing).

## Environment & commands

The conda env is named `collusion_lab` (underscore). The package is **not** pip-installed — there is no `pyproject.toml` / `setup.py` despite what the README suggests. This affects how you invoke things:

- **Tests**: work as-is. They do `sys.path.insert(0, ROOT/"src")` at the top of each test file.
  ```bash
  mamba run -n collusion_lab pytest tests/
  mamba run -n collusion_lab pytest tests/test_runner.py::test_experiment_log_schema
  ```
- **CLI / scripts**: need `PYTHONPATH=src`.
  ```bash
  PYTHONPATH=src mamba run -n collusion_lab python -m collusionlab.runner.experiment --config configs/base.yaml
  ```
- `scripts/smoke_run.py` does its own `sys.path.insert`, so it runs without `PYTHONPATH`.
- Real-LLM smoke runs need `.env` with `OPENAI_API_KEY` (loaded via `python-dotenv`).

If recurring `PYTHONPATH` friction becomes an issue, the right fix is a minimal `pyproject.toml` with `tool.setuptools.packages.find` over `src/`.

## Architecture (the parts that span files)

### Registry-based pluggability

Three independent registries, each populated by side-effect imports. Adding a new variant is "write the class + register it"; no edits to the runner or config schema.

- **Environments** (`environments/base.py`): `register_environment(env_type, game_cls, config_cls)`. `environments/__init__.py` imports `pricing` to seed the registry. `ExperimentConfig` discovers the right `EnvironmentConfig` subclass via `get_environment_classes(env_type)` in a `@model_validator(mode="before")` — this is how YAML deserializes polymorphically without a static discriminated union.
- **Backends** (`agents/backends/`): each backend module self-registers under a string key (`"openai"`, `"anthropic"`, `"scripted"`). `ScriptedModelClient` is the deterministic test backend — it pops replies from a list, which is what makes `test_experiment_reproducibility_byte_identical` possible.
- **Communication handlers** (`environments/communication.py`): `get_comm_handler(mode)` → `NoCommunication` for Phase 3; public/private land in Phase 4.

### Polymorphic env config serialization

`ExperimentConfig.environment` is annotated `SerializeAsAny[EnvironmentConfig]`. Without `SerializeAsAny`, pydantic v2 dumps using the static base type and silently strips subclass-only fields (e.g. `demand_params`, `nash_price`) from the manifest. `test_config_from_yaml_round_trips` has an explicit regression check on `demand_params`. If you add new env subclasses, do not change this annotation.

### Env-agnostic runner

The runner never imports `pricing` directly. It reaches into the env via:
- `env.system_prompt_vars(agent_id) -> dict` for prompt template variables
- `env.parse_action`, `env.default_action`
- `env.reward_elevation_baseline() -> (nash_profit, monopoly_profit) | None` for the `reward_elevation` trajectory signal

Anything pricing-specific belongs behind these methods, not in `runner/experiment.py`.

### Output layout & manifest

Every run writes `{output_dir}/{run_id}/log.jsonl` (one JSON object per round) and `manifest.json` (config + per-agent token/cost totals + start/end times). The JSONL schema is pinned by `test_experiment_log_schema` — exactly these keys: `run_id, env_type, round, actions, rewards, cumulative_rewards, observations, messages, audit_event, trajectory_signals, reasoning` (per-agent internal action-turn text, private to each agent in-loop). Phase 3 `trajectory_signals` is exactly `{action_spread, reward_elevation}`; auditor-dependent flags are deferred to Phase 4.

`messages` shape per comm mode is pinned: `none → []`, `public → to: "all"`, `private → to: <agent_id>`.

### Sweep runner

`runner/sweep.py` provides `SweepConfig` (grid or list mode) and `SweepRunner` (parallel execution via `ProcessPoolExecutor`). Key contracts:

- **Override syntax**: strict dot-path (`environment.seed`, `agents.0.temperature`). Unknown paths and type mismatches raise `ValueError`. `int` ↔ `float` is allowed.
- **Failure policy**: continue-on-error. Failed runs are recorded as `status: "failed"` in `sweep_manifest.json`; the sweep completes.
- **Output**: `{output_dir}/sweep_{sweep_id}/sweep_manifest.json` with per-run `status`, `error`, timestamps, config snapshot, and `manifest_path`.
- **Windows spawn safety**: `_init_worker_path` initializer propagates `sys.path` to worker processes so the sweep works without pip-installing the package.
- **Determinism criterion**: set-equivalence across worker counts (same configs + same per-run outputs; completion order may differ).

### Reproducibility contract

Same config + same seed + scripted backend → byte-identical `log.jsonl`. Real LLM calls are non-deterministic, so reproducibility tests must use `ScriptedModelClient`.

## Conventions worth knowing

- Token accounting in `ScriptedModelClient` is `len(reply) // 4` — short replies (e.g. `"8"`) yield `output_tokens == 0`. Tests assert `>= 0`, not `> 0`.
- `OversightConfig` and `CommunicationHandler` ship as null stubs in Phase 3; Phase 4 swaps in real implementations via the registries without runner edits.
- `configs/base.yaml` may contain a free-form `_calibration_note` key under `environment` — `ExperimentConfig.from_yaml` strips it before validation. Don't propagate it into other code paths.
