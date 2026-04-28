# CollusionLab Tech Stack

This document specifies a minimal, low-dependency tech stack for CollusionLab, consistent with the core design principles: low compute, reproducibility, interpretability, and one strong environment before many weak ones.

## 1. Environment and Package Management

CollusionLab uses a Python-first stack with mamba for environment management. There are two environments:

- `collusionlab` – core environment (lightweight, CPU-friendly, no deep learning frameworks required).
- `collusionlab-gpu` – optional environment for local GPU-backed models, not required for core experiments.

### 1.1 Core Environment (`collusionlab`)

- Env manager: `mamba` (or `micromamba`).
- Python version: 3.11.
- Principle: one small `environment.yml` listing core dependencies; avoid database stacks, heavy ML frameworks, and unnecessary services.

The core environment must be able to run:

- The repeated pricing environment.
- Agent wrappers for small, API-accessible LLMs.
- Experiment runners and sweeps, including parallel execution across cores.
- Oversight and metrics.
- The local replay/analysis UI (Streamlit) and analysis notebooks.

### 1.2 Optional GPU Environment (`collusionlab-gpu`)

- Purpose: allow local GPU inference with small models for power users without burdening the main environment.
- Not required for any of the core success criteria or the first milestone.
- Contains only the extra dependencies needed for local model inference (e.g., PyTorch and CUDA tooling), plus the same light dependencies as the core env.

## 2. Repository Structure

The tech stack is aligned with the suggested layout in the main README:

```text
collusionlab/
├── README.md
├── configs/
│   ├── base.yaml
│   ├── pricing_public_chat.yaml
│   ├── pricing_private_chat.yaml
│   └── pricing_audit.yaml
├── src/
│   └── collusionlab/
│       ├── environments/
│       │   ├── base.py
│       │   ├── communication.py
│       │   └── pricing/
│       │       ├── __init__.py
│       │       ├── game.py
│       │       ├── demand.py
│       │       ├── config.py
│       │       ├── metrics.py
│       │       └── signals.py
│       ├── agents/
│       │   ├── model_client.py
│       │   ├── backends/
│       │   │   ├── openai_client.py
│       │   │   └── anthropic_client.py
│       │   ├── memory.py
│       │   └── llm_agent.py
│       ├── runner/
│       │   ├── config.py
│       │   ├── experiment.py
│       │   └── sweep.py
│       ├── auditing/
│       │   ├── base.py
│       │   ├── transcript_auditor.py
│       │   ├── behavior_auditor.py
│       │   └── oversight_manager.py
│       ├── metrics/
│       │   ├── base.py
│       │   ├── collusion.py
│       │   └── concealment.py
│       └── ui/
│           └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── prompts/
│   └── pricing/
│       ├── system.txt
│       ├── message_turn.txt
│       └── action_turn.txt
├── notebooks/
│   └── analysis.ipynb
├── docs/
│   └── design.md
├── tests/
├── environment.yml
├── environment-gpu.yml
├── .env.example
└── .gitignore
```

The tech stack is chosen to support this structure without introducing heavy framework coupling.

## 3. Core Backend Stack

### 3.1 Language and Philosophy

- Language: Python 3.11.
- Philosophy: plain modules, minimal dependencies, and no mandatory deep learning frameworks in the core environment.

Core modules:

- `src/collusionlab/environments/base.py`, `environments/pricing/`
  - Deterministic repeated pricing/auction game with seeded randomness; `GameEnvironment` ABC and registry for future environments.
- `src/collusionlab/agents/llm_agent.py`, `agents/model_client.py`, `agents/memory.py`
  - LLM agent wrapper(s) with prompts, bounded memory, and communication permissions; pluggable `ModelClient` ABC with OpenAI and Anthropic backends.
- `src/collusionlab/runner/experiment.py`, `runner/sweep.py`
  - Command-line runners for single experiments and parameter sweeps, with parallel execution support.
- `src/collusionlab/auditing/transcript_auditor.py`, `auditing/behavior_auditor.py`
  - Lightweight oversight via transcript keyword filters, random audits, and behavior checks on price paths.
- `src/collusionlab/metrics/collusion.py`, `metrics/concealment.py`
  - Metrics such as price elevation vs competitive baseline, price convergence, profit uplift, persistence after audits, and language vs behavior gaps.

### 3.2 Core Python Dependencies

Core environment packages:

- `numpy` – numerics and RNG.
- `pandas` – metrics and tabular analysis (rounds, runs, metrics tables).
- `pyyaml` – reading/writing YAML configs under `configs/`.
- `pydantic` – configuration and run schemas with validation.
- `tenacity` – retry and exponential backoff for API calls during sweeps.
- `python-dotenv` (optional) – environment variable and API key management.

Provider SDKs are **not** core dependencies. Each `ModelClient` backend declares its own SDK requirements (see Section 6). The core environment does not import any provider SDK directly.

Standard library modules heavily used:

- `argparse` – CLIs for `experiment.py` and `sweep.py`.
- `logging` – run logs and system-level logs.
- `random`, `secrets` – base RNG.
- `concurrent.futures` – multicore parallel sweep execution (see Section 8).
- `pathlib`, `os`, `json`, `csv` – filesystem, JSONL/CSV logs, config paths.

### 3.3 Deep Learning Libraries

In the **core `collusionlab` environment**, there are no required deep learning frameworks:

- Not included by default: `torch`, `tensorflow`, `jax`, `cudatoolkit`.
- This follows the "low compute first" design principle and avoids heavy installs on student hardware.

Deep learning libraries, if needed, are isolated into the optional GPU environment.

## 4. Storage, Configs, and Reproducibility

### 4.1 File-Based Storage

For V1, CollusionLab uses file-based storage only:

- `configs/`
  - YAML configs fully define runs: environment type, model, prompts, communication mode, oversight mode, round count, seed, etc.
- `data/raw/`
  - Round-level logs (JSONL) with states, prices/bids, agent messages, actions, rewards, and audit events.
- `data/processed/`
  - Derived metrics tables and summary statistics.

No database is required for the first milestone.

### 4.2 Reproducibility

- Each run must be reproducible from its config and seed.
- Use consistent RNG handling via `random` and `numpy.random`.
- Configs and logs store all seeds and key hyperparameters explicitly.

## 5. Analysis UI and Notebooks

The "Analysis UI" is implemented with Streamlit, plus Jupyter notebooks for deeper analysis.

### 5.1 Streamlit Dashboard

- Location: `src/collusionlab/ui/app.py`.
- Purpose: full local control panel and analysis tool with four top-level pages:
  - **Run page**: YAML config editor with live Pydantic validation, launch button that runs a single experiment in a background thread with live per-round progress (action chart, running cost estimate), and a link to the Analyze page on completion.
  - **Sweep page**: sweep config editor, parameter grid preview table, background sweep execution with live progress bar and per-run status, and a link to the Compare page on completion.
  - **Analyze page (single run)**: config tab (formatted YAML + optional diff against a second run), trajectory tab (action time series with Nash/monopoly reference lines and onset annotation, normalized reward elevation, rolling covert/hollow coordination rates), transcript tab (round-by-round timeline with per-round signals, color-coded flags, inline keyword highlights, filter controls, and full-text search), and metrics tab (all summary metrics including steganographic score, onset/transition rounds, price-follow rate).
  - **Compare page (sweep results)**: threshold heatmap (onset rate by two sweep dimensions), transition view (mean covert/hollow coordination rate over time per condition), filterable run browser (sortable by steganographic score), and CSV/PNG/SVG export.

UI dependencies (core environment):

- `streamlit` – main UI framework (Python-only, minimal boilerplate).
- `plotly` – time series and metric charts integrated natively into Streamlit.

The Streamlit app reads from `configs/`, `data/raw/`, and `data/processed/`, and uses the same core modules for metrics to avoid duplication.

### 5.2 Jupyter Notebooks

- Location: `notebooks/analysis.ipynb`.
- Purpose:
  - Exploratory data analysis.
  - Additional plots for papers.
  - Sanity checks on metrics and oversight behavior.

Notebook dependencies:

- `jupyterlab`.
- `pandas`, `numpy`, `plotly`.

## 6. Model Client Abstraction

To keep provider and GPU choices swappable without touching agent logic, CollusionLab uses a pluggable model client interface.

### 6.1 Model Client Interface

Define a base class (e.g., `ModelClient`) that all backends implement:

```python
class ModelClient:
    def generate(self, messages: list[dict], **kwargs) -> str: ...
```

Agents in `src/agents/llm_agent.py` depend only on this interface. No provider SDK is imported anywhere outside a backend implementation file.

All implementations should wrap calls with `tenacity`-based retry and exponential backoff to handle transient API errors and rate limits gracefully during sweeps.

### 6.2 Implementations

Backends are registered by name in config YAML so the runner can instantiate the right one at startup. Three tiers are planned:

- **`OpenAIModelClient`** (core environment, primary for V1):
  - Uses the `openai` Python SDK.
  - Default model: `gpt-4o-mini` — small, cheap, strong instruction-following.

- **`AnthropicModelClient`** (core environment, secondary):
  - Uses the `anthropic` Python SDK.
  - Used for cross-model comparison runs (e.g., Haiku or Sonnet).

- **`LocalGpuModelClient`** (optional, GPU environment):
  - Lives in `collusionlab-gpu` and uses a local inference runtime (e.g., vLLM or a `transformers` pipeline).
  - Only this module imports heavy GPU or DL dependencies.

Each backend's SDK (`anthropic`, `openai`, etc.) is declared as a dependency of that backend only, not of the core environment. For V1, install whichever backends you intend to use alongside the core env.

This design means swapping or adding a provider requires only adding one new file and one new entry in `environment.yml` — no changes to agent, runner, or metrics code.

## 7. Environments Summary

### 7.1 Core Environment (`collusionlab`)

Goals:

- Runs on a laptop with no GPU, minimal install friction.
- Satisfies all recommended first milestone requirements:
  - One repeated pricing environment.
  - One small model family (API-based).
  - Communication modes: none, public, private.
  - Oversight modes: none, audit-plus-penalty.
  - Full logging and replay.
  - One empirical study on collusion onset and oversight effects.

Key packages:

- Python 3.11
- `numpy`, `pandas`
- `pyyaml`, `pydantic`
- `tenacity`
- `streamlit`, `plotly`
- `jupyterlab`
- Provider SDKs installed alongside core env as needed (e.g., `anthropic`, `openai`)

### 7.2 GPU Environment (`collusionlab-gpu`)

Goals:

- Optional, for local GPU inference with small models.
- Does not affect reproducibility of core experiments, which should be possible with remote APIs alone.

Key additions:

- Deep learning stack (e.g., `pytorch` + `cudatoolkit`, or another runtime), chosen later.
- The same light dependencies as the core env for shared code.

## 8. Sweep Parallelism

Parameter sweeps must run in parallel. The design targets two tiers:

### 8.1 Laptop-Level Parallelism (V1)

- Implementation: `concurrent.futures.ProcessPoolExecutor` from the Python standard library.
- Each worker runs one experiment config as a subprocess-isolated unit; workers share no mutable state.
- The sweep coordinator in `src/runner/sweep.py` maps configs to the pool and collects results.
- The number of workers defaults to the available CPU core count but is configurable.
- This requires no additional dependencies and runs out of the box on a MacBook or Windows laptop.

**Rate limiting:** with many workers hitting the same provider simultaneously, API rate limits will be triggered. Each `ModelClient` implementation must wrap calls with `tenacity` retry + exponential backoff. Additionally, the sweep coordinator should expose a configurable `--max-workers` flag so the user can throttle concurrency per provider tier (e.g., free-tier APIs need fewer workers than paid tiers).

### 8.2 HPC / Distributed Path (Extension)

- The sweep interface is designed so that the executor backend is swappable.
- Candidate orchestrators: **Ray** (simple drop-in for the `concurrent.futures` interface), **Dask**, or a SLURM job array for traditional HPC clusters.
- Adding HPC support should only require changing the executor implementation in `sweep.py`, not the experiment logic.
- This upgrade path is not required for V1 but should be achievable without rewriting the runner.

## 9. Testing

CollusionLab uses Python's built-in `unittest` (or `pytest` as a drop-in runner) with no
additional test framework dependencies in the core environment.

- **Fixtures:** synthetic JSONL logs and minimal configs are stored under `tests/fixtures/` and
  used to test the metrics and runner layers without real API calls.
- **API mocking:** `unittest.mock.patch` is used to stub `ModelClient.generate()` in runner and
  agent tests so no network calls are made.
- **Determinism tests:** reproducibility is verified by running the same config twice and
  comparing JSONL output byte-for-byte.
- **No heavy test infra:** no test database, no containers, no CI-only services. Tests must pass
  on a laptop with only the core environment installed.

Each implementation phase has a corresponding test file under `tests/` (see `plan.md`).
