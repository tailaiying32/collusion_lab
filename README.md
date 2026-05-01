# CollusionLab

A unified, low-compute research infrastructure for studying how LLM agents respond
to oversight in repeated pricing games.

## Research Question

Prior work (e.g. [Agrawal et al., 2025](https://arxiv.org/abs/2507.01413))
establishes that LLM agents will collude when given communication channels.
CollusionLab addresses the follow-on question:

> When oversight penalizes explicit coordination, do LLM agents cease colluding —
> or do they adapt, sustaining above-Nash profits while eliminating collusion
> language from their transcripts?

This decomposes into two linked subproblems:

- **Onset** — which conditions (communication access, repeated interaction,
  memory, agent count) produce collusion. Tested under Calvano logit demand
  (symmetric to prior work) and Bertrand winner-take-all demand (where the
  Bertrand paradox makes tacit collusion especially notable).
- **Concealment** — the novel contribution: whether probabilistic auditing and
  profit penalties eliminate collusion, or instead produce *covert coordination*
  (clean language, above-Nash profits) or *hollow coordination* (collusion
  language persists; profits return to Nash).

## Platform

CollusionLab provides the infrastructure layer that prior collusion studies have
each had to reimplement from scratch: a pricing environment, an agent loop, a
communication layer, an auditing pipeline, and a persistence backend — all as
composable, config-driven building blocks:

- **Pluggable demand models** — Calvano logit or Bertrand winner-take-all;
  additional market structures can be added without modifying agent or runner code
- **Pluggable LLM backends** — OpenAI, Anthropic, and DeepSeek are supported
  behind a single `ModelClient` interface; local GPU inference is available via
  the optional environment
- **Pluggable oversight mechanisms** — keyword auditing, behavioral signals,
  temporal pattern detection, and LLM-judge auditing are all orchestrated by
  a single `OversightManager`
- **Pluggable storage backends** — flat JSONL files, SQLite, or PostgreSQL,
  all behind a single `RunStore` interface configured via one environment variable
- **Parallel sweep runner** — parameter sweeps execute with full config
  snapshots, continue-on-error behavior, and aggregate manifests
- **Reproducible replay** — every run is fully recoverable from its seed and
  config; the Streamlit dashboard supports transcript inspection from flat files
  or a database

The Agrawal et al. double-auction study is an example of an experiment
expressible as a CollusionLab config. The goal is to make collusion experiments
cheap to define, run, and compare across conditions.

## Comparison to Prior Work

| | Agrawal et al. (2025) | CollusionLab |
|---|---|---|
| **Core question** | Do agents collude? | Do agents adapt under oversight? |
| **Market structure** | Continuous double auction | Calvano logit + Bertrand (pluggable) |
| **Oversight** | Binary (CEO urgency / overseer) | Probabilistic audit + penalty; LLM judge or behavioral |
| **Detection** | Pricing metrics | Multi-signal auditing layer (transcript, behavioral, temporal, LLM judge, risk fusion) |
| **Storage** | None | SQLite or PostgreSQL via unified `RunStore` protocol |
| **Models tested** | GPT-4.1, Claude Sonnet | Small/affordable models (gpt-4o-mini, DeepSeek) |
| **Artifact** | Workshop paper | Open-source research infrastructure |

## Architecture

Six modules, cleanly separated:

| Module | Location | Purpose |
|---|---|---|
| Environment engine | `src/collusionlab/environments/` | Repeated pricing games; Calvano and Bertrand demand models |
| Agent layer | `src/collusionlab/agents/` | LLM wrappers with bounded memory and pluggable backends |
| Experiment runner | `src/collusionlab/runner/` | Single runs and parallel parameter sweeps |
| Oversight + detection | `src/collusionlab/auditing/` | Transcript, behavioral, temporal, and LLM-judge auditors; risk fusion; penalty enforcement |
| Storage | `src/collusionlab/storage.py` | Pluggable `RunStore` protocol: flat JSONL, SQLite, or PostgreSQL |
| Analysis UI | `src/collusionlab/ui/app.py` | Streamlit dashboard for trajectory replay and transcript inspection |

### Repository Layout

```text
collusionlab/
├── configs/                    # YAML run configs
│   ├── base.yaml
│   ├── pricing_public_chat.yaml
│   ├── pricing_private_chat.yaml
│   └── pricing_audit.yaml
├── src/
│   └── collusionlab/
│       ├── environments/       # Game environments
│       │   ├── base.py
│       │   ├── communication.py
│       │   └── pricing/        # Calvano and Bertrand repeated pricing games
│       ├── agents/             # LLM agent wrappers and model clients
│       │   ├── model_client.py
│       │   ├── backends/       # OpenAI, Anthropic, and DeepSeek backends
│       │   ├── memory.py
│       │   └── llm_agent.py
│       ├── runner/             # Experiment and sweep runners
│       ├── auditing/           # Oversight, transcript, behavioral, temporal, and LLM-judge auditors
│       ├── metrics/            # Collusion and concealment metrics
│       ├── storage.py          # RunStore protocol: SQLite and PostgreSQL backends
│       └── ui/                 # Streamlit analysis dashboard
├── data/
│   ├── raw/                    # Round-level JSONL logs (flat file backend)
│   └── processed/              # Derived metrics tables
├── prompts/pricing/            # System and turn prompt templates
├── notebooks/                  # Jupyter analysis notebooks
├── scripts/                    # Utility scripts (e.g. Calvano calibration)
├── tests/                      # Unit tests
├── docs/                       # Design notes and implementation plan
├── environment.yml             # Core conda environment
├── environment-gpu.yml         # Optional GPU environment
└── .env.example                # API key template
```

## Installation

### Prerequisites

- [Mamba](https://mamba.readthedocs.io/) or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- An OpenAI API key (required for the default backend)
- An Anthropic API key (optional)
- A DeepSeek API key (optional)
- A PostgreSQL connection string (optional; SQLite or flat files require no additional setup)

### 1. Create the Environment

```bash
mamba env create -f environment.yml
mamba activate collusion_lab
```

For local GPU inference:

```bash
mamba env create -f environment-gpu.yml
mamba activate collusion_lab_gpu
```

### 2. Configure API Keys and Storage

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...              # optional
DEEPSEEK_API_KEY=...               # optional
COLLUSIONLAB_STORAGE_URL=...       # optional; see Storage
```

### 3. Install the Package

```bash
pip install -e src/
```

## Storage

Three storage backends are supported, all conforming to the `RunStore` interface.
The backend is selected by setting `COLLUSIONLAB_STORAGE_URL` in `.env` or
passing `--storage-url` to the runner:

| Backend | URI format | Recommended use |
|---|---|---|
| Flat JSONL (default) | *(unset)* | Local experiments; no additional setup required |
| SQLite | `sqlite:///path/to/runs.db` | Single-machine persistence; UI reads without shared filesystem |
| PostgreSQL | `postgresql://user:pass@host/db` | Multi-machine or shared deployments |

SQLite uses WAL mode to support concurrent reads from the Streamlit UI while a
sweep is in progress. PostgreSQL requires `psycopg[binary]`, included in
`environment.yml`.

## Usage

### Single Experiment

```bash
python -m collusionlab.runner.experiment --config configs/base.yaml
```

Output is written to `data/raw/` as JSONL, and to the configured database if
`COLLUSIONLAB_STORAGE_URL` is set.

### Parameter Sweep

```bash
PYTHONPATH=src python -m collusionlab.runner.sweep \
  --sweep configs/sweep_comm.yaml --max-workers 4
```

Runs execute in parallel via `ProcessPoolExecutor`. Each run produces a
`log.jsonl` and `manifest.json` (and a database row if storage is configured).
The sweep produces an aggregate `sweep_manifest.json` with per-run status,
timing, and config snapshots. Failed runs are recorded without aborting the sweep.

Example sweep configs:

| Config | Mode | Description |
|---|---|---|
| `sweep_comm.yaml` | grid | 3 seeds × 3 communication modes (9 runs) |
| `sweep_list_example.yaml` | list | 3 explicit configurations |

### Analysis Dashboard

```bash
PYTHONPATH=src streamlit run src/collusionlab/ui/app.py
```

Four navigation pages:

- **Analyze** — single-run inspection: config metadata, price trajectory charts, round-by-round transcript, and summary metrics
- **Run** *(forthcoming)* — YAML config editor with live validation
- **Sweep** *(forthcoming)* — parameter grid preview and background sweep execution
- **Compare** *(forthcoming)* — heatmaps and cross-run comparison for sweep results

### Configuration

Every run is fully specified by a YAML config file:

```yaml
environment:
  n_agents: 2
  n_rounds: 25
  seed: 42
  demand_model: bertrand   # or: calvano
  nash_price: 8
  monopoly_price: 10

agent:
  backend: openai         # openai | anthropic | deepseek
  model: gpt-4o-mini
  memory_window: 5
  temperature: 0.2

communication:
  mode: none              # none | public | private

oversight:
  mode: none              # none | audit-plus-penalty
  audit_probability: 0.0
  penalty_factor: 1.0
```

Pre-built variant configs:

| Config | Communication | Oversight |
|---|---|---|
| `base.yaml` | none | none |
| `pricing_public_chat.yaml` | public | none |
| `pricing_private_chat.yaml` | private | none |
| `pricing_audit.yaml` | public | audit + penalty |

## Key Metrics

| Metric | Description |
|---|---|
| Price elevation | Agent prices relative to the competitive Nash baseline |
| Price convergence | Reduction in price dispersion over rounds |
| Profit uplift | Profit gain relative to the no-communication condition |
| Covert coordination | Collusion language declines while above-baseline profits persist |
| Hollow coordination | Collusion language persists while profits converge back to Nash |
| Price-follow rate | Frequency with which one agent mirrors the other's price movement |
| Audit response | Co-variation of flag rate and profit elevation as audit probability increases |

## Design Principles

- **Concealment-first** — the auditing and oversight infrastructure is a primary research target, not an afterthought
- **Unified interface** — a single config schema, runner, log format, and storage protocol spans all experiment variants; extending any module does not require modifying unrelated code
- **Low compute** — prompt-only agents and small models; no GPU required for the core environment
- **Reproducibility** — every run is fully determined by its config and seed and can be replayed from flat files or the database
- **Extensibility** — demand models, LLM backends, auditors, communication modes, and storage backends are independently swappable

## Testing

```bash
pytest tests/
```

Tests use synthetic JSONL fixtures and mock `ModelClient.generate()` calls;
no network access is required. PostgreSQL storage tests require a live connection.

## Related Work

1. [Evaluating LLM Agent Collusion in Double Auctions](https://arxiv.org/abs/2507.01413)
2. [Secret Collusion among AI Agents: Multi-Agent Deception via Steganography](https://openreview.net/forum?id=bnNSQhZJ88)
3. [Detecting Steganographic Collusion in Multi-Agent LLMs](https://arxiv.org/abs/2510.04303)
4. [Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151)
5. [A Survey of Collusion Risk in LLM-Powered Multi-Agent Systems](https://openreview.net/pdf?id=Ylh8617Qyd)
6. [Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](https://www.sciencedirect.com/science/article/pii/S0950705126007938)
