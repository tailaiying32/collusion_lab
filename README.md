# CollusionLab

A low-compute research platform for studying the **onset** and **concealment** of collusion in small LLM multi-agent systems.

## Overview

CollusionLab is a reproducible testbed for experiments in which multiple LLM agents interact in repeated games, first under conditions that may induce collusion and then under increasing oversight designed to suppress explicit coordination. The platform targets small, affordable models and short-context runs, making it practical for student research without an HPC cluster.

The central research question is:

> Under what environmental and communication conditions do small LLM agents begin colluding, and how do they adapt when oversight penalizes explicit collusion?

This breaks into two linked subproblems:

- **Onset** — which variables (communication access, repeated interaction, memory, agent count) make collusion appear.
- **Concealment** — whether oversight stops collusion entirely or instead pushes it into less explicit forms that are harder to detect from transcripts alone.

## Architecture

The platform is organized into five modules.

| Module | Location | Purpose |
|---|---|---|
| Environment engine | `src/collusionlab/environments/` | Repeated games with deterministic rules and seeded RNG |
| Agent layer | `src/collusionlab/agents/` | LLM wrappers with prompts, bounded memory, and pluggable model clients |
| Experiment runner | `src/collusionlab/runner/` | Single runs and parameter sweeps with parallel execution |
| Detection layer | `src/collusionlab/auditing/`, `src/collusionlab/metrics/` | Collusion, concealment, and oversight-response metrics |
| Analysis UI | `src/collusionlab/ui/app.py` | Streamlit app for trajectory replay and cross-run comparison |

### Repository layout

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
│       │   └── pricing/        # Calvano-style repeated pricing game
│       ├── agents/             # LLM agent wrappers and model clients
│       │   ├── model_client.py
│       │   ├── backends/       # OpenAI and Anthropic backends
│       │   ├── memory.py
│       │   └── llm_agent.py
│       ├── runner/             # Experiment and sweep runners
│       ├── auditing/           # Oversight and transcript auditing
│       ├── metrics/            # Collusion and concealment metrics
│       └── ui/                 # Streamlit analysis dashboard
├── data/
│   ├── raw/                    # Round-level JSONL logs
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

- [Mamba](https://mamba.readthedocs.io/) or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for environment management
- An OpenAI API key (required for the default backend)
- An Anthropic API key (optional, for the Anthropic backend)

### 1. Create the environment

```bash
mamba env create -f environment.yml
mamba activate collusion_lab
```

For local GPU inference, use the optional environment instead:

```bash
mamba env create -f environment-gpu.yml
mamba activate collusion_lab_gpu
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...   # optional
```

### 3. Install the package (editable)

```bash
pip install -e src/
```

## Usage

### Running a single experiment

```bash
python -m collusionlab.runner.experiment --config configs/base.yaml
```

Output logs are written to `data/raw/` as JSONL files.

### Running a parameter sweep

```bash
python -m collusionlab.runner.sweep --config configs/base.yaml --max-workers 4
```

Results are collected in `data/processed/`.

### Configuration

Every run is fully defined by a YAML config. The base config (`configs/base.yaml`) sets defaults; variant configs extend it:

```yaml
environment:
  n_agents: 2
  n_rounds: 50
  seed: 42
  demand_model: calvano
  nash_price: 8
  monopoly_price: 10

agent:
  backend: openai      # openai | anthropic
  model: gpt-4o-mini
  memory_window: 5
  temperature: 0.2

communication:
  mode: none           # none | public | private

oversight:
  mode: none           # none | audit-plus-penalty
  audit_probability: 0.0
  penalty_factor: 1.0
```

Pre-built variant configs:

| Config | Communication | Oversight |
|---|---|---|
| `base.yaml` | none | none |
| `pricing_public_chat.yaml` | public | none |
| `pricing_private_chat.yaml` | private | none |
| `pricing_audit.yaml` | public | audit-plus-penalty |

### Analysis UI

Launch the Streamlit dashboard to inspect experiment runs:

```bash
PYTHONPATH=src streamlit run src/collusionlab/ui/app.py
```

The dashboard provides four navigation pages (Analyze is currently implemented; others are placeholders):

- **Analyze** (available now) — single-run inspection with four tabs:
  - **Config** — run metadata (elapsed time, tokens, cost) and full YAML config
  - **Trajectory** — per-agent action and reward elevation charts with Nash/monopoly reference lines
  - **Transcript** — round-by-round timeline with messages, audit events, and flags; supports filtering and keyword search
  - **Metrics** — quick summary stats (mean spread, elevation, audit/flag counts)
- **Run** (coming soon) — YAML config editor with live validation and per-round progress
- **Sweep** (coming soon) — parameter grid preview and background sweep execution
- **Compare** (coming soon) — heatmaps and cross-run comparison for sweep results

### Notebooks

Jupyter notebooks for exploratory analysis are in `notebooks/`. Launch with:

```bash
jupyter lab
```

## Key Metrics

| Metric | Description |
|---|---|
| Price elevation | Agent prices relative to the competitive Nash baseline |
| Price convergence | Reduction in price dispersion over rounds |
| Profit uplift | Profit gain relative to no-communication settings |
| Covert coordination | "Clean language, dirty behavior" — collusion language declines but above-baseline profits persist |
| Hollow coordination | Collusion language persists while profits remain near baseline |
| Price-follow rate | Implicit coordination signal: how often one agent mirrors the other's price move |

## Testing

```bash
pytest tests/
```

Tests use synthetic JSONL fixtures and mock `ModelClient.generate()` calls — no network access required.

## Design Principles

- **Low compute first** — prompt-only agents and short-context interactions; no GPU required for the core environment.
- **Reproducibility first** — every run is fully defined by a config and seed; surprises can always be replayed.
- **Interpretability first** — logs and the Streamlit replay UI are core features, not afterthoughts.
- **One strong environment before many** — depth in a single pricing game before breadth across domains.
- **Extensible architecture** — demand models, model clients, environments, auditors, and communication modes are all pluggable; adding a new variant never requires touching core logic.

## Related Work

1. [Evaluating LLM Agent Collusion in Double Auctions](https://arxiv.org/abs/2507.01413)
2. [Secret Collusion among AI Agents: Multi-Agent Deception via Steganography](https://openreview.net/forum?id=bnNSQhZJ88)
3. [Detecting Steganographic Collusion in Multi-Agent LLMs](https://arxiv.org/abs/2510.04303)
4. [Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151)
5. [A Survey of Collusion Risk in LLM-Powered Multi-Agent Systems](https://openreview.net/pdf?id=Ylh8617Qyd)
6. [Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](https://www.sciencedirect.com/science/article/pii/S0950705126007938)
