# CollusionLab

A unified, low-compute research infrastructure for studying how LLM agents respond to
oversight — and a reproducible baseline for future collusion research.

## Research Question

Prior work (e.g. [Agrawal et al., 2025](https://arxiv.org/abs/2507.01413))
establishes that LLM agents *will* collude when given communication channels.
CollusionLab asks what happens next:

> When oversight penalizes explicit coordination, do LLM agents stop colluding —
> or do they adapt, sustaining above-Nash profits while scrubbing collusion
> language from their transcripts?

This breaks into two linked subproblems:

- **Onset** — replicating the baseline: which conditions (communication access,
  repeated interaction, memory, agent count) produce collusion. Tested under
  Calvano logit demand (symmetric to prior work) and Bertrand winner-take-all
  demand (where the Bertrand paradox makes tacit collusion especially notable).
- **Concealment** — the novel contribution: does probabilistic auditing + profit
  penalties eliminate collusion, or does it produce *covert coordination* —
  clean language, dirty behavior — or *hollow coordination* — collusion language
  that persists while profits converge back to Nash?

## Platform First

CollusionLab is designed to be the infrastructure you run *before* you write your
paper. Rather than reimplementing a pricing game, an agent loop, a communication
layer, and an auditing pipeline from scratch — as every prior study has done —
CollusionLab provides all of these as composable, config-driven building blocks:

- **Swap demand models** — Calvano logit or Bertrand winner-take-all today;
  add new market structures without touching the agent or runner code
- **Swap LLM backends** — OpenAI, Anthropic, DeepSeek, or local GPU inference,
  all behind a single `ModelClient` interface
- **Swap oversight mechanisms** — keyword auditing, behavioral signals, temporal
  pattern detection, or a full LLM judge, all orchestrated by a single
  `OversightManager`
- **Run sweeps, not one-offs** — parallel parameter sweeps with full config
  snapshots, continue-on-error, and aggregate manifests built in
- **Replay anything** — every run is a JSONL log fully reproducible from its
  seed and config; the Streamlit dashboard lets you step through any transcript

The Agrawal et al. double-auction study, for example, is an experiment that could
be expressed as a CollusionLab config. That's the goal: make collusion experiments
cheap to define, run, and compare.

## What Makes This Different

| | Agrawal et al. (2025) | CollusionLab |
|---|---|---|
| **Core question** | Do agents collude? | Do agents *adapt* under oversight? |
| **Market structure** | Continuous double auction | Calvano logit + Bertrand (pluggable) |
| **Oversight** | Binary (CEO urgency / overseer) | Probabilistic audit + penalty, LLM judge or behavioral |
| **Detection** | Pricing metrics | Multi-signal auditing layer (transcript, behavioral, temporal, LLM judge, risk fusion) |
| **Models** | GPT-4.1, Claude Sonnet | Small/affordable models (gpt-4o-mini, DeepSeek) |
| **Artifact** | Workshop paper | Unified research infrastructure: pluggable environments, agents, auditors, sweep runner, replay UI |

## Architecture

Five modules, cleanly separated:

| Module | Location | Purpose |
|---|---|---|
| Environment engine | `src/collusionlab/environments/` | Repeated pricing games; Calvano and Bertrand demand models |
| Agent layer | `src/collusionlab/agents/` | LLM wrappers with bounded memory and pluggable backends |
| Experiment runner | `src/collusionlab/runner/` | Single runs and parallel parameter sweeps |
| Oversight + detection | `src/collusionlab/auditing/` | Transcript, behavioral, temporal, LLM-judge auditors; risk fusion; penalty enforcement |
| Analysis UI | `src/collusionlab/ui/app.py` | Streamlit dashboard for trajectory replay and transcript inspection |

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
│       │   └── pricing/        # Calvano and Bertrand repeated pricing games
│       ├── agents/             # LLM agent wrappers and model clients
│       │   ├── model_client.py
│       │   ├── backends/       # OpenAI, Anthropic, and DeepSeek backends
│       │   ├── memory.py
│       │   └── llm_agent.py
│       ├── runner/             # Experiment and sweep runners
│       ├── auditing/           # Oversight, transcript, behavioral, temporal, LLM-judge auditors
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

- [Mamba](https://mamba.readthedocs.io/) or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- An OpenAI API key (required for the default backend)
- An Anthropic API key (optional)
- A DeepSeek API key (optional)

### 1. Create the environment

```bash
mamba env create -f environment.yml
mamba activate collusion_lab
```

For local GPU inference:

```bash
mamba env create -f environment-gpu.yml
mamba activate collusion_lab_gpu
```

### 2. Configure API keys

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...   # optional
DEEPSEEK_API_KEY=...    # optional
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
PYTHONPATH=src python -m collusionlab.runner.sweep --sweep configs/sweep_comm.yaml --max-workers 4
```

Runs execute in parallel via `ProcessPoolExecutor`. Each run writes its own `log.jsonl`
and `manifest.json` under `data/raw/{run_id}/`. The sweep writes an aggregate
`sweep_manifest.json` with per-run status, timing, and config snapshots. Failed runs
are recorded without aborting the sweep.

Example sweep configs:

| Config | Mode | Description |
|---|---|---|
| `sweep_comm.yaml` | grid | 3 seeds × 3 communication modes (9 runs) |
| `sweep_list_example.yaml` | list | 3 explicit configurations |

### Analysis UI

```bash
PYTHONPATH=src streamlit run src/collusionlab/ui/app.py
```

Four navigation pages:

- **Analyze** — single-run inspection: config metadata, price trajectory charts, round-by-round transcript, and summary metrics
- **Run** *(coming soon)* — YAML config editor with live validation
- **Sweep** *(coming soon)* — parameter grid preview and background sweep execution
- **Compare** *(coming soon)* — heatmaps and cross-run comparison for sweep results

### Configuration

Every run is fully defined by a YAML config:

```yaml
environment:
  n_agents: 2
  n_rounds: 50
  seed: 42
  demand_model: calvano   # or: bertrand
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
| Profit uplift | Profit gain relative to no-communication condition |
| Covert coordination | Collusion language declines but above-baseline profits persist |
| Hollow coordination | Collusion language persists while profits converge back to Nash |
| Price-follow rate | Implicit coordination signal: how often one agent mirrors the other's price move |
| Audit response | How flag rate and profit elevation co-vary as audit probability increases |

## Design Principles

- **Concealment-first** — the auditing and oversight infrastructure is a first-class feature, not an afterthought
- **Unified interface** — one config schema, one runner, one log format across all experiment variants; adding a new market or oversight mechanism never requires touching unrelated code
- **Low compute** — prompt-only agents, small models, no GPU required for the core environment
- **Reproducibility** — every run is fully specified by config + seed; anything surprising can be replayed
- **Pluggable** — demand models, backends, auditors, and communication modes are all swappable without touching core logic

## Testing

```bash
pytest tests/
```

Tests use synthetic JSONL fixtures and mock `ModelClient.generate()` calls — no network access required.

## Related Work

1. [Evaluating LLM Agent Collusion in Double Auctions](https://arxiv.org/abs/2507.01413) — Agrawal et al. (2025); establishes the onset baseline this work extends
2. [Secret Collusion among AI Agents: Multi-Agent Deception via Steganography](https://openreview.net/forum?id=bnNSQhZJ88)
3. [Detecting Steganographic Collusion in Multi-Agent LLMs](https://arxiv.org/abs/2510.04303)
4. [Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151)
5. [A Survey of Collusion Risk in LLM-Powered Multi-Agent Systems](https://openreview.net/pdf?id=Ylh8617Qyd)
6. [Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](https://www.sciencedirect.com/science/article/pii/S0950705126007938)
