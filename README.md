# CollusionLab

A low-compute research platform for studying how LLM agents respond to
oversight — specifically, whether penalizing explicit collusion suppresses
it or drives it underground.

## Research Question

Prior work (e.g. [Agrawal et al., 2025](https://arxiv.org/abs/2507.01413))
establishes that LLM agents *will* collude when given communication channels.
CollusionLab asks what happens next:

> When oversight penalizes explicit coordination, do LLM agents stop colluding —
> or do they adapt, sustaining above-Nash profits while scrubbing colluding
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

## What Makes This Different

| | Agrawal et al. (2025) | CollusionLab |
|---|---|---|
| **Core question** | Do agents collude? | Do agents *adapt* under oversight? |
| **Market structure** | Continuous double auction | Calvano logit + Bertrand (pluggable) |
| **Oversight** | Binary (CEO urgency / overseer) | Probabilistic audit + penalty, LLM judge or behavioral |
| **Detection** | Pricing metrics | Multi-signal auditing layer (transcript, behavioral, temporal, LLM judge, risk fusion) |
| **Models** | GPT-4.1, Claude Sonnet | Small/affordable models (gpt-4o-mini, DeepSeek) |
| **Artifact** | Workshop paper | Open-source, reproducible platform |

## Architecture

Five modules, cleanly separated:

| Module | Location | Purpose |
|---|---|---|
| Environment engine | `src/collusionlab/environments/` | Repeated pricing games; Calvano and Bertrand demand models |
| Agent layer | `src/collusionlab/agents/` | LLM wrappers with bounded memory and pluggable backends |
| Experiment runner | `src/collusionlab/runner/` | Single runs and parallel parameter sweeps |
| Oversight + detection | `src/collusionlab/auditing/` | Transcript, behavioral, temporal, LLM-judge auditors; risk fusion; penalty enforcement |
| Analysis UI | `src/collusionlab/ui/app.py` | Streamlit dashboard for trajectory replay and transcript inspection |

## Key Metrics

| Metric | Description |
|---|---|
| Price elevation | Agent prices relative to Nash baseline |
| Profit uplift | Profit gain vs. no-communication condition |
| Covert coordination | Collusion language declines but above-baseline profits persist |
| Hollow coordination | Collusion language persists; profits converge back to Nash |
| Price-follow rate | Implicit coordination signal: frequency of mirroring rival price moves |
| Audit response | How flag rate and profit elevation co-vary as audit probability increases |

## Installation

### Prerequisites
- [Mamba](https://mamba.readthedocs.io/) or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- OpenAI API key (required); Anthropic and DeepSeek keys (optional)

```bash
mamba env create -f environment.yml
mamba activate collusion_lab
cp .env.example .env   # fill in API keys
pip install -e src/
```

For local GPU inference: `mamba env create -f environment-gpu.yml`

## Usage

### Single experiment
```bash
python -m collusionlab.runner.experiment --config configs/base.yaml
```

### Parameter sweep
```bash
PYTHONPATH=src python -m collusionlab.runner.sweep \
  --sweep configs/sweep_comm.yaml --max-workers 4
```

### Analysis dashboard
```bash
PYTHONPATH=src streamlit run src/collusionlab/ui/app.py
```

## Configs

| Config | Communication | Oversight |
|---|---|---|
| `base.yaml` | none | none |
| `pricing_public_chat.yaml` | public | none |
| `pricing_private_chat.yaml` | private | none |
| `pricing_audit.yaml` | public | audit + penalty |

Core parameters (from `configs/base.yaml`):
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

## Design Principles

- **Concealment-first** — the auditing and oversight infrastructure is a first-class feature, not an afterthought.
- **Low compute** — prompt-only agents, small models, no GPU required.
- **Reproducibility** — every run fully specified by config + seed; anything surprising can be replayed.
- **Pluggable** — demand models, backends, auditors, and communication modes are all swappable without touching core logic.

## Related Work

1. [Evaluating LLM Agent Collusion in Double Auctions](https://arxiv.org/abs/2507.01413) — Agrawal et al. (2025); establishes onset baseline this work extends
2. [Secret Collusion among AI Agents: Multi-Agent Deception via Steganography](https://openreview.net/forum?id=bnNSQhZJ88)
3. [Detecting Steganographic Collusion in Multi-Agent LLMs](https://arxiv.org/abs/2510.04303)
4. [Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151)
5. [A Survey of Collusion Risk in LLM-Powered Multi-Agent Systems](https://openreview.net/pdf?id=Ylh8617Qyd)
6. [Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](https://www.sciencedirect.com/science/article/pii/S0950705126007938)
