# CollusionLab

A low-compute research platform for studying the **onset** and **concealment** of collusion in small LLM multi-agent systems.

## Overview

CollusionLab is a reproducible testbed for experiments in which multiple LLM agents interact in repeated games, first under conditions that may induce collusion and then under increasing oversight designed to suppress explicit coordination. The platform is intended for small, affordable models and short-context runs, making it practical for student research without an HPC cluster.

The main goal is to study a transition that current work only partially addresses: agents moving from normal competitive behavior to overt collusion, and then from overt collusion to subtler or partially concealed coordination when audits or penalties are introduced.

## Research Question

The central question is:

> Under what environmental and communication conditions do small LLM agents begin colluding, and how do they adapt when oversight penalizes explicit collusion?

This breaks into two linked subproblems:

- **Onset:** which variables make collusion appear, such as communication access, repeated interaction, memory, and agent count.
- **Concealment:** whether oversight stops collusion entirely or instead pushes it into less explicit forms that are harder to detect from transcripts alone.

## Why This Project

Prior work has already shown that LLM agents can collude in market-like settings and that multi-agent steganographic or covert coordination is a credible risk. However, the literature still lacks a compact, reusable, small-model-oriented platform that combines reproducible experiment configuration, trajectory logging, replay, and detection-oriented analysis into one system.

This project is designed to fill that gap by making the experiment harness itself a contribution, not just a single result.

## Scope

The recommended first version is intentionally narrow:

- One core environment: a repeated pricing or auction-style game.
- One main model family: small API-accessible models.
- A small set of communication modes: none, public, and private.
- A small set of oversight modes: none and audit-plus-penalty.
- A strong emphasis on reproducibility, logging, and replay.

This scope is large enough to generate meaningful results while remaining feasible on a modest budget.

## Core Architecture

The platform should be organized into five main modules.

| Module | Purpose |
| --- | --- |
| Environment engine | Runs repeated games with deterministic rules and seeded randomness |
| Agent layer | Wraps LLMs with prompts, bounded memory, and communication permissions via a pluggable model client interface |
| Experiment runner | Executes sweeps over configurations and reproduces runs exactly |
| Detection layer | Computes collusion, concealment, and oversight-response metrics |
| Analysis UI | Replays trajectories and supports inspection of messages, actions, and rewards |

This modular structure follows the main needs highlighted by recent work on collusion evaluation, auditing, and multi-agent interpretability: controlled environments, complete logs, and repeatable manifests.

## Must-Have Features

### Reproducible configuration

Every run should be fully defined by a machine-readable config that specifies the environment, model, prompts, communication mode, round count, seed, and oversight settings. This makes it possible to reproduce surprising trajectories and compare variants cleanly across sweeps.

### Full trajectory logging

Each run should store round-level state, agent messages, actions, rewards, and audit events. Since the research focus is on emergence dynamics, not just final outcomes, full logs are essential.

### Structured replay and analysis UI

A local Streamlit application serves as both the experiment control panel and the primary analysis tool. It should support:

- Launching and monitoring individual runs and parameter sweeps with live progress.
- Round-by-round trajectory inspection: actions, rewards, messages, audit events, and per-round detection signals.
- Config diffing and transcript search.
- Summary metric views and cross-run comparison for sweep results.

Replay and analysis are core product features, not afterthoughts. The dynamics this project studies only become interpretable when researchers can navigate from a summary metric directly to the transcript rounds that produced it.

### Core collusion metrics

The first version should compute a minimal but useful set of metrics:

- price elevation relative to a competitive baseline,
- price dispersion and convergence,
- profit uplift relative to non-communication settings,
- persistence of coordinated behavior after shocks or audits,
- transcript-level collusion flags.

### Oversight module

The system should support at least one lightweight auditor that can inspect messages or behavior and impose penalties. Good initial choices are keyword-based transcript filters, random audits, and simple behavioral checks based on suspicious price trajectories.

### Concealment indicators

Because the project is about bad emergent behavior under pressure, the platform should track two complementary gaps:

- **Covert coordination:** explicit collusion language declines while above-baseline profits persist. This “clean language, dirty behavior” gap is the primary concealment signal.
- **Hollow coordination:** explicit collusion language persists while profits remain near baseline. This indicates either posturing, or that oversight is suppressing behavioral outcomes while agent language lags.

The detection layer should also measure implicit coordination signals — price-following rates and action convergence — that can reveal coordination even when agents produce no flagged language. These lightweight indicators are part of the V1 detection layer and are distinct from the steganographic benchmark described under Possible Extensions.

### Batch experiment runner

The platform needs a way to sweep over a small set of conditions, such as communication on/off, audit probability low/high, and number of agents small/large. Without sweeps, the result will remain anecdotal instead of becoming a proper experimental study.

Sweeps must support parallel execution. The baseline target is multicore parallelism on a single laptop. The architecture should leave a clear path to HPC or distributed orchestration for larger studies without requiring a rewrite of the sweep logic.

## Nice-to-Have Features

### Multiple communication topologies

After the initial version works, add support for no communication, shared public chat, private pairwise chat, and delayed message boards. This makes it easier to study how social structure changes the onset of collusion.

### Early-warning detector

A lightweight predictor could classify whether a run is likely to become collusive based only on its first several rounds. This would be a strong extension because most existing work focuses on identifying ongoing collusion rather than forecasting it early.

### Cross-model evaluation (post-V1)

Testing 2–3 cheap models would help determine whether onset thresholds and concealment tendencies differ by model family. The infrastructure supports this via the `AnthropicModelClient` backend, but it is explicitly deferred until after the first empirical study is complete.

### Intervention sweeps

The platform could test anti-collusion interventions such as noisy communication, forced paraphrasing, random agent replacement, and injected honest agents. This would strengthen the practical safety angle of the project.

## Possible Extensions

### Steganographic coordination benchmark

A second environment could explicitly pressure agents to coordinate while punishing obvious collusion, creating a dedicated benchmark for covert or coded communication. This should be treated as a later extension rather than the first milestone. Note that lightweight implicit coordination signals — price-following rates and action mutual information — are included in the V1 detection layer; the steganographic benchmark refers to this dedicated adversarial environment, not those metrics.

### Multi-agent interpretability hooks

If open-weight models are added later, the platform could incorporate probe-based or representation-level analysis inspired by recent multi-agent interpretability work on collusion detection.

### Cross-domain transfer

Detectors or heuristics developed in pricing environments could later be tested in committee voting, procurement, or ranking settings. This would help answer whether collusion signatures generalize across domains.

### Misalignment transfer studies

A later study could vary narrow goals like “maximize revenue” or “win at all costs” and test whether these prompts increase deceptive or collusive behavior in otherwise identical environments.

### Human-in-the-loop auditing

The oversight module could eventually compare automated auditors with human reviewers who only see partial logs, bringing the platform closer to realistic governance settings.

## Recommended First Milestone

A strong first milestone is a working V1 with:

- one repeated pricing environment,
- one small model family,
- three communication modes: none, public, private,
- two oversight modes: none and audit-plus-penalty,
- full logging and replay,
- and one empirical study showing when collusion emerges and how oversight changes the form it takes.

This is enough to demonstrate both platform value and a clear research contribution.

## Success Criteria

The project should count as successful if it can show the following:

- Runs are reproducible from saved configs.
- Communication increases collusion in at least one setting.
- Oversight reduces explicit collusion markers.
- In at least some conditions, suspicious coordination persists even after explicit collusion language decreases.
- The replay and logging tools make those transitions easy to inspect.

## Suggested Repository Layout

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

This structure keeps the platform maintainable and leaves room for later additions like new environments, new auditors, or open-model interpretability support.

## Design Principles

- **Low compute first:** prioritize prompt-only agents and short-context interactions over training-heavy approaches.
- **Reproducibility first:** every surprising result should map to a saved config and seed.
- **Interpretability first:** logs and replay are core product features, not afterthoughts.
- **One strong environment before many weak ones:** depth matters more than breadth for the first paper.
- **Extensible architecture:** the first version should make later additions easy rather than trying to solve every variant at once.

## Future Direction

Once the first version is stable, the natural path is to broaden from overt market collusion into partial concealment, then into more explicitly covert coordination and richer detection strategies. That progression preserves a clean research arc while keeping each stage practical and publishable.

## Suggested Reading

1. **Evaluating LLM Agent Collusion in Double Auctions**  
   https://arxiv.org/abs/2507.01413

2. **Evaluating LLM Agent Collusion in Double Auctions — review summary**  
   https://www.themoonlight.io/en/review/evaluating-llm-agent-collusion-in-double-auctions

3. **Secret Collusion among AI Agents: Multi-Agent Deception via Steganography**  
   https://openreview.net/forum?id=bnNSQhZJ88

4. **Detecting Steganographic Collusion in Multi-Agent LLMs**  
   https://arxiv.org/abs/2510.04303

5. **Detecting Multi-Agent Collusion Through Multi-Agent Interpretability**  
   https://arxiv.org/abs/2604.01151

6. **A Survey of Collusion Risk in LLM-Powered Multi-Agent Systems**  
   https://openreview.net/pdf?id=Ylh8617Qyd

7. **Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems**  
   https://www.sciencedirect.com/science/article/pii/S0950705126007938
