# Phase 8 Oversight Enforcement Plan

Goal: cleanly separate two oversight regimes in Part B so results answer two different questions:

- `fused` enforcement (current): does a multi-signal regulator suppress collusion?
- `transcript_only` enforcement: does language-only oversight induce overt-to-covert adaptation?

## Why this change

Right now, `audit-penalty` always includes behavior and temporal auditors in the fusion decision. This is useful as a strict regulator baseline, but it can blur the specific mechanism behind transition effects. For Phase 8 claims, we need a policy switch that is explicit and reproducible.

## Scope

### 1) Config schema additions

File: `src/collusionlab/runner/config.py`

- Add to `OversightConfig`:
  - `enforcement_policy: Literal["fused", "transcript_only"] = "fused"`

Rationale: make enforcement policy a first-class experimental variable rather than implicit in code.

### 2) Oversight manager policy branch

File: `src/collusionlab/auditing/oversight_manager.py`

- Keep current behavior for `enforcement_policy == "fused"`.
- Add transcript-only branch for `enforcement_policy == "transcript_only"`:
  - Run auditors as today (for logging transparency), but set `penalty_applied` based only on transcript evidence:
    - transcript auditor flagged, or
    - hard rule trigger (`public numeric price intent`).
  - Preserve full `results` and `policy_decision` in `audit_event` for diagnostics.
  - Add `policy_decision["enforcement_policy"]`.

Rationale: maintain observability while isolating the penalty mechanism.

### 3) Expose policy in metrics rows

Files:
- `src/collusionlab/metrics/base.py` (`RunData`)
- `src/collusionlab/environments/pricing/metrics.py`

Changes:
- Add `RunData.oversight_policy` property from `config["oversight"]["enforcement_policy"]` (default `"fused"`).
- Include `oversight_policy` in sweep metric rows.

Rationale: compare both policies in a single sweep and in UI group-bys without relying on separate sweep IDs.

### 4) UI grouping support

File: `src/collusionlab/ui/app.py`

- Add `oversight_policy` as optional dimension/filter in Compare views.
- Keep default views unchanged for backward compatibility.

Rationale: no workflow breakage, but policy-aware comparisons become one click.

### 5) Tests

Files:
- `tests/test_auditing.py`
- `tests/test_metrics.py`
- `tests/test_runner.py` (if needed for manifest/config assertions)

Add:
- `transcript_only` enforces penalty when transcript flags and does not require behavior gate.
- `fused` behavior remains unchanged.
- Sweep metrics include `oversight_policy`.

## Rollout sequence

1. Add schema + manager branch.
2. Add tests for both policy paths.
3. Add metrics/UI exposure.
4. Run Part B pilot with both policies before full launch.

## Acceptance criteria

- Same run config except `enforcement_policy` yields different penalty outcomes when transcript evidence exists but behavior gate is weak.
- Compare view can separate `fused` vs `transcript_only` in one sweep.
- Existing configs without `enforcement_policy` still run and default to `fused`.
