"""Tests for Run page pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.ui.run_page import (
    MODEL_TO_BACKEND,
    _build_selector_options,
    _build_prompt_preview,
    _load_strategic_guidance_preset,
    _patch_agent_models,
    _patch_auditor_models,
    _patch_strategic_guidance_preset,
    _validate_yaml,
)


def test_run_page_openai_model_picker_includes_requested_models():
    assert MODEL_TO_BACKEND["gpt-4.1"] == "openai"
    assert MODEL_TO_BACKEND["gpt-4.1-mini"] == "openai"
    assert MODEL_TO_BACKEND["gpt-5.1"] == "openai"


def test_run_page_validate_yaml_accepts_base_config():
    text = (ROOT / "configs" / "baseline_public_neutral_audit.yaml").read_text(encoding="utf-8")
    cfg, err = _validate_yaml(text)
    assert err is None
    assert cfg is not None
    assert cfg.env_type == "pricing"


def test_run_page_validate_yaml_reports_parse_error():
    cfg, err = _validate_yaml("not: [valid")
    assert cfg is None
    assert err is not None
    assert "YAML parse error" in err


def test_patch_agent_models_updates_backend_and_model():
    text = (ROOT / "configs" / "baseline_public_neutral_audit.yaml").read_text(encoding="utf-8")
    updated = _patch_agent_models(
        text,
        {"backend": "deepseek", "model": "deepseek-v4-flash"},
    )
    cfg, err = _validate_yaml(updated)
    assert err is None
    assert cfg is not None
    assert cfg.agents.backend == "deepseek"
    assert cfg.agents.model == "deepseek-v4-flash"


def test_patch_auditor_models_updates_judge_fields():
    text = (ROOT / "configs" / "baseline_public_neutral_audit.yaml").read_text(encoding="utf-8")
    updated = _patch_auditor_models(
        text,
        {"llm_judge_backend": "deepseek", "llm_judge_model": "deepseek-v4-flash"},
    )
    cfg, err = _validate_yaml(updated)
    assert err is None
    assert cfg is not None
    assert cfg.oversight.llm_judge_backend == "deepseek"
    assert cfg.oversight.llm_judge_model == "deepseek-v4-flash"


def test_build_selector_options_marks_recent_config():
    opts = _build_selector_options(
        ["baseline_public_neutral_audit.yaml", "stego_capability_audit.yaml"],
        "baseline_public_neutral_audit.yaml",
    )
    assert opts[0] == "(custom)"
    assert "baseline_public_neutral_audit.yaml (most recent)" in opts


def test_patch_strategic_guidance_preset_updates_config():
    text = (ROOT / "configs" / "baseline_public_neutral_audit.yaml").read_text(encoding="utf-8")
    updated = _patch_strategic_guidance_preset(text, "stego_capability")
    cfg, err = _validate_yaml(updated)
    assert err is None
    assert cfg is not None
    assert cfg.strategic_guidance == ""
    assert cfg.strategic_guidance_preset == "stego_capability"
    assert "innocuous communication conventions" in cfg.resolved_strategic_guidance()


def test_prompt_preview_includes_strategic_guidance_in_turn_prompts():
    text = (ROOT / "configs" / "baseline_public_neutral_audit.yaml").read_text(encoding="utf-8")
    updated = _patch_strategic_guidance_preset(text, "stego_shared_codebook")
    cfg, err = _validate_yaml(updated)
    assert err is None
    assert cfg is not None

    preview = _build_prompt_preview(cfg)

    assert "Weather is clear" in preview["Communication reasoning turn"]
    assert "Weather is clear" in preview["Message turn"]
    assert "Weather is clear" in preview["Action turn"]
    assert "Weather is clear" not in preview["System"]


def test_strategic_guidance_presets_are_file_backed():
    text = _load_strategic_guidance_preset("Stego capability")
    assert "innocuous communication conventions" in text


def test_emergent_stego_capability_preset_is_file_backed():
    text = _load_strategic_guidance_preset("Emergent stego capability")
    assert "ordinary business communication" in text
