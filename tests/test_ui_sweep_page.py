"""Tests for Sweep page pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.ui.sweep_page import (
    MODEL_TO_BACKEND,
    _build_selector_options,
    _patch_agent_models,
    _validate_base_yaml,
    _validate_sweep_yaml,
)


def test_sweep_page_openai_model_picker_includes_requested_models():
    assert MODEL_TO_BACKEND["gpt-4.1"] == "openai"
    assert MODEL_TO_BACKEND["gpt-4.1-mini"] == "openai"
    assert MODEL_TO_BACKEND["gpt-5.1"] == "openai"


def test_sweep_page_validate_sweep_yaml_accepts_example():
    text = (ROOT / "configs" / "sweep_stego_study.yaml").read_text(encoding="utf-8")
    cfg, err = _validate_sweep_yaml(text)
    assert err is None
    assert cfg is not None
    assert cfg.mode in {"grid", "list"}


def test_sweep_page_validate_base_yaml_accepts_base_config():
    text = (ROOT / "configs" / "baseline_public_neutral_audit.yaml").read_text(encoding="utf-8")
    msg, data = _validate_base_yaml(text)
    assert data is not None
    assert "Base config valid" in msg


def test_sweep_page_validate_base_yaml_reports_error():
    msg, data = _validate_base_yaml("not: [valid")
    assert data is None
    assert "parse error" in msg


def test_sweep_selector_options_marks_recent_entry():
    opts = _build_selector_options(
        ["sweep_stego_study.yaml", "baseline_public_neutral_audit.yaml"],
        "sweep_stego_study.yaml",
    )
    assert opts[0] == "(custom)"
    assert "sweep_stego_study.yaml (most recent)" in opts


def test_sweep_patch_agent_models_updates_backend_and_model():
    text = (ROOT / "configs" / "baseline_public_neutral_audit.yaml").read_text(encoding="utf-8")
    updated = _patch_agent_models(
        text,
        {"backend": "deepseek", "model": "deepseek-v4-flash"},
    )
    msg, data = _validate_base_yaml(updated)
    assert data is not None
    assert "Base config valid" in msg
    assert data["agents"]["backend"] == "deepseek"
    assert data["agents"]["model"] == "deepseek-v4-flash"
