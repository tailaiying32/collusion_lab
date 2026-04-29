"""Tests for Sweep page pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.ui.sweep_page import _validate_base_yaml, _validate_sweep_yaml


def test_sweep_page_validate_sweep_yaml_accepts_example():
    text = (ROOT / "configs" / "sweep_comm.yaml").read_text(encoding="utf-8")
    cfg, err = _validate_sweep_yaml(text)
    assert err is None
    assert cfg is not None
    assert cfg.mode in {"grid", "list"}


def test_sweep_page_validate_base_yaml_accepts_base_config():
    text = (ROOT / "configs" / "base.yaml").read_text(encoding="utf-8")
    msg, data = _validate_base_yaml(text)
    assert data is not None
    assert "Base config valid" in msg


def test_sweep_page_validate_base_yaml_reports_error():
    msg, data = _validate_base_yaml("not: [valid")
    assert data is None
    assert "parse error" in msg
