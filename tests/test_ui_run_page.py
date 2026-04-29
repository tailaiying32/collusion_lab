"""Tests for Run page pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.ui.run_page import _validate_yaml


def test_run_page_validate_yaml_accepts_base_config():
    text = (ROOT / "configs" / "base.yaml").read_text(encoding="utf-8")
    cfg, err = _validate_yaml(text)
    assert err is None
    assert cfg is not None
    assert cfg.env_type == "pricing"


def test_run_page_validate_yaml_reports_parse_error():
    cfg, err = _validate_yaml("not: [valid")
    assert cfg is None
    assert err is not None
    assert "YAML parse error" in err
