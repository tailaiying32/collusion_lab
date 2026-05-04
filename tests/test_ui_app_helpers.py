"""Tests for pure CollusionLab app UI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.storage import make_db_run_ref
from collusionlab.ui.app import (
    _default_side_by_side_labels,
    _run_index_side_by_side_options,
)


def _run_index(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "firm_model": "gpt-4o-mini",
        "n_rounds": 10,
        "n_agents": 2,
        "comm_mode": "public",
        "oversight_mode": "none",
        "audit_probability": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_run_index_side_by_side_options_preserve_arbitrary_run_refs(tmp_path):
    db_ref = make_db_run_ref(str(tmp_path / "runs.sqlite"), "db-run-1")
    fs_ref = tmp_path / "fs-run-1"
    df = _run_index([
        {"run_id": "db-run-1", "run_dir": db_ref},
        {"run_id": "fs-run-1", "run_dir": fs_ref},
    ])

    options = _run_index_side_by_side_options(df)

    assert [opt["run_id"] for opt in options] == ["db-run-1", "fs-run-1"]
    assert options[0]["run_dir"] == db_ref
    assert options[1]["run_dir"] == fs_ref


def test_run_index_side_by_side_options_make_duplicate_labels_unique(tmp_path):
    df = _run_index([
        {"run_id": "abcdefgh-run-a", "run_dir": tmp_path / "run-a"},
        {"run_id": "abcdefgh-run-b", "run_dir": tmp_path / "run-b"},
    ])

    options = _run_index_side_by_side_options(df)
    labels = [opt["label"] for opt in options]

    assert len(labels) == len(set(labels))
    assert labels[1].endswith("(2)")


def test_default_side_by_side_labels_prefer_current_then_next(tmp_path):
    options = _run_index_side_by_side_options(_run_index([
        {"run_id": "run-1", "run_dir": tmp_path / "run-1"},
        {"run_id": "run-2", "run_dir": tmp_path / "run-2"},
        {"run_id": "run-3", "run_dir": tmp_path / "run-3"},
    ]))

    defaults = _default_side_by_side_labels(options, "run-2")

    assert defaults == [options[1]["label"], options[0]["label"]]
