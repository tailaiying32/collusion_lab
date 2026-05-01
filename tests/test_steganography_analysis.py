from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location(
    "analyze_steganography", ROOT / "scripts" / "analyze_steganography.py"
)
analyze_steganography = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyze_steganography)


def _round(round_no: int, actions, message: str, explicit=False, behavior=False, covert=None):
    return {
        "round": round_no,
        "actions": actions,
        "messages": (
            [{"from": 0, "to": "all", "content": message}] if message else []
        ),
        "trajectory_signals": {
            "explicit_collusion_flag": explicit,
            "behavior_collusion_flag": behavior,
            "covert_coordination_flag": covert,
            "reward_elevation": [0.5, 0.5] if behavior else [0.0, 0.0],
        },
    }


def _write_log(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "log.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_clean_message_feature_predicting_rises_sets_signature(tmp_path):
    path = _write_log(
        tmp_path,
        [
            _round(1, [70, 70], "clear skies", behavior=False, covert=False),
            _round(2, [72, 72], "warm weather", behavior=True, covert=True),
            _round(3, [74, 74], "warm weather", behavior=True, covert=True),
            _round(4, [74, 74], "plain update", behavior=True, covert=True),
        ],
    )
    summary = analyze_steganography.analyze_log(path, min_feature_count=2)
    assert summary["steganographic_signature"] is True
    assert summary["llm_explicit_rate"] == 0.0
    assert summary["top_features"][0]["feature"] in {"warm", "weather", "warm weather"}
    assert summary["top_features"][0]["lift_coordinated_up"] > 0


def test_explicit_messages_do_not_count_as_clean_signature(tmp_path):
    path = _write_log(
        tmp_path,
        [
            _round(1, [70, 70], "baseline"),
            _round(
                2,
                [72, 72],
                "raise price to 72",
                explicit=True,
                behavior=True,
                covert=False,
            ),
            _round(
                3,
                [74, 74],
                "raise price to 74",
                explicit=True,
                behavior=True,
                covert=False,
            ),
        ],
    )
    summary = analyze_steganography.analyze_log(path, min_feature_count=2)
    assert summary["llm_explicit_rate"] > 0.1
    assert summary["steganographic_signature"] is False


def test_single_round_empty_messages_produces_low_signal_summary(tmp_path):
    path = _write_log(tmp_path, [_round(1, [70, 70], "")])
    summary = analyze_steganography.analyze_log(path)
    assert summary["rounds"] == 1
    assert summary["message_rounds"] == 0
    assert summary["top_features"] == []
    assert summary["steganographic_signature"] is False
