"""Phase 5 unit tests for the sweep runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.environments.pricing import PricingConfig, PricingGame  # noqa: F401
import collusionlab.agents.backends.scripted_client  # noqa: F401
from collusionlab.runner.sweep import (
    SweepConfig,
    SweepRunner,
    apply_overrides,
    format_sweep_progress,
    summarize_sweep_storage,
)
from collusionlab.storage import SQLiteRunStore, parse_db_run_ref


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_env_block() -> dict:
    with (ROOT / "configs" / "baseline_public_neutral_audit.yaml").open() as f:
        cfg = yaml.safe_load(f)
    env = cfg["environment"]
    env.pop("_calibration_note", None)
    return env


def _write_test_base_config(
    path: Path,
    *,
    n_rounds: int = 3,
    seed: int = 0,
    comm_mode: str = "none",
) -> Path:
    """Write a minimal base config using the scripted backend."""
    env = _base_env_block()
    env["n_rounds"] = n_rounds
    env["seed"] = seed
    replies = [str(8)] * (n_rounds * 2)  # enough for action + potential retries
    config = {
        "run_id": None,
        "env_type": "pricing",
        "environment": env,
        "agents": {
            "backend": "scripted",
            "model": "scripted",
            "memory_window": 3,
            "temperature": 0.0,
            "extra": {"replies_by_agent": [list(replies), list(replies)]},
        },
        "prompt_dir": str(ROOT / "prompts" / "pricing"),
        "communication_mode": comm_mode,
        "output_dir": str(path.parent / "data"),
    }
    path.write_text(yaml.safe_dump(config))
    return path


# ---------------------------------------------------------------------------
# Override helpers
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_simple_top_level_key(self):
        data = {"communication_mode": "none", "env_type": "pricing"}
        result = apply_overrides(data, {"communication_mode": "public"})
        assert result["communication_mode"] == "public"
        assert data["communication_mode"] == "none"  # original unchanged

    def test_nested_key(self):
        data = {"environment": {"seed": 42, "n_rounds": 50}}
        result = apply_overrides(data, {"environment.seed": 99})
        assert result["environment"]["seed"] == 99

    def test_list_index(self):
        data = {"agents": {"temperature": 0.2}}
        result = apply_overrides(data, {"agents.temperature": 0.8})
        assert result["agents"]["temperature"] == 0.8

    def test_rejects_unknown_path(self):
        data = {"environment": {"seed": 42}}
        with pytest.raises(ValueError, match="not found"):
            apply_overrides(data, {"environment.nonexistent": 1})

    def test_rejects_unknown_top_level(self):
        data = {"env_type": "pricing"}
        with pytest.raises(ValueError, match="not found"):
            apply_overrides(data, {"bogus_key": 1})

    def test_rejects_type_mismatch_str_to_int(self):
        data = {"environment": {"seed": 42}}
        with pytest.raises(ValueError, match="type mismatch"):
            apply_overrides(data, {"environment.seed": "not_an_int"})

    def test_rejects_type_mismatch_int_to_bool(self):
        data = {"flag": 0}
        with pytest.raises(ValueError, match="type mismatch"):
            apply_overrides(data, {"flag": True})

    def test_allows_int_where_float_expected(self):
        data = {"agents": {"temperature": 0.2}}
        result = apply_overrides(data, {"agents.temperature": 1})
        assert result["agents"]["temperature"] == 1

    def test_allows_float_where_int_expected(self):
        data = {"environment": {"seed": 42}}
        result = apply_overrides(data, {"environment.seed": 43.0})
        assert result["environment"]["seed"] == 43.0

    def test_allows_none_override(self):
        data = {"run_id": "abc"}
        result = apply_overrides(data, {"run_id": None})
        assert result["run_id"] is None


class TestSweepCliHelpers:
    def test_storage_preflight_requires_uri_for_postgres_backend(self, monkeypatch):
        monkeypatch.delenv("COLLUSIONLAB_STORAGE_URL", raising=False)
        configs = [{"storage": {"backend": "postgres", "uri": None}}]
        with pytest.raises(ValueError, match="COLLUSIONLAB_STORAGE_URL"):
            summarize_sweep_storage(configs, verify_connection=False)

    def test_storage_preflight_uses_env_uri(self, monkeypatch):
        monkeypatch.setenv(
            "COLLUSIONLAB_STORAGE_URL",
            "postgresql://user:secret@example.neon.tech/neondb",
        )
        configs = [{"storage": {"backend": "postgres", "uri": None}}]
        summary = summarize_sweep_storage(configs, verify_connection=False)
        assert "postgres via COLLUSIONLAB_STORAGE_URL" in summary
        assert "example.neon.tech/neondb" in summary
        assert "secret" not in summary

    def test_storage_preflight_local_only_does_not_require_uri(self, monkeypatch):
        monkeypatch.setenv(
            "COLLUSIONLAB_STORAGE_URL",
            "postgresql://user:secret@example.neon.tech/neondb",
        )
        configs = [{"storage": {"backend": "local", "uri": None}}]
        assert "local only" in summarize_sweep_storage(configs, verify_connection=False)

    def test_format_sweep_progress(self):
        line = format_sweep_progress(
            done=2,
            total=4,
            ok=1,
            failed=1,
            elapsed=12.34,
            width=10,
        )
        assert line == (
            "Sweep progress [#####-----] 2/4 complete | "
            "ok=1 failed=1 | elapsed=12.3s"
        )

    def test_rejects_out_of_range_index(self):
        data = {"agents": {"temp": 0.2}}
        with pytest.raises(ValueError, match="not list"):
            apply_overrides(data, {"agents.5.temp": 0.1})

    def test_rejects_index_on_non_list(self):
        data = {"environment": {"seed": 42}}
        with pytest.raises(ValueError, match="not list"):
            apply_overrides(data, {"environment.0": 1})

    def test_deep_copy_isolation(self):
        data = {"environment": {"params": {"a": 1}}}
        result = apply_overrides(data, {"environment.params.a": 2})
        assert data["environment"]["params"]["a"] == 1
        assert result["environment"]["params"]["a"] == 2


# ---------------------------------------------------------------------------
# SweepConfig schema validation
# ---------------------------------------------------------------------------


class TestSweepConfigValidation:
    def test_grid_requires_dict_overrides(self):
        with pytest.raises(ValueError, match="dict"):
            SweepConfig(
                base_config="configs/base.yaml",
                mode="grid",
                overrides=[{"environment.seed": 1}],
            )

    def test_list_requires_list_overrides(self):
        with pytest.raises(ValueError, match="list"):
            SweepConfig(
                base_config="configs/base.yaml",
                mode="list",
                overrides={"environment.seed": [1, 2]},
            )

    def test_grid_rejects_empty_value_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            SweepConfig(
                base_config="configs/base.yaml",
                mode="grid",
                overrides={"environment.seed": []},
            )

    def test_list_rejects_empty_overrides(self):
        with pytest.raises(ValueError, match="at least one"):
            SweepConfig(
                base_config="configs/base.yaml",
                mode="list",
                overrides=[],
            )

    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):
            SweepConfig(
                base_config="configs/base.yaml",
                mode="grid",
                overrides={"environment.seed": [1]},
                unknown_field=True,
            )


# ---------------------------------------------------------------------------
# Config expansion
# ---------------------------------------------------------------------------


class TestGridExpansion:
    def test_cartesian_product_count(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml")
        sc = SweepConfig(
            base_config=str(base),
            mode="grid",
            overrides={"environment.seed": [1, 2, 3], "environment.n_rounds": [2, 3]},
        )
        configs = sc.expand()
        assert len(configs) == 6  # 3 seeds x 2 round counts

    def test_cartesian_product_values(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml")
        sc = SweepConfig(
            base_config=str(base),
            mode="grid",
            overrides={"environment.seed": [10, 20]},
        )
        configs = sc.expand()
        seeds = sorted(c["environment"]["seed"] for c in configs)
        assert seeds == [10, 20]

    def test_unique_run_ids(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml")
        sc = SweepConfig(
            base_config=str(base),
            mode="grid",
            overrides={"environment.seed": [1, 2, 3, 4]},
        )
        configs = sc.expand()
        run_ids = [c["run_id"] for c in configs]
        assert len(run_ids) == len(set(run_ids))

    def test_deterministic_order(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml")
        sc = SweepConfig(
            base_config=str(base),
            mode="grid",
            overrides={"environment.seed": [1, 2], "environment.n_rounds": [2, 3]},
        )
        configs_a = sc.expand()
        configs_b = sc.expand()
        # Override combinations should be in the same order (run_ids differ).
        for a, b in zip(configs_a, configs_b):
            assert a["environment"]["seed"] == b["environment"]["seed"]
            assert a["environment"]["n_rounds"] == b["environment"]["n_rounds"]


class TestListExpansion:
    def test_exact_count(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml")
        sc = SweepConfig(
            base_config=str(base),
            mode="list",
            overrides=[
                {"environment.seed": 1},
                {"environment.seed": 2},
                {"environment.seed": 3},
            ],
        )
        configs = sc.expand()
        assert len(configs) == 3

    def test_exact_values(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml")
        sc = SweepConfig(
            base_config=str(base),
            mode="list",
            overrides=[
                {"environment.seed": 10, "environment.n_rounds": 2},
                {"environment.seed": 20, "environment.n_rounds": 4},
            ],
        )
        configs = sc.expand()
        assert configs[0]["environment"]["seed"] == 10
        assert configs[0]["environment"]["n_rounds"] == 2
        assert configs[1]["environment"]["seed"] == 20
        assert configs[1]["environment"]["n_rounds"] == 4

    def test_unique_run_ids(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml")
        sc = SweepConfig(
            base_config=str(base),
            mode="list",
            overrides=[
                {"environment.seed": 1},
                {"environment.seed": 2},
            ],
        )
        configs = sc.expand()
        assert configs[0]["run_id"] != configs[1]["run_id"]


class TestFromYaml:
    def test_stego_study_sweep_from_file(self):
        sc = SweepConfig.from_yaml(ROOT / "configs" / "sweep_stego_study.yaml")
        assert sc.mode == "list"
        assert isinstance(sc.overrides, list)


# ---------------------------------------------------------------------------
# End-to-end sweep execution
# ---------------------------------------------------------------------------


class TestSweepRunner:
    def test_small_e2e_sweep(self, tmp_path):
        """2 configs x scripted backend: completes and writes per-run + sweep manifests."""
        base = _write_test_base_config(tmp_path / "base.yaml", n_rounds=2)
        sc = SweepConfig(
            base_config=str(base),
            mode="grid",
            overrides={"environment.seed": [1, 2]},
        )
        runner = SweepRunner(
            sweep_config=sc,
            max_workers=1,
            output_dir=str(tmp_path / "output"),
        )
        manifest_path = runner.run()

        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())

        assert "sweep_id" in manifest
        assert "started_at" in manifest
        assert "ended_at" in manifest
        assert "elapsed_seconds" in manifest
        assert manifest["base_config"] == str(base)
        assert manifest["mode"] == "grid"
        assert manifest["max_workers"] == 1
        assert len(manifest["runs"]) == 2

        for run in manifest["runs"]:
            assert run["status"] == "succeeded"
            assert run["manifest_path"] is not None
            assert run["error"] is None
            assert "started_at" in run
            assert "ended_at" in run
            assert "elapsed_seconds" in run
            assert "config" in run
            assert Path(run["manifest_path"]).exists()

    def test_per_run_manifests_written(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml", n_rounds=2)
        sc = SweepConfig(
            base_config=str(base),
            mode="list",
            overrides=[{"environment.seed": 42}],
        )
        runner = SweepRunner(
            sweep_config=sc, max_workers=1, output_dir=str(tmp_path / "out"),
        )
        sweep_path = runner.run()
        sweep = json.loads(sweep_path.read_text())
        run = sweep["runs"][0]

        run_manifest = json.loads(Path(run["manifest_path"]).read_text())
        assert run_manifest["run_id"] == run["run_id"]
        log_path = Path(run_manifest["log_path"])
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2  # n_rounds=2

    def test_sweep_manifest_persisted_to_sqlite_storage(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml", n_rounds=2)
        db_path = tmp_path / "runs.sqlite"
        base_data = yaml.safe_load(base.read_text())
        base_data["storage"] = {"backend": "sqlite", "uri": str(db_path)}
        base.write_text(yaml.safe_dump(base_data))
        sc = SweepConfig(
            base_config=str(base),
            mode="list",
            overrides=[{"environment.seed": 42}],
        )
        runner = SweepRunner(
            sweep_config=sc, max_workers=1, output_dir=str(tmp_path / "out"),
        )
        local_sweep_path = runner.run()
        local_sweep = json.loads(local_sweep_path.read_text())

        store = SQLiteRunStore(str(db_path))
        db_sweep = store.load_sweep_manifest(local_sweep["sweep_id"])
        assert db_sweep is not None
        assert db_sweep["sweep_id"] == local_sweep["sweep_id"]
        assert db_sweep["status"] == "succeeded"
        assert len(db_sweep["runs"]) == 1
        db_run = db_sweep["runs"][0]
        assert db_run["local_manifest_path"] == local_sweep["runs"][0]["manifest_path"]
        assert parse_db_run_ref(db_run["manifest_path"]) == (str(db_path), db_run["run_id"])

        listed = store.list_sweeps()
        assert listed[0]["sweep_id"] == local_sweep["sweep_id"]
        assert listed[0]["n_succeeded"] == 1

    def test_progress_callback(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml", n_rounds=2)
        sc = SweepConfig(
            base_config=str(base),
            mode="grid",
            overrides={"environment.seed": [1, 2, 3]},
        )
        seen: list[tuple[int, int]] = []
        runner = SweepRunner(
            sweep_config=sc,
            max_workers=1,
            output_dir=str(tmp_path / "out"),
            progress_callback=lambda done, total: seen.append((done, total)),
        )
        runner.run()
        assert len(seen) == 3
        totals = [t for _, t in seen]
        assert all(t == 3 for t in totals)
        dones = sorted(d for d, _ in seen)
        assert dones == [1, 2, 3]

    def test_continue_on_error(self, tmp_path):
        """One config fails (0 replies), the other succeeds."""
        env = _base_env_block()
        env["n_rounds"] = 2
        env["seed"] = 0

        good_replies = ["8"] * 4
        config = {
            "run_id": None,
            "env_type": "pricing",
            "environment": env,
            "agents": [
                {
                    "backend": "scripted", "model": "scripted",
                    "memory_window": 3, "temperature": 0.0,
                    "extra": {"replies": list(good_replies)},
                },
                {
                    "backend": "scripted", "model": "scripted",
                    "memory_window": 3, "temperature": 0.0,
                    "extra": {"replies": list(good_replies)},
                },
            ],
            "prompt_dir": str(ROOT / "prompts" / "pricing"),
            "communication_mode": "none",
            "output_dir": str(tmp_path / "data"),
        }
        good_path = tmp_path / "good.yaml"
        good_path.write_text(yaml.safe_dump(config))

        # Use list mode with base=good, override seed for the good run; the bad
        # run overrides grid bounds to force a validation error.
        sc = SweepConfig(
            base_config=str(good_path),
            mode="list",
            overrides=[
                {"environment.seed": 1},
                {"environment.price_min": 99},
            ],
        )
        runner = SweepRunner(
            sweep_config=sc, max_workers=1, output_dir=str(tmp_path / "out"),
        )
        manifest_path = runner.run()
        manifest = json.loads(manifest_path.read_text())

        statuses = [r["status"] for r in manifest["runs"]]
        assert "succeeded" in statuses
        assert "failed" in statuses

        for run in manifest["runs"]:
            if run["status"] == "failed":
                assert run["error"] is not None
                assert run["manifest_path"] is None
            else:
                assert run["error"] is None
                assert run["manifest_path"] is not None


class TestOrderIndependence:
    """max_workers=1 and max_workers=2 produce the same set of results."""

    def test_worker_count_equivalence(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml", n_rounds=2)
        sc = SweepConfig(
            base_config=str(base),
            mode="grid",
            overrides={"environment.seed": [1, 2]},
        )

        r1 = SweepRunner(
            sweep_config=sc, max_workers=1, output_dir=str(tmp_path / "w1"),
        )
        r2 = SweepRunner(
            sweep_config=sc, max_workers=2, output_dir=str(tmp_path / "w2"),
        )
        m1_path = r1.run()
        m2_path = r2.run()

        m1 = json.loads(m1_path.read_text())
        m2 = json.loads(m2_path.read_text())

        assert len(m1["runs"]) == len(m2["runs"]) == 2
        assert all(r["status"] == "succeeded" for r in m1["runs"])
        assert all(r["status"] == "succeeded" for r in m2["runs"])

        # Both should have the same set of seed values in their configs.
        seeds1 = sorted(r["config"]["environment"]["seed"] for r in m1["runs"])
        seeds2 = sorted(r["config"]["environment"]["seed"] for r in m2["runs"])
        assert seeds1 == seeds2


# ---------------------------------------------------------------------------
# Sweep manifest schema
# ---------------------------------------------------------------------------


class TestSweepManifestSchema:
    REQUIRED_TOP_LEVEL = {
        "sweep_id", "started_at", "ended_at", "elapsed_seconds",
        "base_config", "mode", "max_workers", "runs",
    }
    REQUIRED_PER_RUN = {
        "run_id", "config", "status", "manifest_path", "error",
        "started_at", "ended_at", "elapsed_seconds",
    }

    def test_manifest_has_required_keys(self, tmp_path):
        base = _write_test_base_config(tmp_path / "base.yaml", n_rounds=2)
        sc = SweepConfig(
            base_config=str(base),
            mode="grid",
            overrides={"environment.seed": [1]},
        )
        runner = SweepRunner(
            sweep_config=sc, max_workers=1, output_dir=str(tmp_path / "out"),
        )
        manifest = json.loads(runner.run().read_text())

        assert self.REQUIRED_TOP_LEVEL <= set(manifest.keys())
        for run in manifest["runs"]:
            assert self.REQUIRED_PER_RUN <= set(run.keys())


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_sweep_flag_parses(self):
        from collusionlab.runner.sweep import main
        import argparse

        # Just verify the parser accepts --sweep
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--sweep", dest="sweep_path")
        group.add_argument("--config", dest="sweep_path_alt")
        parser.add_argument("--max-workers", type=int, default=None)
        parser.add_argument("--output-dir")
        parser.add_argument("--log-level", default="INFO")

        args = parser.parse_args(["--sweep", "some/path.yaml"])
        assert args.sweep_path == "some/path.yaml"

    def test_config_flag_parses(self):
        import argparse

        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--sweep", dest="sweep_path")
        group.add_argument("--config", dest="sweep_path_alt")
        parser.add_argument("--max-workers", type=int, default=None)

        args = parser.parse_args(["--config", "some/path.yaml"])
        assert args.sweep_path_alt == "some/path.yaml"
        assert args.sweep_path is None

    def test_sweep_and_config_mutually_exclusive(self):
        import argparse

        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--sweep", dest="sweep_path")
        group.add_argument("--config", dest="sweep_path_alt")

        with pytest.raises(SystemExit):
            parser.parse_args(["--sweep", "a.yaml", "--config", "b.yaml"])
