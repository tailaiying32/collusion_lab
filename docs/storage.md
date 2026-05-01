# Experiment Storage

CollusionLab keeps the existing file layout as the default:

```yaml
output_dir: data/raw
storage:
  backend: local
  uri: null
```

For database-backed persistence, configure SQLite:

```yaml
storage:
  backend: sqlite
  uri: sqlite:///data/collusionlab_runs.sqlite
```

The runner still writes the legacy `{output_dir}/{run_id}/log.jsonl` and
`manifest.json` files for compatibility, and also mirrors the run manifest and
per-round logs into the database.  The UI can read from the database by setting:

```powershell
$env:COLLUSIONLAB_STORAGE_URL="sqlite:///data/collusionlab_runs.sqlite"
streamlit run src/collusionlab/ui/app.py
```

This is suitable for a shared database file or mounted volume.  For a true
multi-host server setup, the same storage interface should be extended with a
PostgreSQL backend rather than relying on SQLite over a network filesystem.
