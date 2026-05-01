# Experiment Storage

CollusionLab keeps the existing file layout as the default:

```yaml
output_dir: data/raw
storage:
  backend: local
  uri: null
```

For laptop-to-HPC persistence, use managed Postgres. Neon is the recommended
default because CollusionLab writes data in intermittent bursts and Neon can
scale compute to zero when idle.

Create a Neon project, copy the pooled connection string, and store it outside
git:

```powershell
$env:COLLUSIONLAB_STORAGE_URL="postgresql://USER:PASSWORD@HOST/collusionlab?sslmode=require"
streamlit run src/collusionlab/ui/app.py
```

On the HPC cluster, export the same value in the job script before launching the
experiment:

```bash
export COLLUSIONLAB_STORAGE_URL='postgresql://USER:PASSWORD@HOST/collusionlab?sslmode=require'
PYTHONPATH=src mamba run -n collusion_lab python -m collusionlab.runner.experiment --config configs/base.yaml --storage-uri "$COLLUSIONLAB_STORAGE_URL"
```

You can also put non-secret backend selection in YAML and omit the CLI flag:

```yaml
storage:
  backend: postgres
  uri: null  # read from COLLUSIONLAB_STORAGE_URL
```

For local database-backed development, configure SQLite:

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

SQLite is suitable for a shared database file or mounted volume.  Use Postgres
for true multi-host server/laptop access.
