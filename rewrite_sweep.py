import json
import os
from pathlib import Path

from collusionlab.runner.sweep import db_persistable_sweep_manifest

sweep_path = Path(r"data/raw/sweep_9ae3d8d5-3675-4637-9ca0-6577200e8f65/sweep_manifest.json")
uri = os.environ["COLLUSIONLAB_STORAGE_URL"]

m = json.loads(sweep_path.read_text(encoding="utf-8"))
db_m = db_persistable_sweep_manifest(m, uri)
sweep_path.write_text(json.dumps(db_m, indent=2, sort_keys=True), encoding="utf-8")

print("rewritten:", sweep_path)
print("first manifest_path:", db_m["runs"][0]["manifest_path"])
