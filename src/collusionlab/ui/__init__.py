"""CollusionLab UI package.

Streamlit-based analysis dashboard for inspecting experiment runs.

Launch:
    streamlit run src/collusionlab/ui/app.py
"""

from collusionlab.ui.data_loading import (
    extract_trajectory_df,
    list_runs,
    load_log_rows,
    load_manifest,
)

__all__ = [
    "list_runs",
    "load_manifest",
    "load_log_rows",
    "extract_trajectory_df",
]
