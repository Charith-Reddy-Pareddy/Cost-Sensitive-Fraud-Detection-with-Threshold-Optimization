"""Pin MLflow's tracking store to this project's own `mlruns/` directory.

Without this, MLflow resolves its default tracking URI relative to the process's ambient
working directory rather than this project — which, run from a shell whose cwd is a sibling
project, silently points experiment tracking at that *other* project's store. Setting an
explicit, absolute, project-rooted URI makes tracking deterministic regardless of cwd.
"""

from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = PROJECT_ROOT / "mlruns"


def use_local_tracking_store() -> None:
    mlflow.set_tracking_uri(f"file:{TRACKING_DIR}")
