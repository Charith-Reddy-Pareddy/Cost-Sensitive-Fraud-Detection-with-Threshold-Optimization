"""Pin MLflow's tracking store to this project's own directory.

Without this, MLflow resolves its default tracking URI relative to the process's ambient
working directory rather than this project — which, run from a shell whose cwd is a sibling
project, silently points experiment tracking at that *other* project's store. Setting an
explicit, absolute, project-rooted URI makes tracking deterministic regardless of cwd.

This project's MLflow version (3.x) has put the plain filesystem store into maintenance mode
and requires a database-backed store, so this uses a project-local SQLite file rather than
`file:./mlruns`.
"""

from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DB = PROJECT_ROOT / "mlflow.db"


def use_local_tracking_store() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
