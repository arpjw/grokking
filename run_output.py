"""Create a fresh directory for each training run."""
from __future__ import annotations

import os
import re


def prepare_run_dir(run_name: str, root: str = "results") -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError("run_name must be a single, nonempty directory name")

    os.makedirs(root, exist_ok=True)
    out_dir = os.path.join(root, run_name)
    try:
        os.mkdir(out_dir)
    except FileExistsError as exc:
        raise FileExistsError(
            f"{out_dir} already exists; choose a new run_name to preserve its results"
        ) from exc
    return out_dir
