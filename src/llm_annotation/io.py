"""Run bookkeeping: the unit of work, output directories, checkpoints, saving.

Output bases resolve against the repository root, not the working directory, so
a script started from anywhere writes to the same place.
"""

import json
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: llm_annotation/ -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Item:
    """One thing to annotate: a stable key, the user turn, and output columns.

    `key` is the resume key and the batch custom_id, so it must be unique
    across the run — passage grain for Likert, pair grain for event relations.
    `meta` is carried straight through to the output row.
    """

    key: str
    message: str
    meta: dict = field(default_factory=dict)


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(" ", "_")


def get_output_dir(model_name: str, output_base: str = "outputs") -> Path:
    base = Path(output_base)
    if not base.is_absolute():
        base = REPO_ROOT / base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base / sanitize_model_name(model_name) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_predictions(df: pd.DataFrame, output_dir: Path) -> str:
    path = Path(output_dir) / "predictions.csv"
    df.to_csv(path, index=False)
    return str(path)


def save_metadata(output_dir: Path, metadata: dict) -> None:
    path = Path(output_dir) / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def checkpoint_interval(total: int, floor_: int = 50) -> int:
    """Rows between checkpoint writes, scaled to the job size.

    Each checkpoint rewrites the whole accumulated CSV, so a fixed interval is
    quadratic in the row count: 5k rows at every-50 is 100 rewrites of a file
    ending at ~5 MB, but 60k rows is 1,200 rewrites of one ending at ~60 MB.
    Capping the count at ~200 keeps resume granularity useful while the write
    cost stays flat.
    """
    return max(floor_, total // 200)


def load_checkpoint(path: Path, key_of) -> tuple[list[dict], set]:
    """Rows already written by a previous run, and the keys they cover.

    `key_of` rebuilds an Item key from an output row, so resume uses the same
    identity the run does rather than a positional guess.
    """
    if not Path(path).exists():
        return [], set()
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    done = {key_of(row) for row in rows}
    print(f"Resuming: {len(done)} already completed.")
    return rows, done
