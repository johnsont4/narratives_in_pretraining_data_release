"""Where the annotation scripts get their inputs.

Two Hugging Face datasets back every run in this repository, and both are
downloaded manually (see the README):

  annotated_data/   CLS-Lab/narrative-gold-annotations
                    Human-adjudicated labels. One randomly sampled event pair
                    per passage on the event-relation side — `assigned_span1`
                    and `assigned_span2` are that pair.

  llm_labels/       CLS-Lab/narrative-llm-annotations
                    The Gemma scale-up pseudo-labels. The *scale* annotators
                    read their input passages from here with the `pred_*`
                    columns dropped, because the passages themselves are not
                    published separately. Re-running an annotator therefore
                    re-derives the labels over exactly the passages the paper
                    used.

Every path resolves against the repository root, so scripts run from any
working directory.
"""

import pandas as pd
from pathlib import Path

#: llm_annotation/ -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

GOLD_DIR = REPO_ROOT / "annotated_data"
LLM_DIR = REPO_ROOT / "llm_labels"

GOLD_REPO = "CLS-Lab/narrative-gold-annotations"
LLM_REPO = "CLS-Lab/narrative-llm-annotations"

#: Config directories inside the LLM-annotations snapshot.
PASSAGES_CONFIG = "agency_setting_llm_labels"
PAIRS_CONFIG = "event_relation_llm_labels"


def _download_hint(repo: str, local_dir: Path, allow: str) -> str:
    return (
        f"\n\nDownload it first:\n\n"
        f"    from huggingface_hub import snapshot_download\n"
        f"    snapshot_download(\n"
        f"        {repo!r}, repo_type=\"dataset\",\n"
        f"        local_dir={str(local_dir)!r},\n"
        f"        allow_patterns=[{allow}],\n"
        f"    )\n"
    )


# ── gold annotations ─────────────────────────────────────────────────────────

def gold_path(name: str) -> Path:
    """Path to one gold parquet: 'agency', 'setting', or 'event_relation'."""
    path = GOLD_DIR / f"{name}_annotations.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No gold annotations at {path}."
            + _download_hint(GOLD_REPO, GOLD_DIR, '"*_annotations.parquet"')
        )
    return path


def load_gold(name: str, n: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(gold_path(name))
    if n is not None:
        df = df.head(n)
    print(f"Loaded {len(df):,} gold {name} rows from {gold_path(name).name}")
    return df.reset_index(drop=True)


# ── scale-up inputs ──────────────────────────────────────────────────────────

def _read_config(config: str) -> pd.DataFrame:
    """Read every parquet shard of one LLM-annotations config."""
    shards = sorted((LLM_DIR / config).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No parquet shards under {LLM_DIR / config}."
            + _download_hint(LLM_REPO, LLM_DIR, f'"{config}/*"')
        )
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def _drop_preds(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the published labels, leaving the input the annotator was given.

    The scale-up passages were never published on their own, only alongside the
    Gemma predictions over them. Dropping `pred_*` recovers the annotator's
    input exactly; keeping it would leak the labels being reproduced.
    """
    return df[[c for c in df.columns if not c.startswith("pred_")]]


def load_scale_passages(n: int | None = None, path: str | Path | None = None) -> pd.DataFrame:
    """The 25k Likert scale-up passages: `id`, `chunk_index`, `sampled_text`.

    A corpus row is a passage, so the key is (id, chunk_index) — `id` alone is
    a document id and repeats across the passages sampled from that document.
    """
    df = _drop_preds(_read_config(PASSAGES_CONFIG)) if path is None else pd.read_parquet(path)

    missing = [c for c in ("id", "sampled_text") if c not in df.columns]
    if missing:
        raise KeyError(f"Scale passages are missing required column(s): {missing}")
    if n is not None:
        df = df.head(n)
    print(f"Loaded {len(df):,} scale-up passages")
    return df.reset_index(drop=True)


def load_scale_pairs(n: int | None = None, path: str | Path | None = None) -> pd.DataFrame:
    """The 61k event pairs: `id`, `chunk_index`, `pair_idx`, `span1`, `span2`, `sampled_text`.

    These are every *neighboring* event-span pair of each scale-up passage —
    the LLM side annotates all of them, unlike the human gold set, which holds
    one randomly sampled pair per passage. The pairs are already enumerated
    here, so no `event_spans` column is needed to rebuild them.

    `n` limits *passages*, not pairs, so a run never stops midway through one
    passage's pairs and leave it partially scored.
    """
    df = _drop_preds(_read_config(PAIRS_CONFIG)) if path is None else pd.read_parquet(path)

    missing = [c for c in ("id", "pair_idx", "span1", "span2", "sampled_text")
               if c not in df.columns]
    if missing:
        raise KeyError(f"Scale pairs are missing required column(s): {missing}")

    if n is not None:
        key = passage_key_cols(df)
        keep = df[key].drop_duplicates().head(n)
        df = df.merge(keep, on=key, how="inner")

    n_passages = len(df[passage_key_cols(df)].drop_duplicates())
    print(f"Loaded {len(df):,} event pairs across {n_passages:,} passages")
    return df.reset_index(drop=True)


def passage_key_cols(df: pd.DataFrame) -> list[str]:
    """['id', 'chunk_index'], or ['id'] for a document-grain frame."""
    return ["id", "chunk_index"] if "chunk_index" in df.columns else ["id"]


def passage_key(row) -> tuple:
    """The (id, chunk_index) tuple for one row, tolerating document grain."""
    if "chunk_index" in row and pd.notna(row["chunk_index"]):
        return (row["id"], int(row["chunk_index"]))
    return (row["id"],)


def composite_id(key: tuple, pair_idx: int) -> str:
    """Stable per-pair id, unique across a document's passages.

    Must include chunk_index: `id` alone repeats across the passages sampled
    from one document, so `f"{id}__pair_0"` would collide between them and a
    dict keyed on it would silently keep only the last passage's pairs.
    """
    return "::".join(str(k) for k in key) + f"__pair_{pair_idx}"
