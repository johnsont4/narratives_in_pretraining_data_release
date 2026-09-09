"""Annotate temporal order and causal relation between pairs of event spans.

Two splits, which differ in more than size:

    gold   the 440 human-annotated pairs — one randomly sampled pair per
           passage, the pair the annotators saw. Also asks whether each span
           is a true event trigger, as the humans were asked.
    scale  all 61,444 neighboring pairs of the pseudo-label passages. Both
           spans are taken as given, so only the two relation judgments are
           made.

The scale split also aggregates to one row per passage (`text_scores.csv`),
giving the temporal-sequencing and causal-density rates NarraBERT trains on.

    python src/llm_annotation/annotate_events.py --model claude --split gold
    python src/llm_annotation/annotate_events.py --model gemma --split scale --workers 20
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_annotation import clients, data, io
from llm_annotation.prompts import event_relation

MAX_TOKENS = 256


def build_items(df: pd.DataFrame, split: str, variant) -> list[io.Item]:
    """One Item per pair.

    Gold rows carry exactly one pair, in `assigned_span1` / `assigned_span2`.
    Scale rows are already one-pair-per-row, keyed by (id, chunk_index, pair_idx),
    so neither split has to enumerate pairs from an `event_spans` column.
    """
    items = []
    for _, row in df.iterrows():
        if split == "gold":
            key = row["safe_instance_id"]
            span1, span2 = row["assigned_span1"], row["assigned_span2"]
            meta = {"safe_instance_id": key, "sampled_text": row["sampled_text"],
                    "assigned_span1": span1, "assigned_span2": span2}
        else:
            parts = data.passage_key(row)
            pair_idx = int(row["pair_idx"])
            key = data.composite_id(parts, pair_idx)
            span1, span2 = row["span1"], row["span2"]
            meta = dict(zip(data.passage_key_cols(df), parts))
            meta.update({"pair_idx": pair_idx, "span1": span1, "span2": span2,
                         "sampled_text": row["sampled_text"]})
        items.append(io.Item(key=key,
                             message=variant.build_user_message(row["sampled_text"], span1, span2),
                             meta=meta))
    return items


def to_row(item: io.Item, scores: dict | None, variant) -> dict:
    row = dict(item.meta)
    validated = (variant.validate(scores, item.key) if scores
                 else {d: None for d in variant.dimensions})
    row.update({f"pred_{dim}": validated[dim] for dim in variant.dimensions})
    return row


def aggregate_text_scores(pair_rows: list) -> pd.DataFrame:
    """Aggregate pair predictions to one row per *passage*.

    Keyed on (id, chunk_index): `id` alone is a document id and repeats across
    the passages sampled from one document, so keying on it would pool pairs
    across passages and put this output at a coarser grain than the Likert
    scores it has to join against.

    `n_pairs` counts the pairs that actually scored, so it is not the same as
    the passage's neighboring-pair count — unparseable responses are dropped.
    """
    has_chunk = bool(pair_rows) and "chunk_index" in pair_rows[0]
    counts = defaultdict(lambda: {"n_pairs": 0, "n_sequential": 0, "n_causal": 0})
    seen = []

    for row in pair_rows:
        key = (row["id"], int(row["chunk_index"])) if has_chunk else (row["id"],)
        if key not in counts:
            seen.append(key)
        counts[key]["n_pairs"] += 1
        if row.get("pred_temporal_order") in ("span1_first", "span2_first"):
            counts[key]["n_sequential"] += 1
        if row.get("pred_causality_rating") in ("direct_cause", "enables"):
            counts[key]["n_causal"] += 1

    key_cols = ["id", "chunk_index"] if has_chunk else ["id"]
    out = []
    for key in seen:
        c = counts[key]
        n_pairs, n_seq, n_caus = c["n_pairs"], c["n_sequential"], c["n_causal"]
        row = dict(zip(key_cols, key))
        row.update({
            "n_pairs": n_pairs, "n_sequential": n_seq, "n_causal": n_caus,
            "temporal_sequencing": n_seq / n_pairs if n_pairs else None,
            "causal_density": n_caus / n_pairs if n_pairs else None,
        })
        out.append(row)
    return pd.DataFrame(out)


def run(model_key: str, split: str, n: int | None, workers: int, poll: int,
        model_id: str | None = None, batch_id: str | None = None,
        realtime: bool = False, output_dir: Path | None = None,
        input_path: str | None = None) -> str:
    start = datetime.now()
    model = clients.resolve(model_key, model_id)
    variant = event_relation.variant(split, model.style)

    # `n` counts passages, not pairs, so a scale run never stops midway through
    # one passage and leaves it partially scored.
    df = (data.load_gold("event_relation", n=n) if split == "gold"
          else data.load_scale_pairs(n=n, path=input_path))
    items = build_items(df, split, variant)
    by_key = {item.key: item for item in items}
    print(f"Prompt variant: {variant.name} | {len(items):,} pairs")

    if output_dir is None:
        output_dir = io.get_output_dir(model.model_id, f"outputs/events_{split}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / ("predictions.csv" if split == "gold" else "pair_predictions.csv")
    print(f"Output directory: {output_dir}")

    client = clients.get_client(model)

    if model.runner == "batch":
        if batch_id:
            print(f"Resuming batch {batch_id} ...")
        elif realtime:
            results = clients.run_realtime(client, items, model,
                                           variant.system, variant.tool_def, MAX_TOKENS)
        else:
            batch_id = clients.submit_batch(client, items, model,
                                            variant.system, variant.tool_def, MAX_TOKENS)
            print(f"Batch ID: {batch_id}  (resume: --batch-id {batch_id})")

        if batch_id:
            clients.poll_batch(client, batch_id, poll)
            results = clients.collect_batch(client, batch_id)

        rows = [to_row(by_key[k], v, variant) for k, v in results.items() if k in by_key]
        missing = set(results) - set(by_key)
        if missing:
            print(f"Note: {len(missing)} batch results are outside the current selection "
                  f"and were skipped — rerun without --n to collect them.")
    else:
        # Rebuild an Item key from a checkpointed row, so resume uses the same
        # identity the run does.
        passage_cols = data.passage_key_cols(df)

        def key_of(row):
            if split == "gold":
                return row["safe_instance_id"]
            parts = tuple(row[c] for c in passage_cols)
            return data.composite_id(parts, int(row["pair_idx"]))

        rows, done = io.load_checkpoint(pair_path, key_of)
        todo = [i for i in items if i.key not in done]
        print(f"Total: {len(items)} | Completed: {len(done)} | Remaining: {len(todo)}")
        every = io.checkpoint_interval(len(items))

        def on_result(key, scores, n_done):
            rows.append(to_row(by_key[key], scores, variant))
            print(f"[{len(rows)}/{len(items)}] {key} done")
            if n_done % every == 0:
                pd.DataFrame(rows).to_csv(pair_path, index=False)
                print(f"  Checkpoint: {len(rows)} saved "
                      f"({round((datetime.now()-start).total_seconds(),1)}s elapsed)")

        clients.run_threaded(client, todo, model, variant.system, workers, poll, on_result)
        batch_id = None

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(pair_path, index=False)

    text_path = None
    if split == "scale":
        text_df = aggregate_text_scores(rows)
        text_path = output_dir / "text_scores.csv"
        text_df.to_csv(text_path, index=False)
        print(f"Aggregated {len(pair_df):,} pairs to {len(text_df):,} passages")

    duration = round((datetime.now() - start).total_seconds(), 1)
    io.save_metadata(output_dir, {
        "task": "event_relation", "split": split, "variant": variant.name,
        "model": model.model_id, "provider": model.provider, "style": model.style,
        "input": input_path or ("gold: event_relation_annotations.parquet" if split == "gold"
                                else f"{data.LLM_REPO}:{data.PAIRS_CONFIG} (labels dropped)"),
        "n_pairs": len(pair_df),
        "start_time": start.isoformat(), "end_time": datetime.now().isoformat(),
        "duration_seconds": duration, "batch_id": batch_id,
    })
    print(f"Done in {duration}s → {output_dir}")
    return str(text_path or pair_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="claude", choices=sorted(clients.MODELS),
                   help="Annotator (default: claude)")
    p.add_argument("--split", default="gold", choices=["gold", "scale"],
                   help="gold = 440 human-annotated pairs; scale = 61,444 neighboring pairs")
    p.add_argument("--n", type=int, default=None, help="Limit to the first N passages")
    p.add_argument("--model-id", default=None, help="Override the provider's model id")
    p.add_argument("--workers", type=int, default=20, help="Concurrent requests, threaded models (default: 20)")
    p.add_argument("--poll", type=int, default=None,
                   help="Seconds between status checks (default: 60 batched, 2 threaded)")
    p.add_argument("--batch-id", default=None, help="Resume an Anthropic batch by id")
    p.add_argument("--realtime", action="store_true",
                   help="Call Anthropic per item instead of batching")
    p.add_argument("--output-dir", default=None,
                   help="Output directory; pass a prior run's path to resume it")
    p.add_argument("--input", default=None, help="Parquet of event pairs (scale split)")
    args = p.parse_args()

    if args.realtime and args.batch_id:
        p.error("--realtime cannot be used with --batch-id")

    model = clients.resolve(args.model, args.model_id)
    poll = args.poll if args.poll is not None else (60 if model.runner == "batch" else 2)

    out = run(args.model, args.split, args.n, args.workers, poll,
              model_id=args.model_id, batch_id=args.batch_id, realtime=args.realtime,
              output_dir=Path(args.output_dir) if args.output_dir else None,
              input_path=args.input)

    df = pd.read_csv(out)
    print(f"\n{Path(out).name} ({len(df):,} rows), first 20:")
    print(df[[c for c in df.columns if c != "sampled_text"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
