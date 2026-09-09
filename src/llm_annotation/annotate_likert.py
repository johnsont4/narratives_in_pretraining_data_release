"""Score passages on the nine agency/setting dimensions, 1-5.

Two splits:

    gold   the 400 human-annotated passages, for comparing a model against
           the human labels
    scale  the 25,000 pseudo-label passages that train NarraBERT

The prompt is the same for both; only the input and the output key differ —
gold rows are keyed by `safe_instance_id`, scale rows by (id, chunk_index).

    python src/llm_annotation/annotate_likert.py --model claude --split gold
    python src/llm_annotation/annotate_likert.py --model gemma --split scale --workers 20
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_annotation import clients, data, io
from llm_annotation.prompts import likert

MAX_TOKENS = 512


def build_items(df: pd.DataFrame, split: str, style: str) -> list[io.Item]:
    """One Item per passage, keyed by whichever identity the split carries."""
    items = []
    for _, row in df.iterrows():
        if split == "gold":
            key = row["safe_instance_id"]
            meta = {"safe_instance_id": key}
        else:
            parts = data.passage_key(row)
            key = "::".join(str(p) for p in parts)
            meta = dict(zip(data.passage_key_cols(df), parts))
        meta["sampled_text"] = row["sampled_text"]
        items.append(io.Item(key=key, message=likert.build_user_message(row["sampled_text"], style),
                             meta=meta))
    return items


def to_row(item: io.Item, scores: dict | None) -> dict:
    row = dict(item.meta)
    validated = likert.validate(scores, item.key) if scores else {d: None for d in likert.ALL_DIMENSIONS}
    row.update({f"pred_{dim}": validated[dim] for dim in likert.ALL_DIMENSIONS})
    return row


def run(model_key: str, split: str, n: int | None, workers: int, poll: int,
        model_id: str | None = None, batch_id: str | None = None,
        realtime: bool = False, output_dir: Path | None = None,
        input_path: str | None = None) -> str:
    start = datetime.now()
    model = clients.resolve(model_key, model_id)

    df = (data.load_gold("agency", n=n) if split == "gold"
          else data.load_scale_passages(n=n, path=input_path))
    items = build_items(df, split, model.style)
    by_key = {item.key: item for item in items}

    if output_dir is None:
        output_dir = io.get_output_dir(model.model_id, f"outputs/likert_{split}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    print(f"Output directory: {output_dir}")

    client = clients.get_client(model)

    if model.runner == "batch":
        if batch_id:
            print(f"Resuming batch {batch_id} ...")
        elif realtime:
            results = clients.run_realtime(client, items, model,
                                           likert.SYSTEM_PROMPT, likert.TOOL_DEF, MAX_TOKENS)
        else:
            batch_id = clients.submit_batch(client, items, model,
                                            likert.SYSTEM_PROMPT, likert.TOOL_DEF, MAX_TOKENS)
            print(f"Batch ID: {batch_id}  (resume: --batch-id {batch_id})")

        if batch_id:
            clients.poll_batch(client, batch_id, poll)
            results = clients.collect_batch(client, batch_id)

        # A resumed batch may return ids beyond the current --n, so fall back to
        # a bare Item rather than dropping the row.
        rows = [to_row(by_key.get(k, io.Item(k, "", {"safe_instance_id": k})), v)
                for k, v in results.items()]
    else:
        key_cols = ["safe_instance_id"] if split == "gold" else data.passage_key_cols(df)
        rows, done = io.load_checkpoint(
            predictions_path, lambda row: "::".join(str(row[c]) for c in key_cols)
        )
        todo = [i for i in items if i.key not in done]
        print(f"Total: {len(items)} | Completed: {len(done)} | Remaining: {len(todo)}")
        every = io.checkpoint_interval(len(items))

        def on_result(key, scores, n_done):
            rows.append(to_row(by_key[key], scores))
            print(f"[{len(rows)}/{len(items)}] {key} done")
            if n_done % every == 0:
                pd.DataFrame(rows).to_csv(predictions_path, index=False)
                print(f"  Checkpoint: {len(rows)} saved "
                      f"({round((datetime.now()-start).total_seconds(),1)}s elapsed)")

        clients.run_threaded(client, todo, model, likert.SYSTEM_PROMPT, workers, poll, on_result)
        batch_id = None

    results_df = pd.DataFrame(rows)
    results_df.to_csv(predictions_path, index=False)
    duration = round((datetime.now() - start).total_seconds(), 1)
    io.save_metadata(output_dir, {
        "task": "likert", "split": split, "model": model.model_id,
        "provider": model.provider, "style": model.style,
        "input": input_path or ("gold: agency_annotations.parquet" if split == "gold"
                                else f"{data.LLM_REPO}:{data.PASSAGES_CONFIG} (labels dropped)"),
        "n_instances": len(results_df),
        "start_time": start.isoformat(), "end_time": datetime.now().isoformat(),
        "duration_seconds": duration, "batch_id": batch_id,
    })
    print(f"Done in {duration}s → {predictions_path}")
    return str(predictions_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="claude", choices=sorted(clients.MODELS),
                   help="Annotator (default: claude)")
    p.add_argument("--split", default="gold", choices=["gold", "scale"],
                   help="gold = 400 human-annotated passages; scale = 25,000 pseudo-label passages")
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
    p.add_argument("--input", default=None,
                   help="Parquet of passages, with id/chunk_index/sampled_text (scale split)")
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
    key = "safe_instance_id" if args.split == "gold" else "id"
    cols = [c for c in [key] + [f"pred_{d}" for d in likert.ALL_DIMENSIONS] if c in df.columns]
    print(f"\nScores ({len(df):,} passages), first 20:")
    print(df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
