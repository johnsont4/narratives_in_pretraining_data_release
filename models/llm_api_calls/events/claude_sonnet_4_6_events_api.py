import argparse
import json
import os
import sys
import time
import pandas as pd
import anthropic
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import get_output_dir, save_metadata
from event_relation.event_relation_scale_prompt import SYSTEM_PROMPT, TOOL_DEF, DIMENSIONS, build_user_message

PARQUET_PATH = "annotated_data/event_relation_annotations.parquet"
DEFAULT_MODEL = "claude-sonnet-4-6"

VALID_TEMPORAL = {"span1_first", "span2_first", "simultaneous", "same_event", "too_hard_to_tell"}
VALID_CAUSAL = {"direct_cause", "enables", "not_related"}


def _validate_scores(scores: dict, composite_id: str) -> dict:
    validated = dict(scores)
    if validated.get("temporal_order") not in VALID_TEMPORAL:
        print(f"  WARNING: invalid temporal_order '{validated.get('temporal_order')}' for {composite_id} — set to None")
        validated["temporal_order"] = None
    if validated.get("causality_rating") not in VALID_CAUSAL:
        print(f"  WARNING: invalid causality_rating '{validated.get('causality_rating')}' for {composite_id} — set to None")
        validated["causality_rating"] = None
    return validated


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
    return anthropic.Anthropic(api_key=api_key)


def build_id_to_pair(df: pd.DataFrame) -> dict:
    """Map composite IDs to pair metadata for all neighboring event pairs in df."""
    id_to_pair = {}
    for _, row in df.iterrows():
        spans = json.loads(row["event_spans"])
        for i in range(len(spans) - 1):
            composite_id = f"{row['safe_instance_id']}__pair_{i}"
            id_to_pair[composite_id] = {
                "safe_instance_id": row["safe_instance_id"],
                "pair_idx": i,
                "sampled_text": row["sampled_text"],
                "span1": spans[i],
                "span2": spans[i + 1],
            }
    return id_to_pair


def poll_until_complete(client: anthropic.Anthropic, batch_id: str, poll_interval: int) -> None:
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
            print(f"Network hiccup ({type(e).__name__}), retrying in {poll_interval}s...")
            time.sleep(poll_interval)
            continue
        c = batch.request_counts
        print(
            f"[{batch.processing_status}] "
            f"processing={c.processing}  succeeded={c.succeeded}  "
            f"errored={c.errored}  expired={c.expired}"
        )
        if batch.processing_status == "ended":
            return
        time.sleep(poll_interval)


def collect_results(client: anthropic.Anthropic, batch_id: str, id_to_pair: dict) -> list:
    rows = []
    for result in client.messages.batches.results(batch_id):
        composite_id = result.custom_id
        stored = id_to_pair.get(composite_id, {})
        row = {
            "safe_instance_id": stored.get("safe_instance_id", composite_id),
            "pair_idx": stored.get("pair_idx"),
            "span1": json.dumps(stored.get("span1")),
            "span2": json.dumps(stored.get("span2")),
            "sampled_text": stored.get("sampled_text", ""),
        }
        if result.result.type == "succeeded":
            scores = {}
            for block in result.result.message.content:
                if block.type == "tool_use":
                    scores = _validate_scores(dict(block.input), composite_id)
                    break
            row.update({f"pred_{dim}": scores.get(dim) for dim in DIMENSIONS})
        else:
            print(f"  {composite_id}: {result.result.type}")
            row.update({f"pred_{dim}": None for dim in DIMENSIONS})
        rows.append(row)
    return rows


def run_realtime_pairs(client: anthropic.Anthropic, submit_pairs: dict, model: str) -> list:
    rows = []
    total = len(submit_pairs)
    for i, (composite_id, pair) in enumerate(submit_pairs.items(), 1):
        print(f"  [{i}/{total}] {pair['safe_instance_id']} pair {pair['pair_idx']} ...", end=" ", flush=True)
        row = {
            "safe_instance_id": pair["safe_instance_id"],
            "pair_idx": pair["pair_idx"],
            "span1": json.dumps(pair["span1"]),
            "span2": json.dumps(pair["span2"]),
            "sampled_text": pair["sampled_text"],
        }
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                tools=[{**TOOL_DEF, "cache_control": {"type": "ephemeral"}}],
                tool_choice={"type": "tool", "name": "annotate_event_relation"},
                messages=[{"role": "user", "content": build_user_message(
                    pair["sampled_text"],
                    json.dumps(pair["span1"]),
                    json.dumps(pair["span2"]),
                )}],
            )
            scores = {}
            for block in response.content:
                if block.type == "tool_use":
                    scores = _validate_scores(dict(block.input), composite_id)
                    break
            row.update({f"pred_{dim}": scores.get(dim) for dim in DIMENSIONS})
            print("ok")
        except Exception as e:
            row.update({f"pred_{dim}": None for dim in DIMENSIONS})
            print(f"error: {e}")
        rows.append(row)
    return rows


def aggregate_text_scores(pair_rows: list, id_to_n_spans: dict) -> pd.DataFrame:
    counts = defaultdict(lambda: {"n_pairs": 0, "n_sequential": 0, "n_causal": 0})
    seen_ids = set()

    for row in pair_rows:
        sid = row["safe_instance_id"]
        seen_ids.add(sid)
        counts[sid]["n_pairs"] += 1
        if row.get("pred_temporal_order") in ("span1_first", "span2_first"):
            counts[sid]["n_sequential"] += 1
        if row.get("pred_causality_rating") in ("direct_cause", "enables"):
            counts[sid]["n_causal"] += 1

    text_rows = []
    for sid in seen_ids:
        n_spans = id_to_n_spans.get(sid, 0)
        n_neighboring = max(0, n_spans - 1)
        c = counts[sid]
        n_pairs = c["n_pairs"]
        n_seq = c["n_sequential"]
        n_caus = c["n_causal"]
        text_rows.append({
            "safe_instance_id": sid,
            "n_event_spans": n_spans,
            "n_neighboring_pairs": n_neighboring,
            "n_pairs": n_pairs,
            "n_sequential": n_seq,
            "n_causal": n_caus,
            "temporal_sequencing": n_seq / n_pairs if n_pairs > 0 else None,
            "causal_density": n_caus / n_pairs if n_pairs > 0 else None,
        })

    return pd.DataFrame(text_rows)


def run(
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    poll_interval: int = 60,
    batch_id: str | None = None,
    realtime: bool = False,
) -> tuple[str, str]:
    start_time = datetime.now()
    client = get_client()

    full_df = pd.read_parquet(PARQUET_PATH)
    id_to_n_spans = {
        row["safe_instance_id"]: len(json.loads(row["event_spans"]))
        for _, row in full_df.iterrows()
    }

    # Build id_to_pair from full_df so resume always works
    id_to_pair = build_id_to_pair(full_df)

    pair_rows: list = []

    if batch_id:
        print(f"Resuming batch {batch_id} ...")
    else:
        submit_df = full_df.head(limit) if limit else full_df
        n_skipped = (submit_df["event_spans"].apply(lambda x: len(json.loads(x))) < 2).sum()
        if n_skipped:
            print(f"Warning: {n_skipped} text(s) have fewer than 2 event spans and will contribute no pairs.")

        submit_ids = set(submit_df["safe_instance_id"])
        submit_pairs = {cid: p for cid, p in id_to_pair.items() if p["safe_instance_id"] in submit_ids}

        if realtime:
            print(f"Running {len(submit_pairs)} pair requests in real-time ({model}) ...")
            pair_rows = run_realtime_pairs(client, submit_pairs, model)
            batch_id = None
        else:
            print(f"Submitting {len(submit_pairs)} pair requests from {len(submit_df)} texts ({model}) ...")
            batch = client.messages.batches.create(requests=[
                {
                    "custom_id": composite_id,
                    "params": {
                        "model": model,
                        "max_tokens": 256,
                        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                        "tools": [{**TOOL_DEF, "cache_control": {"type": "ephemeral"}}],
                        "tool_choice": {"type": "tool", "name": "annotate_event_relation"},
                        "messages": [{"role": "user", "content": build_user_message(
                            pair["sampled_text"],
                            json.dumps(pair["span1"]),
                            json.dumps(pair["span2"]),
                        )}],
                    },
                }
                for composite_id, pair in submit_pairs.items()
            ])
            batch_id = batch.id
            print(f"Batch ID: {batch_id}  (resume: --batch-id {batch_id})")

    if batch_id is not None:
        print("Polling for completion ...")
        poll_until_complete(client, batch_id, poll_interval)
        print("Retrieving results ...")
        pair_rows = collect_results(client, batch_id, id_to_pair)

    end_time = datetime.now()
    duration = round((end_time - start_time).total_seconds(), 1)

    pair_df = pd.DataFrame(pair_rows)
    text_df = aggregate_text_scores(pair_rows, id_to_n_spans)

    output_dir = get_output_dir(model, output_base="event_relation/outputs")
    pair_path = output_dir / "pair_predictions.csv"
    text_path = output_dir / "text_scores.csv"
    pair_df.to_csv(pair_path, index=False)
    text_df.to_csv(text_path, index=False)
    save_metadata(output_dir, {
        "model": model,
        "n_texts": int(text_df.shape[0]),
        "n_pairs": int(pair_df.shape[0]),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "batch_id": batch_id,
    })

    print(f"Done in {duration}s → {output_dir}")
    return str(pair_path), str(text_path)


def main():
    parser = argparse.ArgumentParser(
        description="Scale event-relation annotation to all neighboring event pairs per text."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N texts (default: all 193)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Claude model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--poll", type=int, default=60, help="Seconds between status checks (default: 60)")
    parser.add_argument("--batch-id", type=str, default=None, help="Resume an existing batch by ID")
    parser.add_argument("--realtime", action="store_true", help="Call API immediately (no Batches API); cannot be used with --batch-id")
    args = parser.parse_args()

    if args.realtime and args.batch_id:
        parser.error("--realtime cannot be used with --batch-id")

    pair_path, text_path = run(args.limit, args.model, args.poll, batch_id=args.batch_id, realtime=args.realtime)

    text_df = pd.read_csv(text_path)
    print("\nText-level scores summary:")
    print(text_df[["safe_instance_id", "n_event_spans", "n_pairs", "temporal_sequencing", "causal_density"]].to_string(index=False))


if __name__ == "__main__":
    main()
