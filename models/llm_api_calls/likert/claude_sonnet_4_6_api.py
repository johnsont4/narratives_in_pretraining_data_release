import argparse
import os
import sys
import time
import pandas as pd
import anthropic
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_data, get_output_dir, save_predictions, save_metadata
from prompts.narrative_prompt import SYSTEM_PROMPT, TOOL_DEF, ALL_DIMENSIONS, build_user_message

PARQUET_PATH = "annotated_data/agency_annotations.parquet"
DEFAULT_MODEL = "claude-sonnet-4-6"


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
    return anthropic.Anthropic(api_key=api_key)


def submit_batch(client: anthropic.Anthropic, df: pd.DataFrame, model: str) -> str:
    requests = [
        {
            "custom_id": row["safe_instance_id"],
            "params": {
                "model": model,
                "max_tokens": 512,
                # Cache the system prompt — identical across all requests in this batch
                "system": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                # Cache the tool definition — also identical across all requests
                "tools": [{**TOOL_DEF, "cache_control": {"type": "ephemeral"}}],
                "tool_choice": {"type": "tool", "name": "annotate_narrative"},
                "messages": [{"role": "user", "content": build_user_message(row["sampled_text"])}],
            },
        }
        for _, row in df.iterrows()
    ]
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def poll_until_complete(
    client: anthropic.Anthropic, batch_id: str, poll_interval: int, prefix: str = ""
) -> None:
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
            print(f"{prefix}Network hiccup ({type(e).__name__}), retrying in {poll_interval}s...")
            time.sleep(poll_interval)
            continue
        c = batch.request_counts
        print(
            f"{prefix}[{batch.processing_status}] "
            f"processing={c.processing}  succeeded={c.succeeded}  "
            f"errored={c.errored}  expired={c.expired}"
        )
        if batch.processing_status == "ended":
            return
        time.sleep(poll_interval)


def collect_results(
    client: anthropic.Anthropic, batch_id: str, id_to_text: dict, prefix: str = ""
) -> list:
    rows = []
    for result in client.messages.batches.results(batch_id):
        safe_id = result.custom_id
        row = {
            "safe_instance_id": safe_id,
            "sampled_text": id_to_text.get(safe_id, ""),
        }
        if result.result.type == "succeeded":
            scores = {}
            for block in result.result.message.content:
                if block.type == "tool_use":
                    scores = block.input
                    break
            row.update({f"pred_{dim}": scores.get(dim) for dim in ALL_DIMENSIONS})
        else:
            print(f"{prefix}{safe_id}: {result.result.type}")
            row.update({f"pred_{dim}": None for dim in ALL_DIMENSIONS})
        rows.append(row)
    return rows


def run(
    n: int,
    model: str = DEFAULT_MODEL,
    poll_interval: int = 60,
    batch_id: str = None,
    prefix: str = "",
) -> str:
    start_time = datetime.now()
    client = get_client()

    full_df = pd.read_parquet(PARQUET_PATH)
    id_to_text = dict(zip(full_df["safe_instance_id"], full_df["sampled_text"]))

    if batch_id:
        print(f"{prefix}Resuming batch {batch_id} ...")
    else:
        df = load_data(PARQUET_PATH, n=n)
        print(f"{prefix}Submitting batch of {len(df)} instances ({model}) ...")
        batch_id = submit_batch(client, df, model)
        print(f"{prefix}Batch ID: {batch_id}  (resume: --batch-id {batch_id})")

    print(f"{prefix}Polling for completion ...")
    poll_until_complete(client, batch_id, poll_interval, prefix)

    print(f"{prefix}Retrieving results ...")
    rows = collect_results(client, batch_id, id_to_text, prefix)

    end_time = datetime.now()
    duration = round((end_time - start_time).total_seconds(), 1)

    results_df = pd.DataFrame(rows)
    output_dir = get_output_dir(model)
    output_path = save_predictions(results_df, output_dir)
    save_metadata(output_dir, {
        "model": model,
        "n_instances": len(results_df),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "batch_id": batch_id,
    })

    print(f"{prefix}Done in {duration}s → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Annotate narrative instances using the Anthropic Message Batches API."
    )
    parser.add_argument("--n", type=int, default=10, help="Number of instances (default: 10)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Claude model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--poll", type=int, default=60, help="Seconds between status checks (default: 60)")
    parser.add_argument("--batch-id", type=str, default=None, help="Resume an existing batch by ID")
    args = parser.parse_args()

    output_path = run(args.n, args.model, args.poll, batch_id=args.batch_id)

    results_df = pd.read_csv(output_path)
    display_cols = ["safe_instance_id"] + [f"pred_{d}" for d in ALL_DIMENSIONS]
    print("\n" + results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
