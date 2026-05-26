import argparse
import json
import os
import sys
import time
import pandas as pd
from openai import OpenAI
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_data, get_output_dir, save_predictions, save_metadata
from prompts.narrative_prompt import SYSTEM_PROMPT, ALL_DIMENSIONS, build_user_message

PARQUET_PATH = "annotated_data/agency_annotations.parquet"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

_DIMENSION_LIST = ", ".join(ALL_DIMENSIONS)


def get_client() -> OpenAI:
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    if not api_key:
        raise EnvironmentError("DOUBLEWORD_API_KEY environment variable not set.")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.doubleword.ai/v1",
    )


def annotate_instance(client: OpenAI, sampled_text: str, model: str, poll_interval: int) -> dict:
    user_message = (
        build_user_message(sampled_text)
        + f"\n\nReturn your scores as a JSON object with exactly these keys: {_DIMENSION_LIST}."
        + " Each value must be an integer from 1 to 5. No extra keys or explanation."
    )
    try:
        resp = client.responses.create(
            model=model,
            input=f"{SYSTEM_PROMPT}\n\n---\n\n{user_message}",
            service_tier="flex",
            background=True,
        )
        while resp.status in {"queued", "in_progress"}:
            print(f"  Status: {resp.status}...")
            time.sleep(poll_interval)
            resp = client.responses.retrieve(resp.id)

        if resp.status != "completed":
            error_detail = getattr(resp, "error", None) or getattr(resp, "last_error", None)
            print(f"  Request ended with status: {resp.status}" + (f" — {error_detail}" if error_detail else ""))
            return {dim: None for dim in ALL_DIMENSIONS}

        scores = json.loads(resp.output_text)
        return {
            dim: max(1, min(5, int(scores[dim]))) if dim in scores else None
            for dim in ALL_DIMENSIONS
        }
    except Exception as e:
        print(f"  Failed: {e}")
        return {dim: None for dim in ALL_DIMENSIONS}


def run(
    n: int,
    model: str = DEFAULT_MODEL,
    poll_interval: int = 2,
    prefix: str = "",
) -> str:
    start_time = datetime.now()
    print(f"{prefix}Loading {n} instances ({model}) ...")
    client = get_client()
    df = load_data(PARQUET_PATH, n=n)

    results = []
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{prefix}[{idx}/{len(df)}] {row['safe_instance_id']}")
        scores = annotate_instance(client, row["sampled_text"], model, poll_interval)
        result = {
            "safe_instance_id": row["safe_instance_id"],
            "sampled_text": row["sampled_text"],
        }
        result.update({f"pred_{dim}": scores.get(dim) for dim in ALL_DIMENSIONS})
        results.append(result)

    end_time = datetime.now()
    duration = round((end_time - start_time).total_seconds(), 1)

    results_df = pd.DataFrame(results)
    output_dir = get_output_dir(model)
    output_path = save_predictions(results_df, output_dir)
    save_metadata(output_dir, {
        "model": model,
        "n_instances": len(results_df),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
    })

    print(f"{prefix}Done in {duration}s → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Annotate narrative instances using the Doubleword AI API.")
    parser.add_argument("--n", type=int, default=10, help="Number of instances (default: 10)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--poll", type=int, default=2, help="Polling interval in seconds (default: 2)")
    args = parser.parse_args()

    output_path = run(args.n, args.model, args.poll)

    results_df = pd.read_csv(output_path)
    display_cols = ["safe_instance_id"] + [f"pred_{d}" for d in ALL_DIMENSIONS]
    print("\n" + results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
