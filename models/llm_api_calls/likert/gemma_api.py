import argparse
import json
import os
import re
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import save_metadata
from prompts.narrative_prompt import SYSTEM_PROMPT, ALL_DIMENSIONS, build_user_message

SCALE_PARQUET = "data_to_scale/dolma_final_sample_s100_n1000000_t0.5_deduped_no_s42_subset_n5000.parquet"
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_WORKERS = 20
CHECKPOINT_EVERY = 50

_DIMENSION_LIST = ", ".join(ALL_DIMENSIONS)


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in response: {text[:200]!r}")


def get_client() -> OpenAI:
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    if not api_key:
        raise EnvironmentError("DOUBLEWORD_API_KEY environment variable not set.")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.doubleword.ai/v1",
    )


def load_scale_data(n: Optional[int] = None) -> pd.DataFrame:
    try:
        df = pd.read_parquet(SCALE_PARQUET)
    except Exception:
        df = pd.read_parquet(SCALE_PARQUET, engine="fastparquet")
    if n is not None:
        df = df.head(n)
    return df.reset_index(drop=True)


def annotate_instance(
    client: OpenAI, instance_id: str, sampled_text: str, model: str, poll_interval: int
) -> dict:
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
            time.sleep(poll_interval)
            resp = client.responses.retrieve(resp.id)

        if resp.status != "completed":
            error_detail = getattr(resp, "error", None) or getattr(resp, "last_error", None)
            print(f"  [{instance_id}] ended with status: {resp.status}" + (f" — {error_detail}" if error_detail else ""))
            return {dim: None for dim in ALL_DIMENSIONS}

        output_text = resp.output_text
        scores = _extract_json(output_text)
        return {
            dim: max(1, min(5, int(scores[dim]))) if dim in scores else None
            for dim in ALL_DIMENSIONS
        }
    except Exception as e:
        print(f"  [{instance_id}] Failed: {e}")
        try:
            print(f"  [{instance_id}] Raw output: {output_text[:300]!r}")
        except NameError:
            pass
        return {dim: None for dim in ALL_DIMENSIONS}


def run(
    n: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    poll_interval: int = 2,
    workers: int = DEFAULT_WORKERS,
    output_dir: Optional[Path] = None,
) -> str:
    start_time = datetime.now()

    if output_dir is None:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        sanitized = model.replace("/", "_").replace(":", "_").replace(" ", "_")
        output_dir = Path("outputs/scale") / sanitized / timestamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Workers: {workers}")

    predictions_path = output_dir / "predictions.csv"

    # Resume: load already-completed instances
    completed_ids: set = set()
    results = []
    if predictions_path.exists():
        existing_df = pd.read_csv(predictions_path)
        completed_ids = set(existing_df["id"].astype(str))
        results = existing_df.to_dict("records")
        print(f"Resuming: {len(completed_ids)} instances already completed.")

    print(f"Loading scale data ...")
    df = load_scale_data(n=n)
    to_do = df[~df["id"].astype(str).isin(completed_ids)].reset_index(drop=True)
    total = len(df)
    print(f"Total: {total} | Completed: {len(completed_ids)} | Remaining: {len(to_do)}")

    if len(to_do) == 0:
        print("All instances already completed.")
        return str(predictions_path)

    client = get_client()

    future_to_meta = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _, row in to_do.iterrows():
            instance_id = str(row["id"])
            future = executor.submit(
                annotate_instance, client, instance_id, row["sampled_text"], model, poll_interval
            )
            future_to_meta[future] = (instance_id, row["sampled_text"])

        completed_count = 0
        for future in as_completed(future_to_meta):
            instance_id, sampled_text = future_to_meta[future]
            scores = future.result()
            result = {"id": instance_id, "sampled_text": sampled_text}
            result.update({f"pred_{dim}": scores.get(dim) for dim in ALL_DIMENSIONS})
            results.append(result)
            completed_count += 1
            print(f"[{len(results)}/{total}] {instance_id} done")

            if completed_count % CHECKPOINT_EVERY == 0:
                pd.DataFrame(results).to_csv(predictions_path, index=False)
                elapsed = round((datetime.now() - start_time).total_seconds(), 1)
                print(f"  Checkpoint: {len(results)} total saved ({elapsed}s elapsed)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(predictions_path, index=False)

    end_time = datetime.now()
    duration = round((end_time - start_time).total_seconds(), 1)

    save_metadata(output_dir, {
        "model": model,
        "n_instances": len(results_df),
        "completed_in_this_run": len(to_do),
        "workers": workers,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
    })

    print(f"Done in {duration}s → {predictions_path}")
    return str(predictions_path)


def main():
    parser = argparse.ArgumentParser(description="Scale-up annotation using Gemma via Doubleword AI.")
    parser.add_argument("--n", type=int, default=None, help="Number of instances (default: all 5000)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--poll", type=int, default=2, help="Polling interval in seconds (default: 2)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of concurrent requests (default: {DEFAULT_WORKERS})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory; pass the same path from a prior run to resume it")
    args = parser.parse_args()

    run(
        n=args.n,
        model=args.model,
        poll_interval=args.poll,
        workers=args.workers,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
