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

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root
sys.path.insert(0, str(Path(__file__).parent))          # event_relation/ for event_relation_shared

from event_relation_shared import TEMPORAL_SECTION, CAUSAL_SECTION, _highlight_text
from utils import load_data, get_output_dir, save_predictions, save_metadata

PARQUET_PATH = "annotated_data/event_relation_annotations.parquet"
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_WORKERS = 10

DIMENSIONS = ["span1_is_event", "span2_is_event", "temporal_order", "causality_rating"]

VALID_TEMPORAL = {"span1_first", "span2_first", "simultaneous", "same_event", "too_hard_to_tell", "not_applicable"}
VALID_CAUSAL   = {"direct_cause", "enables", "not_related", "not_applicable"}

_JSON_SPEC = (
    'Return a JSON object with exactly these keys:\n'
    '- "span1_is_event": true or false\n'
    '- "span2_is_event": true or false\n'
    '- "temporal_order": one of "span1_first", "span2_first", "simultaneous", '
    '"same_event", "too_hard_to_tell", "not_applicable"\n'
    '- "causality_rating": one of "direct_cause", "enables", "not_related", "not_applicable"\n'
    'No extra keys or explanation. JSON only.'
)

SYSTEM_PROMPT = f"""You are an expert linguistic annotator specializing in event semantics and narrative structure across web text. You will be shown a text passage with two highlighted spans [[SPAN1: ...]] and [[SPAN2: ...]] and asked to make four judgments.

---

## Step 1: Is each span a true event trigger?

Events are singular, bounded occurrences where something happens at a particular point in time. Event triggers are the smallest units that can be identified as events.

A span qualifies as an event trigger if you can identify **what happened AND who or what it happened to** from the surrounding text.

Do NOT count hypothetical, negated, or future events that are not clearly indicated as having actually occurred in the text. Do NOT count generic or habitual statements that describe what is generally or repeatedly true rather than what happened on a specific occasion. For example: "travelling from New Zealand to Perth takes 4.5 hours" and "he would often help out" describe recurring patterns, not events.

Judge each span independently:
- **span1_is_event**: Is [[SPAN1]] a true event trigger in context?
- **span2_is_event**: Is [[SPAN2]] a true event trigger in context?

---

## Step 2: Temporal order (only if BOTH spans are events)

{TEMPORAL_SECTION}
If either span is not an event, use **not_applicable**.

---

## Step 3: Causal relation (only if BOTH spans are events)

{CAUSAL_SECTION}

**not_applicable** — Either span is not an event.

---

{_JSON_SPEC}"""


def _build_user_message(sampled_text: str, span1: list, span2: list) -> str:
    highlighted = _highlight_text(sampled_text, span1, span2)
    return (
        f"Please annotate the following text. Two spans are highlighted for evaluation.\n\n"
        f"TEXT:\n{highlighted}\n\n"
        f"SPAN 1: \"{span1[2]}\"\n"
        f"SPAN 2: \"{span2[2]}\"\n\n"
        f"{_JSON_SPEC}"
    )


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


def _validate(scores: dict, instance_id: str) -> dict:
    validated = dict(scores)

    for key in ("span1_is_event", "span2_is_event"):
        v = validated.get(key)
        if isinstance(v, str):
            validated[key] = v.lower() == "true"
        elif not isinstance(v, bool):
            print(f"  WARNING: unexpected {key}={v!r} for {instance_id} — set to None")
            validated[key] = None

    if validated.get("temporal_order") not in VALID_TEMPORAL:
        print(f"  WARNING: invalid temporal_order={validated.get('temporal_order')!r} for {instance_id} — set to None")
        validated["temporal_order"] = None

    if validated.get("causality_rating") not in VALID_CAUSAL:
        print(f"  WARNING: invalid causality_rating={validated.get('causality_rating')!r} for {instance_id} — set to None")
        validated["causality_rating"] = None

    return validated


def get_client() -> OpenAI:
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    if not api_key:
        raise EnvironmentError("DOUBLEWORD_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key, base_url="https://api.doubleword.ai/v1")


def annotate_instance(
    client: OpenAI, instance_id: str, sampled_text: str,
    span1: list, span2: list, model: str, poll_interval: int,
) -> dict:
    user_message = _build_user_message(sampled_text, span1, span2)
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
            return {dim: None for dim in DIMENSIONS}

        output_text = resp.output_text
        raw = _extract_json(output_text)
        return _validate(raw, instance_id)

    except Exception as e:
        print(f"  [{instance_id}] Failed: {e}")
        try:
            print(f"  [{instance_id}] Raw output: {output_text[:300]!r}")
        except NameError:
            pass
        return {dim: None for dim in DIMENSIONS}


def run(
    n: int,
    model: str = DEFAULT_MODEL,
    poll_interval: int = 2,
    workers: int = DEFAULT_WORKERS,
    prefix: str = "",
) -> str:
    start_time = datetime.now()
    print(f"{prefix}Loading {n} instances ({model}) | workers={workers} ...")
    client = get_client()
    df = load_data(PARQUET_PATH, n=n)

    future_to_meta = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _, row in df.iterrows():
            sid = row["safe_instance_id"]
            span1 = json.loads(row["assigned_span1"])
            span2 = json.loads(row["assigned_span2"])
            future = executor.submit(
                annotate_instance, client, sid,
                row["sampled_text"], span1, span2, model, poll_interval,
            )
            future_to_meta[future] = {
                "safe_instance_id": sid,
                "sampled_text": row["sampled_text"],
                "assigned_span1": row["assigned_span1"],
                "assigned_span2": row["assigned_span2"],
            }

        results = []
        for future in as_completed(future_to_meta):
            meta = future_to_meta[future]
            scores = future.result()
            result = {
                "safe_instance_id": meta["safe_instance_id"],
                "sampled_text":     meta["sampled_text"],
                "assigned_span1":   meta["assigned_span1"],
                "assigned_span2":   meta["assigned_span2"],
            }
            result.update({f"pred_{dim}": scores.get(dim) for dim in DIMENSIONS})
            results.append(result)
            print(f"{prefix}[{len(results)}/{len(df)}] {meta['safe_instance_id']} done")

    end_time = datetime.now()
    duration = round((end_time - start_time).total_seconds(), 1)

    results_df = pd.DataFrame(results)
    output_dir = get_output_dir(model, output_base="event_relation/outputs")
    output_path = save_predictions(results_df, output_dir)
    save_metadata(output_dir, {
        "model": model,
        "n_instances": len(results_df),
        "workers": workers,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
    })

    print(f"{prefix}Done in {duration}s → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Annotate event relation pairs using Gemma via Doubleword AI.")
    parser.add_argument("--n", type=int, default=10, help="Number of instances (default: 10)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--poll", type=int, default=2, help="Polling interval in seconds (default: 2)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent requests (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    output_path = run(args.n, args.model, args.poll, args.workers)

    results_df = pd.read_csv(output_path)
    display_cols = ["safe_instance_id"] + [f"pred_{d}" for d in DIMENSIONS]
    print("\n" + results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
