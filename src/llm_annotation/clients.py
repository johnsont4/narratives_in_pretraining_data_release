"""Model registry and the two ways a model gets called.

Every annotator in this repository is one of two shapes:

* **batch** — Anthropic's Message Batches API. One request per item, submitted
  together, polled to completion, then read back. The system prompt and tool
  definition are identical across a batch, so both are cached. A run can be
  resumed from its batch id.
* **threaded** — the Doubleword-hosted open models, through an OpenAI-compatible
  client. Requests go out concurrently and each is polled to completion; results
  are checkpointed as they land, so a run resumes from its own output file.

Which one a model uses is a property of the model, not of the task, so it lives
in `MODELS` rather than in the annotation scripts.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class Model:
    """One annotator: how to reach it, and how it returns structured output.

    `style` selects the prompt variant — "tool" models are given a tool
    definition and return structured input; "json" models are asked for raw
    JSON in the prompt and their text output is parsed.
    """

    key: str
    provider: str      # "anthropic" | "doubleword"
    model_id: str
    style: str         # "tool" | "json"

    @property
    def runner(self) -> str:
        return "batch" if self.provider == "anthropic" else "threaded"


MODELS = {
    "claude": Model("claude", "anthropic", "claude-sonnet-4-6", "tool"),
    "gemma": Model("gemma", "doubleword", "google/gemma-4-31B-it", "json"),
    "qwen": Model("qwen", "doubleword", "Qwen/Qwen3.6-35B-A3B-FP8", "json"),
}

#: Environment variable holding each provider's key.
API_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "doubleword": "DOUBLEWORD_API_KEY"}

DOUBLEWORD_BASE_URL = "https://api.doubleword.ai/v1"


def resolve(key: str, model_id: str | None = None) -> Model:
    """Look up a model by short name, optionally overriding the model id."""
    try:
        model = MODELS[key]
    except KeyError:
        raise SystemExit(
            f"Unknown model {key!r}. Choose one of: {', '.join(MODELS)}"
        ) from None
    return Model(model.key, model.provider, model_id or model.model_id, model.style)


def get_client(model: Model):
    """An SDK client for the model's provider, from the environment key."""
    env = API_KEY_ENV[model.provider]
    api_key = os.environ.get(env)
    if not api_key:
        raise EnvironmentError(f"{env} environment variable not set.")

    if model.provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=DOUBLEWORD_BASE_URL)


# ── response parsing ─────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model's raw text output.

    Tries the whole string first, then falls back to the first {...} block —
    open models fence their output or add a sentence around it often enough
    that a strict parse alone loses usable responses.
    """
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in response: {text[:200]!r}")


def _tool_input(content) -> dict:
    """The first tool_use block's input, or {} if the model returned none."""
    for block in content:
        if block.type == "tool_use":
            return dict(block.input)
    return {}


# ── batch runner (Anthropic) ─────────────────────────────────────────────────

def submit_batch(client, items, model: Model, system: str, tool_def: dict,
                 max_tokens: int) -> str:
    """Submit one request per item and return the batch id.

    The system prompt and tool definition are identical across every request,
    so both carry a cache_control breakpoint.
    """
    requests = [
        {
            "custom_id": item.key,
            "params": {
                "model": model.model_id,
                "max_tokens": max_tokens,
                "system": [{"type": "text", "text": system,
                            "cache_control": {"type": "ephemeral"}}],
                "tools": [{**tool_def, "cache_control": {"type": "ephemeral"}}],
                "tool_choice": {"type": "tool", "name": tool_def["name"]},
                "messages": [{"role": "user", "content": item.message}],
            },
        }
        for item in items
    ]
    return client.messages.batches.create(requests=requests).id


def poll_batch(client, batch_id: str, poll_interval: int) -> None:
    """Block until the batch ends, surviving transient network failures."""
    import anthropic

    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
            print(f"Network hiccup ({type(e).__name__}), retrying in {poll_interval}s...")
            time.sleep(poll_interval)
            continue
        c = batch.request_counts
        print(f"[{batch.processing_status}] processing={c.processing}  "
              f"succeeded={c.succeeded}  errored={c.errored}  expired={c.expired}")
        if batch.processing_status == "ended":
            return
        time.sleep(poll_interval)


def collect_batch(client, batch_id: str) -> dict[str, dict | None]:
    """Map each custom_id to its tool input, or None if the request failed."""
    out: dict[str, dict | None] = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            out[result.custom_id] = _tool_input(result.result.message.content)
        else:
            print(f"  {result.custom_id}: {result.result.type}")
            out[result.custom_id] = None
    return out


def run_realtime(client, items, model: Model, system: str, tool_def: dict,
                 max_tokens: int) -> dict[str, dict | None]:
    """Call the model item by item instead of batching. Useful for small runs."""
    out: dict[str, dict | None] = {}
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item.key} ...", end=" ", flush=True)
        try:
            response = client.messages.create(
                model=model.model_id,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                tools=[{**tool_def, "cache_control": {"type": "ephemeral"}}],
                tool_choice={"type": "tool", "name": tool_def["name"]},
                messages=[{"role": "user", "content": item.message}],
            )
            out[item.key] = _tool_input(response.content)
            print("ok")
        except Exception as e:
            out[item.key] = None
            print(f"error: {e}")
    return out


# ── threaded runner (Doubleword) ─────────────────────────────────────────────

def _annotate_one(client, item, model: Model, system: str, poll_interval: int) -> dict | None:
    """One background request, polled to completion and parsed."""
    output_text = None
    try:
        resp = client.responses.create(
            model=model.model_id,
            input=f"{system}\n\n---\n\n{item.message}",
            service_tier="flex",
            background=True,
        )
        while resp.status in {"queued", "in_progress"}:
            time.sleep(poll_interval)
            resp = client.responses.retrieve(resp.id)

        if resp.status != "completed":
            detail = getattr(resp, "error", None) or getattr(resp, "last_error", None)
            print(f"  [{item.key}] ended with status: {resp.status}"
                  + (f" — {detail}" if detail else ""))
            return None

        output_text = resp.output_text
        return extract_json(output_text)
    except Exception as e:
        print(f"  [{item.key}] Failed: {e}")
        if output_text is not None:
            print(f"  [{item.key}] Raw output: {output_text[:300]!r}")
        return None


def run_threaded(client, items, model: Model, system: str, workers: int,
                 poll_interval: int, on_result: Callable[[str, dict | None, int], None]) -> None:
    """Annotate `items` concurrently, handing each result to `on_result`.

    Results arrive out of order, so `on_result` receives the item key with the
    parsed response and the running completion count. Checkpointing is the
    caller's business — it is the only side that knows the output columns.
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_annotate_one, client, item, model, system, poll_interval): item
            for item in items
        }
        for n, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            on_result(item.key, future.result(), n)
