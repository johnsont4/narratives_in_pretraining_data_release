"""Event-relation prompts: temporal order and causal relation between two spans.

Four variants, from two independent choices:

* **gold** vs **scale** — the gold set holds one randomly sampled pair per
  passage, the pair the human annotators saw, and asks whether each span is a
  true event trigger at all. The scale-up annotates every neighboring pair and
  takes both spans as given.
* **tool** vs **json** — Claude is given a tool definition and returns
  structured input; the Doubleword-hosted models are asked for raw JSON and
  their output is parsed. The guideline text is identical either way.

Pick one with `variant(split, style)`. The guideline sections below are the
single copy of that text; every variant embeds them.
"""

import json
from dataclasses import dataclass, field
from typing import Callable

TEMPORAL_SECTION = """Which event started first in the narrative timeline?

- **span1_first** — SPAN1 started before SPAN2
- **span2_first** — SPAN2 started before SPAN1
- **simultaneous** — Both events genuinely share a start point and are distinct events
- **same_event** — The two spans refer to the same event from different angles
- **too_hard_to_tell** — The start times could feasibly be in either order, or the events occur in completely unrelated temporal frames

If either span is not an event, use **not_applicable**.

**Inferring order:** Use explicit connectives ("before," "after," "then," "when," "while," "as"), backward-looking cues ("preceded," "followed," "prior to"), and logical necessity. A cause precedes its effect, a question is asked before it is answered, a proposal precedes an agreement. When one event logically presupposes the other, infer the ordering.

**Reporting and retrospective verbs:** Story-world events precede the speech acts or memory acts that describe them, even when the reporting verb appears first in the text. "The coach expressed postgame that losing the player had impacted the offense" -> span2_first. Memory verbs (*recalls, remembers, reflects*) similarly occur after the events they describe.

**Simultaneous vs. sequential:** Use simultaneous only when neither event initiates the other and the text marks coincident starts ("just as X, Y"). When "as" or "while" appears, ask whether both events genuinely begin at the same instant (-> simultaneous) or whether one was already underway when the other began (-> span1_first or span2_first). Process/result pairs and physical chains are sequenced even when compressed: the initiating event comes first.
    Example: "While touring in Germany, he released a record." -> span1_first. Touring was already underway when the release happened. "while" marks an ongoing background activity, not a coincident start. The touring initiated first, the release occurred during it.
    

**Same event:** Use same_event when both spans describe the same occurrence and neither adds new information the other omits. This includes a noun and verb anchoring the same action, two descriptions of the same scope, a quote introduced and then attributed. Continuation verbs (*adding, continuing, noting*) extend a prior speech act —> use span1_first.

**Too hard to tell:** You can think of this label as ambiguous, some event pairs are difficult to order definitively. Remember text position alone is not evidence. 
Use when: 
    (a) two events describe different aspects of the same episode with no logical ordering between them ("he provided information during the meeting. she refilled water during the meeting.")
    (b) the events involve different unrelated actors with no shared context ("he hit the ball. she built a car.")
    (c) the ordering requires world knowledge inference the text does not support.
If you find yourself inferring order from narrative position, world knowledge, or plausible sequence rather than explicit textual evidence, use too_hard_to_tell. Uncertainty is not a failure. too_hard_to_tell is the correct answer for genuinely ambiguous cases.

"""
CAUSAL_SECTION = """
**Story-world vs. discourse-world:** Story-world events (actions, occurrences, mental states) and discourse events (*said, reported, wrote, announced*) operate at different levels and cannot causally relate to each other. A fire does not cause a spokesperson's statement; the statement is an independent communicative act. 
When one span is a reporting verb and the other is a story-world event, use **not_related**. Two discourse events can causally relate to each other.

**direct_cause** — E1 is sufficient to produce E2; given E1, E2 was bound to happen without any intervening decision or action.
- Physical chain: "She dropped her phone. The screen cracked."
- Involuntary reaction: "He read the message. His stomach dropped."

**enables** — E1 creates a necessary precondition for E2: if E1 had not happened, E2 could not have happened in the same way. The dependency must be traceable in the text. Mere co-occurrence is not enough. 
Enablement can run in either direction: "They decided to investigate the corals, which had been glowing for weeks" — the glowing enabled the decision even though it came first. 
Enablement also includes cases where E1 places an agent or object in the specific situation or state that made E2 possible.

**not_related** — No traceable causal dependency. Events that are merely co-present in the narrative without a direct conditional link are not_related.

**Distinguishing tip:** Was E2 bound to happen given E1 alone? -> direct_cause. Did E1 create a clear necessary precondition? -> enables. Otherwise -> not_related."""




def highlight_text(text: str, span1: list, span2: list) -> str:
    """Wrap the two spans in the [[SPAN1: ...]] / [[SPAN2: ...]] markers.

    Applied right-to-left so the first insertion does not shift the second
    span's character offsets.
    """
    spans = [
        (span1[0], span1[1], "SPAN1"),
        (span2[0], span2[1], "SPAN2"),
    ]
    spans.sort(key=lambda s: s[0], reverse=True)
    for start, end, label in spans:
        word = text[start:end]
        text = text[:start] + f"[[{label}: {word}]]" + text[end:]
    return text


# ── The four variants ────────────────────────────────────────────────────────
#
# The guideline sections above are the shared bulk (~4,750 of ~6,200 chars).
# What follows differs per variant in ways that are load-bearing — enum order,
# the closing instruction, even blank lines are part of what the model saw —
# so each template is written out rather than assembled from flags.

#: The "is this span an event?" step, asked on the gold set only.
_EVENT_STEP = """## Step 1: Is each span a true event trigger?

Events are singular, bounded occurrences where something happens at a particular point in time. Event triggers are the smallest units that can be identified as events.

A span qualifies as an event trigger if you can identify **what happened AND who or what it happened to** from the surrounding text.

Do NOT count hypothetical, negated, or future events that are not clearly indicated as having actually occurred in the text. Do NOT count generic or habitual statements that describe what is generally or repeatedly true rather than what happened on a specific occasion. For example: "travelling from New Zealand to Perth takes 4.5 hours" and "he would often help out" describe recurring patterns, not events.

Judge each span independently:
- **span1_is_event**: Is [[SPAN1]] a true event trigger in context?
- **span2_is_event**: Is [[SPAN2]] a true event trigger in context?"""

_INTRO_GOLD = "You are an expert linguistic annotator specializing in event semantics and narrative structure across web text. You will be shown a text passage with two highlighted spans [[SPAN1: ...]] and [[SPAN2: ...]] and asked to make four judgments."
_INTRO_SCALE = """You are an expert linguistic annotator specializing in event semantics and narrative structure across web text. You will be shown a text passage with two highlighted event spans — [[SPAN1: ...]] and [[SPAN2: ...]] — and asked to make two judgments about how they relate.

Both highlighted spans have been pre-confirmed as events. Focus only on the relations between them."""

GOLD_DIMENSIONS = ["span1_is_event", "span2_is_event", "temporal_order", "causality_rating"]
SCALE_DIMENSIONS = ["temporal_order", "causality_rating"]

# Order matters: these lists reach the model as the tool schema's `enum` and as
# the JSON instruction's option list. Sorting them would silently change the
# prompt, so they are written in the order the annotators used.
GOLD_TEMPORAL = ["span1_first", "span2_first", "simultaneous", "same_event",
                 "too_hard_to_tell", "not_applicable"]
GOLD_CAUSAL = ["direct_cause", "enables", "not_related", "not_applicable"]
SCALE_TEMPORAL = ["span1_first", "span2_first", "simultaneous", "same_event", "too_hard_to_tell"]
SCALE_CAUSAL = ["direct_cause", "enables", "not_related"]

_GOLD_JSON_SPEC = (
    'Return a JSON object with exactly these keys:\n'
    '- "span1_is_event": true or false\n'
    '- "span2_is_event": true or false\n'
    '- "temporal_order": one of "span1_first", "span2_first", "simultaneous", '
    '"same_event", "too_hard_to_tell", "not_applicable"\n'
    '- "causality_rating": one of "direct_cause", "enables", "not_related", "not_applicable"\n'
    'No extra keys or explanation. JSON only.'
)
_SCALE_JSON_SPEC = (
    'Return a JSON object with exactly these keys:\n'
    '- "temporal_order": one of "span1_first", "span2_first", "simultaneous", '
    '"same_event", "too_hard_to_tell"\n'
    '- "causality_rating": one of "direct_cause", "enables", "not_related"\n'
    'No extra keys or explanation. JSON only.'
)

_GOLD_TOOL_DEF = {
    "name": "annotate_event_relation",
    "description": "Return event validity, temporal order, and causal relation for the two highlighted spans.",
    "input_schema": {
        "type": "object",
        "properties": {
            "span1_is_event": {
                "type": "boolean",
                "description": "Is [[SPAN1]] a true event trigger — a singular, bounded occurrence on a specific occasion (not generic, habitual, hypothetical, negated, or future)?",
            },
            "span2_is_event": {
                "type": "boolean",
                "description": "Is [[SPAN2]] a true event trigger — a singular, bounded occurrence on a specific occasion (not generic, habitual, hypothetical, negated, or future)?",
            },
            "temporal_order": {
                "type": "string",
                "enum": GOLD_TEMPORAL,
                "description": "Which event came first in the narrative timeline? Use not_applicable if either span is not an event.",
            },
            "causality_rating": {
                "type": "string",
                "enum": GOLD_CAUSAL,
                "description": "Causal relation between the two events. Use not_applicable if either span is not an event.",
            },
        },
        "required": GOLD_DIMENSIONS,
    },
}

_SCALE_TOOL_DEF = {
    "name": "annotate_event_relation",
    "description": "Return temporal order and causal relation for the two highlighted event spans.",
    "input_schema": {
        "type": "object",
        "properties": {
            "temporal_order": {
                "type": "string",
                "enum": SCALE_TEMPORAL,
                "description": "Which event came first in the narrative timeline?",
            },
            "causality_rating": {
                "type": "string",
                "enum": SCALE_CAUSAL,
                "description": "Causal relation between the two events (in either direction). Must be exactly one of: direct_cause, enables, not_related. Do not use same_event or any other value here.",
            },
        },
        "required": SCALE_DIMENSIONS,
    },
}

_GOLD_TOOL_SYSTEM = f"""{_INTRO_GOLD}

---

{_EVENT_STEP}

---

## Step 2: Temporal order (only if BOTH spans are events)

{TEMPORAL_SECTION}

If either span is not an event, use **not_applicable**.

---

## Step 3: Causal relation (only if BOTH spans are events)

{CAUSAL_SECTION}

**not_applicable** — Either span is not an event.

Use the annotate_event_relation tool to return all four judgments."""

_GOLD_JSON_SYSTEM = f"""{_INTRO_GOLD}

---

{_EVENT_STEP}

---

## Step 2: Temporal order (only if BOTH spans are events)

{TEMPORAL_SECTION}
If either span is not an event, use **not_applicable**.

---

## Step 3: Causal relation (only if BOTH spans are events)

{CAUSAL_SECTION}

**not_applicable** — Either span is not an event.

---

{_GOLD_JSON_SPEC}"""

_SCALE_TOOL_SYSTEM = f"""{_INTRO_SCALE}

---

## Step 1: Temporal order

{TEMPORAL_SECTION}

---

## Step 2: Causal relation

{CAUSAL_SECTION}

Only use the values listed above: **direct_cause**, **enables**, or **not_related**. Do not use any other value for causality_rating.

Use the annotate_event_relation tool to return your two judgments."""

_SCALE_JSON_SYSTEM = f"""{_INTRO_SCALE}

---

## Step 1: Temporal order

{TEMPORAL_SECTION}

---

## Step 2: Causal relation

{CAUSAL_SECTION}

Only use the values listed above: **direct_cause**, **enables**, or **not_related**. Do not use any other value for causality_rating.

---

{_SCALE_JSON_SPEC}"""


@dataclass(frozen=True)
class Variant:
    """One prompt configuration: system text, dimensions, and validation."""

    name: str
    system: str
    dimensions: list[str]
    valid_temporal: list[str]
    valid_causal: list[str]
    tool_def: dict | None
    json_spec: str

    @property
    def judges_events(self) -> bool:
        """Whether the variant asks if each span is a true event trigger."""
        return "span1_is_event" in self.dimensions

    def build_user_message(self, sampled_text: str, span1, span2) -> str:
        """The user turn. `span1`/`span2` accept a JSON string or a parsed list."""
        if isinstance(span1, str):
            span1 = json.loads(span1)
        if isinstance(span2, str):
            span2 = json.loads(span2)
        highlighted = highlight_text(sampled_text, span1, span2)
        if self.tool_def:
            n = "four" if self.judges_events else "two"
            closing = f"Use the annotate_event_relation tool to return your {n} judgments."
        else:
            closing = self.json_spec
        label = "Two spans" if self.judges_events else "Two event spans"
        return (
            f"Please annotate the following text. {label} are highlighted for evaluation.\n\n"
            f"TEXT:\n{highlighted}\n\n"
            f'SPAN 1: "{span1[2]}"\n'
            f'SPAN 2: "{span2[2]}"\n\n'
            f"{closing}"
        )

    def validate(self, scores: dict, item_id: str) -> dict:
        """Coerce a response to this variant's label sets, nulling anything else.

        A null means the model returned something outside the schema, which
        downstream treats as a missing label rather than a wrong one.
        """
        out = dict(scores)

        if self.judges_events:
            for key in ("span1_is_event", "span2_is_event"):
                v = out.get(key)
                if isinstance(v, str):
                    out[key] = v.lower() == "true"
                elif not isinstance(v, bool):
                    print(f"  WARNING: unexpected {key}={v!r} for {item_id} — set to None")
                    out[key] = None

        for key, valid in (("temporal_order", self.valid_temporal),
                           ("causality_rating", self.valid_causal)):
            if out.get(key) not in valid:
                print(f"  WARNING: invalid {key}={out.get(key)!r} for {item_id} — set to None")
                out[key] = None

        return {dim: out.get(dim) for dim in self.dimensions}


VARIANTS = {
    ("gold", "tool"): Variant("gold/tool", _GOLD_TOOL_SYSTEM, GOLD_DIMENSIONS,
                              GOLD_TEMPORAL, GOLD_CAUSAL, _GOLD_TOOL_DEF, _GOLD_JSON_SPEC),
    ("gold", "json"): Variant("gold/json", _GOLD_JSON_SYSTEM, GOLD_DIMENSIONS,
                              GOLD_TEMPORAL, GOLD_CAUSAL, None, _GOLD_JSON_SPEC),
    ("scale", "tool"): Variant("scale/tool", _SCALE_TOOL_SYSTEM, SCALE_DIMENSIONS,
                               SCALE_TEMPORAL, SCALE_CAUSAL, _SCALE_TOOL_DEF, _SCALE_JSON_SPEC),
    ("scale", "json"): Variant("scale/json", _SCALE_JSON_SYSTEM, SCALE_DIMENSIONS,
                               SCALE_TEMPORAL, SCALE_CAUSAL, None, _SCALE_JSON_SPEC),
}


def variant(split: str, style: str) -> Variant:
    """The prompt for one (split, style) pair — e.g. ('scale', 'json')."""
    try:
        return VARIANTS[(split, style)]
    except KeyError:
        raise ValueError(
            f"No event-relation prompt for split={split!r} style={style!r}; "
            f"expected one of {sorted(VARIANTS)}"
        ) from None
