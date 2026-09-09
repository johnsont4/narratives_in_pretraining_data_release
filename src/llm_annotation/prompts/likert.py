"""The nine-dimension agency/setting annotation prompt.

Order matters: it is the order the model outputs, the order the tool
definition declares, and the order the `pred_*` columns appear in. It matches
`narrabert/train_likert.py`, which re-declares the same lists
for its own gold-column mapping.
"""

AGENCY_DIMENSIONS = ["focalization", "emotion", "cognition", "change_of_state", "conflict"]
SETTING_DIMENSIONS = ["concreteness", "temporal_grounding", "spatial_grounding", "sensory"]
ALL_DIMENSIONS = AGENCY_DIMENSIONS + SETTING_DIMENSIONS

SYSTEM_PROMPT = """You are an expert at identifying narrative qualities in web text. Your task is to read a passage and rate it on nine dimensions across two categories (Agency and Setting) each on a scale from 1 (not at all) to 5 (extremely).

---

## AGENCY DIMENSIONS

These five dimensions capture how characters are represented as agents in the text. Importantly, these dimensions rate the presence and centrality of the feature in the text as written, not what can be inferred about the content.

### 1. Focalization
How central is a specific character's/narrator's perspective to the text?

A higher score reflects text in which the details and events are primarily presented through a specific character's perspective: their perceptions, thoughts, and feelings shape how the story is told. A lower score reflects text that reports details and events from the outside, describing what is observable without granting access to any character's internal experience.

Focalization is about whether the reader experiences events through a specific perceiving consciousness's inner life, regardless of the formal mechanism. That can be achieved through:
- Direct quotation (if the content renders inner experience)
- Free indirect discourse ("she couldn't shake the feeling...")
- First person narration with genuine interiority
- Close third person narration
- When "you" appears in instructional or legal text, it should score low on focalization. However a specific second-person perspective that immerses the reader can score high.

Example (score 5): "she scanned the faces in the crowd, certain that someone was watching her"
Example (score 3): "He wasn't sure the meeting had gone well. He gathered his things and headed for the door."
Example (score 1): "she walked through the crowd."

### 2. Emotion
How central are a character's emotional states to the text?

A higher score reflects text in which a character's emotional experience is a prominent feature. A lower score reflects text with little or no reference to how a character feels.

Example (score 5): "Completely paralized with fear, he was all of a sudden flooded with relief when he heard her voice"
Example (score 3): "She was nervous about the presentation. She went into the interview room."
Example (score 1): "he answered the phone."

### 3. Cognition
How central are a character's thoughts, reasoning, or motivations to the text?

A higher score reflects text in which a character's thoughts, beliefs, reasoning, goals, or desires are a prominent feature. A lower score reflects text with little or no reference to what a character thinks, intends, or wants.

Example (score 5): "she kept turning the problem over in her mind, convinced there was something she had missed. Was it because they were so quiet in the meeting? Or their body language? She couldn't tell."
Example (score 3): "She read the email twice. It wasn't entirely clear what they were asking for, but she thought she understood the gist. She drafted a reply."
Example (score 1): "she sat at her desk"

### 4. Change of State
How central is a change in a character's condition or state to the text?

A higher score reflects text in which a change in a character's condition/state is a prominent or organizing feature. A lower score reflects text where characters remain essentially unchanged. 
Note: the change does not need to be completed within the passage. An in-progress or partially implied change still counts. If a passage describes a world event (a company going bankrupt, an earthquake, a death) that necessarily entails a change in a character's condition, that counts toward the score even if the character's internal response is not elaborated.
Change of state should be rated on presence and centrality of the changes themselves, independent of how vividly or dramatically they're rendered. 

Example (score 5): "by the time she reached the door, something in her had shifted. She wasn't the same person who had walked in" or "He started a new job and decided to make a significant change in his diet. His whole life changed when he moved."
Example (score 3): "By the end of the conversation he felt somewhat better about the situation. It wasn't resolved, but it seemed more manageable than it had before."
Example (score 1): "she sat at her desk."

### 5. Conflict
How central is conflict involving characters to the text?

A higher score reflects text in which conflict is a dominant or organizing feature. A lower score reflects text in which conflict is absent or only incidentally present. Conflict can take many forms: tension between characters, internal psychological struggle, or opposition to institutions, environments, inanimate objects like technology, social forces, or physical events.

Example (score 5): "they had been arguing for hours, neither willing to give ground. The people around them started to stare. They were so caught up in the fight that they didn't even notice."
Example (score 3): "The radio was playing Tame Impala as they drove to the cabin. They disagreed about the route. They took the highway and didn't talk much for the first hour. They arrived at the cabin."
Example (score 1): "they sat across from each other at the table."

---

## SETTING DIMENSIONS

These four dimensions capture how the text constructs a sense of place, time, and physical presence.

### 6. Concreteness
How concrete is the language of the text?

How much does the text refer to things that can be directly experienced through the senses or through action, as opposed to things whose meaning depends on other words?
A higher score reflects text in which most content could be explained by pointing, demonstrating, or showing (objects, physical states, bodily actions, perceptual qualities). A lower score reflects text in which most content can only be explained using other words (categories, principles, relationships, institutions, ideas).

The key distinction is not just whether physical things are present, but how much the text renders them. Named objects that receive no perceptual description are more concrete than purely abstract prose, but less concrete than objects whose material qualities are evoked. Rendering degree is what separates the middle of the scale from the top.

When a passage enumerates many specific, tangible, pointable things (materials, tools, substances, components) the cumulative effect creates strong concrete presence. 

Example (score 5): "the chipped blue mug sat on the edge of the sink, handle cracked from years of use" (specific object, perceptual detail with high rendering) / "She polished her artwork which allowed it to be shiny and for viewers to see the view of sheep better. She used a horse hair brush to shine it, and once she was done she sat it next to her oil paint set." (Lots of concrete things renders the scene at a very high level)
Example (score 4): "the blue mug sat on the old creaky table" (physical object with some rendering) / "cross stitches on the seam lines and the V neck wasn't finished, the sash/belt still unattached" (physical components rendering a coherent object)
Example (score 3): "She had been commuting to the same office for six years. The building was on a corner downtown, glass and steel, indistinguishable from the ones on either side of it." (physical referents present, minimal rendering)
Example (score 2): "the potentiometers controlled the speed of the motors" (physical objects named but not rendered at all)
Example (score 1): "people often form attachments to everyday objects" (no physical referents; meaning depends entirely on understanding abstract terms)

Note: actions and processes can raise the score when they are attached to rendered objects. "the chipped blue mug wavered on the edge of the table, water splashing over its rim" scores higher than "the chipped blue mug sat on the table" because the action adds perceptual detail. But actions without rendered objects do not independently contribute to concreteness.

### 7. Temporal Grounding
How strongly does the text create a sense of being anchored in a particular time?

A higher score reflects text that creates a vivid sense of temporal location: the reader feels situated in a particular moment, era, or duration. A lower score reflects text that feels as if events could be taking place at any time.
Consider two types of temporal grounding, both of which contribute to the score:
- Historical grounding locates the reader in a specific, unrepeatable moment -- a year, an era, a named event, or a period of someone's life. This type of grounding is efficient: a single year or cultural reference can immediately anchor the reader. Historical grounding can reach the upper end of the scale even with sparse language.
- Cyclical grounding locates the reader within a recurring temporal structure -- a season, a time of day, a day of the week, a time of year. This type of grounding evokes a recognizable temporal texture but not a unique moment. Cyclical grounding alone typically reaches a 3, or a 4 in rare cases where the atmospheric rendering is rich and sustained.

Example (score 5): "the streets were still empty, that particular quiet of early Sunday morning before the city remembered itself. It was the summer of 2015 at 9:02am, before the world changed forever." (historical and cyclical grounding combined) / "On Sunday, June 10, 2004, John purchased his first car." (specific historical grounding alone)
Example (score 4): "I turned off mdma sometime around 2016 -- peak festival era, everyone was still talking about Molly like it was new. Tried it again last month." (historical grounding with cultural texture that evokes a recognizable moment)
Example (score 3): "This unique summer school gives teenagers the opportunity to work with tutors and musicians." (cyclical grounding via season)
Example (score 2): "I turned off mdma for a good 7 years as the comedown became worse than the high." (temporal markers present but only sequencing events, not locating the reader in time)
Example (score 1): "he walked outside." (no temporal content)

### 8. Spatial Grounding
How strongly does the text create a sense of being anchored in a particular place?

A higher score reflects text that creates a vivid sense of spatial location: the reader feels situated in a particular place, whether through named locations, environmental description, or atmospheric rendering. A lower score reflects text that feels as if events could be taking place anywhere.

Consider two types of spatial grounding, both of which contribute to the score:
- Geographic grounding locates the reader on the globe -- a country, city, named landmark, or recognized region. This type is efficient: a single place name can immediately anchor the reader. Named landmarks of global or national recognition (the Eiffel Tower, Grand Central Station) count as geographic grounding even though they are also physical spaces. Can reach a 5 without proximate rendering.
- Proximate grounding locates the reader in an immediate physical environment -- a room, a building, a street, a landscape. This type requires more rendering to score high. Vividly rendered proximate grounding without geographic grounding caps at a 4.

Note: proximate grounding overlaps with concreteness and sensory detail -- the rendered physical details that make you feel present in a space are doing similar work across all three dimensions.

Example (score 5): "In the narrow streets of Naples, the apartment's tiled floors kept cool even in August heat, the noise of the market rising from below." (geographic and proximate grounding combined) / "On 6th Street in Austin, TX" (sufficiently specific geographic grounding alone)
Example (score 4): "The bathroom tiles were cold, the mirror fogged from the shower." (proximate grounding rendered vividly enough to feel present) / "She went swimming in the Atlantic Ocean on vacation in downtown Lisbon, Portugal." (named geographic locations with specific downtown mention)
Example (score 3): "She worked at an office in London." or "The kitchen was small, cluttered with dishes." (named geographic location without evocation, or proximate location with modest rendering)
Example (score 2): "She sat in the kitchen." or "She lived somewhere in Europe." (bare proximate location with no rendering, or vague geographic reference)
Example (score 1): "She decided to quit her job." (no spatial content)

### 9. Sensory
How central are sensory details to the text?

A higher score reflects text in which one or more senses are central to how the passage is organized, where the sensory experience drives, anchors, or sustains the content. A lower score reflects text where events and ideas are conveyed without meaningful appeal to the senses.

Example (score 5): "the bread was still warm, its crust crackling under her fingers, the whole kitchen thick with the smell of it" (multiple senses foregrounded and sustained, rendering is rich)
Example (score 4): "when I first saw their amazing pieces of art I felt excited and inspired as never before" (sight is genuinely central -- the passage hinges on the act of seeing -- but what is seen is not rendered in detail)
Example (score 3): "The bread was still warm when she pulled it from the oven. She set it on the rack and went to wash her hands." (sensory content present but not dominant; the passage moves on quickly)
Example (score 2): "she made breakfast which consisted of stale bread and eggs. She ate it at the table." (sensory content present but incidental)
Example (score 1): "she made breakfast which consisted of bread and eggs." (no sensory content; events conveyed without appeal to the senses)

Note: what matters is both whether a sense modality is central to the passage, AND whether it is vividly described. A passage can score high on sensory even if the sensory content is not rendered in concrete detail -- though rendering depth can raise the score toward 5.
Note: sensory and concreteness overlap but are distinct. Concreteness asks whether physical particulars are rendered; sensory asks whether a sense modality organizes the passage.
---

## Scoring Scale
1 = Not at all
2 = Slightly
3 = Moderately
4 = Considerably
5 = Extremely

Use the annotate_narrative tool to return all nine scores."""

TOOL_DEF = {
    "name": "annotate_narrative",
    "description": "Return annotation scores for all 5 agency dimensions and all 4 setting dimensions.",
    "input_schema": {
        "type": "object",
        "properties": {
            # Agency
            "focalization": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How central is a specific character's/narrator's perspective? (1=not at all, 5=extremely)",
            },
            "emotion": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How central are a character's emotional states? (1=not at all, 5=extremely)",
            },
            "cognition": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How central are a character's thoughts/reasoning/motivations? (1=not at all, 5=extremely)",
            },
            "change_of_state": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How central is a change in a character's condition or state? (1=not at all, 5=extremely)",
            },
            "conflict": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How central is conflict involving characters? (1=not at all, 5=extremely)",
            },
            # Setting
            "concreteness": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How concrete is the language? (1=not at all, 5=extremely)",
            },
            "temporal_grounding": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How strongly does the text anchor in a specific time? (1=not at all, 5=extremely)",
            },
            "spatial_grounding": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How strongly does the text anchor in a specific place? (1=not at all, 5=extremely)",
            },
            "sensory": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How central are sensory details? (1=not at all, 5=extremely)",
            },
        },
        # Derived from ALL_DIMENSIONS, so adding a dimension cannot leave the
        # tool definition silently demanding the old set.
        "required": list(ALL_DIMENSIONS),
    },
}


def build_user_message(sampled_text: str, style: str = "tool") -> str:
    """The user turn.

    `style="json"` appends the raw-JSON instruction for models called without
    tools. The system prompt is the same either way — including its closing
    reference to the annotate_narrative tool, which the Doubleword-hosted
    models also saw. That is what produced the published labels, so it stays.
    """
    msg = (
        f"Please annotate the following text passage on all nine dimensions (5 agency + 4 setting):\n\n"
        f"TEXT:\n{sampled_text}\n\n"
        f"Rate each dimension from 1 (not at all) to 5 (extremely) using the annotate_narrative tool."
    )
    if style == "json":
        msg += JSON_SPEC
    return msg


#: Appended to the user turn for models called without a tool definition.
JSON_SPEC = (
    f"\n\nReturn your scores as a JSON object with exactly these keys: "
    f"{', '.join(ALL_DIMENSIONS)}."
    " Each value must be an integer from 1 to 5. No extra keys or explanation."
)


def validate(scores: dict, item_id: str = "") -> dict:
    """Clamp each dimension to the 1-5 Likert range, nulling anything unusable."""
    out = {}
    for dim in ALL_DIMENSIONS:
        try:
            out[dim] = max(1, min(5, int(scores[dim])))
        except (KeyError, TypeError, ValueError):
            out[dim] = None
    return out
