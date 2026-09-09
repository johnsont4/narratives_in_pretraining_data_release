# Narratives in Pretraining Data

> **Characterizing Narrative Content in Web-scale LLM Pretraining Data**
> *Teagan Johnson, Elliott Ash, Andrew Piper, Maria Antoniak* · [arXiv:2606.19468](https://arxiv.org/abs/2606.19468)

We annotate web-scale pretraining text for narrative structure along three axes: **agency and setting**
(nine dimensions, 1–5 Likert), and **event relations** (temporal sequencing and causality between event
span pairs, detected from the LitBank event span detector). Human gold labels train LLM annotators which in turn produce pseudo-labels for two
RoBERTa-based classifiers (**NarraBERT**) applied to ~13M passages of Dolma. Since the max token length for both models is 200, we chunk full web documents and pool narrative feature scores.

**All model weights and data live on Hugging Face:
[Narratives in LLM Pretraining Data](https://huggingface.co/collections/teagrjohnson/narratives-in-llm-pretraining-data).**
This repository holds the code that produced them, plus tutorials for using the models.

To explore the full NarraDolma dataset, check out our [NarraDolma explorer](http://antoniak-lab.colorado.edu/tejo9855/corpus_filter/)!

---

## Models

| Model | Task |
|---|---|
| [`CLS-Lab/narrative-likert-roberta`](https://huggingface.co/CLS-Lab/narrative-likert-roberta) | Multi-task regression over 9 agency/setting dimensions (1–5) |
| [`CLS-Lab/narrative-event-relation-roberta`](https://huggingface.co/CLS-Lab/narrative-event-relation-roberta) | Binary classification of temporal sequencing and causality for event span pairs |

Both are fine-tuned from `roberta-base`. Each repo ships `model.pt` (a raw `state_dict`, so
`AutoModel.from_pretrained` will not work, see the tutorials for instructions), a `tokenizer/`, and a `config.json` with the
full hyperparameter and training record.

## Data

| Dataset | Contents |
|---|---|
| [`CLS-Lab/narrative-gold-annotations`](https://huggingface.co/datasets/CLS-Lab/narrative-gold-annotations) | Adjudicated human annotations: 400 agency, 400 setting, 440 event-relation passages, with secondary annotators for agreement |
| [`CLS-Lab/narrative-llm-annotations`](https://huggingface.co/datasets/CLS-Lab/narrative-llm-annotations) | LLM pseudo-labels used to train NarraBERT: 25k Likert-scored passages, 61k event pairs |
| [`CLS-Lab/narradolma`](https://huggingface.co/datasets/CLS-Lab/narradolma) | NarraBERT labels over Dolma at scale: 13.3M scored passages, 12.2M event pairs |

Column-level documentation lives on each dataset card.

Also, check out the full [NarraDolma explorer](http://antoniak-lab.colorado.edu/tejo9855/corpus_filter/)!

---

## Tutorials

Two notebooks in `tutorials/` show how to load a model, score your own text, and reproduce the gold-set evaluation.

- [`using_narrative_likert_bert.ipynb`](tutorials/using_narrative_likert_bert.ipynb)
- [`using_event_relation_bert.ipynb`](tutorials/using_event_relation_bert.ipynb)

---

## Repository layout

```
src/
├── llm_annotation/
│   ├── data.py            # where inputs come from
│   ├── io.py              # the unit of work, output dirs, checkpoints
│   ├── clients.py         # model registry; batch and threaded runners
│   ├── prompts/
│   │   ├── likert.py           # the 9-dimension agency/setting prompt
│   │   └── event_relation.py   # the temporal + causal guidelines, 4 variants
│   ├── annotate_likert.py      # --model {claude,gemma,qwen} --split {gold,scale}
│   └── annotate_events.py      # --model {claude,gemma,qwen} --split {gold,scale}
└── narrabert/
    ├── training.py             # the loop both models share
    ├── train_likert.py         # trains narrative-likert-roberta
    └── train_event_relation.py # trains narrative-event-relation-roberta
tutorials/
```

Two entry points cover all seven annotation runs in the paper. A run is a
`--model` and a `--split`. Model is how the model is reached (Anthropic's batch API vs.
concurrent requests to Doubleword) and split is how it returns structure (a tool
definition vs. raw JSON) follow from the model, so they are properties of the
registry in `clients.py`, not of the scripts.

Here are some example runs you could do yourself:

| Run | Command |
|---|---|
| Claude, agency/setting, gold | `annotate_likert.py --model claude --split gold` |
| Gemma, agency/setting, gold | `annotate_likert.py --model gemma --split gold` |
| Qwen, agency/setting, gold | `annotate_likert.py --model qwen --split gold` |
| Gemma, agency/setting, scale | `annotate_likert.py --model gemma --split scale` |
| Claude, event relation, gold | `annotate_events.py --model claude --split gold` |
| Gemma, event relation, gold | `annotate_events.py --model gemma --split gold` |
| Gemma, event relation, scale | `annotate_events.py --model gemma --split scale` |

---

## Setup

```bash
conda create -n narrabert-env python=3.12 && conda activate narrabert-env
pip install -r requirements.txt
```

---

## Running the pipeline

### 1. Get the data

Download both datasets before running anything.

```python
from huggingface_hub import snapshot_download

# Human gold labels, used by the gold annotation runs and by both trainers
# for held-out evaluation.
snapshot_download(
    "CLS-Lab/narrative-gold-annotations", repo_type="dataset", local_dir="annotated_data",
    allow_patterns=["agency_annotations.parquet",
                    "setting_annotations.parquet",
                    "event_relation_annotations.parquet"],
)

# LLM pseudo-labels. The scale-up annotators read their *input passages* from
# here with the `pred_*` columns dropped. The passages were never published on
# their own so re-running one re-derives the labels over exactly the passages
# the paper used.
snapshot_download(
    "CLS-Lab/narrative-llm-annotations", repo_type="dataset", local_dir="llm_labels",
    allow_patterns=["agency_setting_llm_labels/*", "event_relation_llm_labels/*"],
)
```

### 2. LLM annotation

Claude scripts use the [Anthropic Message Batches API](https://docs.anthropic.com/en/api/creating-message-batches)
with prompt caching, and resume from a batch ID. Gemma and Qwen scripts call the
Doubleword AI API with concurrent requests and checkpointing; pass a prior
`--output-dir` to resume one.

```bash
export ANTHROPIC_API_KEY=...      # --model claude
export DOUBLEWORD_API_KEY=...     # --model gemma, --model qwen

# Gold runs — compare a model against the human labels
python src/llm_annotation/annotate_likert.py --model claude --split gold
python src/llm_annotation/annotate_events.py --model gemma  --split gold

# Scale-up runs — produce NarraBERT's pseudo-labels
python src/llm_annotation/annotate_likert.py --model gemma --split scale --workers 20
python src/llm_annotation/annotate_events.py --model gemma --split scale --workers 20

# Try a handful of items first: --n limits *passages*, so an event run never
# stops midway through one passage's pairs
python src/llm_annotation/annotate_events.py --model claude --split scale --n 5 --realtime

# Resume
python src/llm_annotation/annotate_likert.py --model claude --split gold --batch-id <batch_id>
python src/llm_annotation/annotate_likert.py --model gemma --split scale --output-dir <prior run>
```

Every run writes to a timestamped directory (`outputs/likert_<split>/<model>/<ts>/`
or `outputs/events_<split>/<model>/<ts>/`) holding `predictions.csv` (or
`pair_predictions.csv` and `text_scores.csv` for the event scale-up) alongside a
`run_metadata.json` recording the model, split, prompt variant, input, timings,
and batch id.

### 3. NarraBERT training

Both scripts run two modes.

**Pseudo-label mode** (`--pseudo-labels` / `--scale-up-dir`) trains on a 90/10
split of the LLM labels, early-stops on the 10% validation split, and then
evaluates on the held-out human gold set, writing `test_results.csv` and
`test_predictions.csv`. The gold set was excluded from the scale-up draw, so no
gold passage appears in the pseudo-labels, it is the only held-out signal here,
since the 10% validation split is pseudo-labels too and measures agreement with
the LLM annotator rather than accuracy.

**Cross-validation mode** (no flag) runs 5-fold CV and writes `cv_results.csv`.
Note the two scripts cross-validate over different data: `train_likert.py` uses
the 400 gold passages, while `train_event_relation.py` uses the pseudo-labels,
falling back to the most recent local scale-up run under
`outputs/events_scale/` unless `SCALE_UP_DIR` names one.

Checkpoints, `config.json`, and per-fold or test metrics land in a timestamped
directory under `checkpoints/`.

```bash
python src/narrabert/train_likert.py
python src/narrabert/train_likert.py \
    --pseudo-labels llm_labels/agency_setting_llm_labels/train-00000-of-00001.parquet

python src/narrabert/train_event_relation.py \
    --scale-up-dir llm_labels/event_relation_llm_labels
```

Pass a run directory from the scale-up annotators instead to train on labels you
generated yourself:

```bash
python src/narrabert/train_likert.py \
    --pseudo-labels outputs/likert_scale/google_gemma-4-31B-it/<ts>/predictions.csv
python src/narrabert/train_event_relation.py \
    --scale-up-dir outputs/events_scale/google_gemma-4-31B-it/<ts>
```

---

## Citation

```bibtex
@misc{johnson2026characterizingnarrativecontentwebscale,
      title={Characterizing Narrative Content in Web-scale LLM Pretraining Data}, 
      author={Teagan Johnson and Elliott Ash and Andrew Piper and Maria Antoniak},
      year={2026},
      eprint={2606.19468},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.19468}, 
}
```
