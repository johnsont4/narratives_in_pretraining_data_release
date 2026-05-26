# Narratives in Pretraining Data — Data and Model Release

This repository contains the data and modeling code accompanying the paper:

> **Characterizing Narrative Content in Web-scale LLM Pretraining Data**
> *[AUTHORS REDACTED FOR REVIEW]*

We release a human-annotated gold dataset, large-scale LLM and NarraBERT predictions over subset of NarraDolma (a Dolma pretraining corpus subset), and the training code and model configs for our NarraBERT classifiers.

> **Note:** Model weights and full NarraDolma dataset are not included in this release due to file size constraints. Weights and the full corpus will be made publicly available upon paper acceptance.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Dataset Overview](#dataset-overview)
3. [Annotation Schema](#annotation-schema)
4. [Data Files](#data-files)
5. [Models](#models)
6. [LLM Annotation Scripts](#llm-annotation-scripts)
7. [NarraBERT Training Scripts](#narrabert-training-scripts)
8. [Reproducing the Pipeline](#reproducing-the-pipeline)

---

## Repository Structure

```
.
├── data/
│   ├── human_gold_labels/
│   │   └── gold_annotations.parquet          # 456-instance human-annotated gold set
│   ├── llm_labels/
│   │   ├── llm_gold_preds/                   # LLM predictions on the gold set
│   │   │   ├── claude_sonnet_4_6_agency_setting_gold_preds.csv
│   │   │   ├── claude_sonnet_4_6_events_gold_preds.csv
│   │   │   ├── gemma_agency_setting_gold_preds.csv
│   │   │   ├── gemma_events_gold_preds.csv
│   │   │   └── qwen_agency_setting_gold_preds.csv
│   │   └── scaled_preds/                     # Large-scale LLM predictions over NarraDolma
│   │       ├── text_scores.parquet           # Text-level Likert scores (Gemma)
│   │       └── gemma_global_event_scores.parquet
│   └── narradolma/
│       ├── narrabert_gold_preds/             # NarraBERT predictions on the gold set
│       │   ├── narrarbert_agency_setting_gold_preds.csv
│       │   └── narrabert_finegrained_events_gold_preds.csv
│       └── scaled_preds_subset/              # NarraBERT predictions over NarraDolma subset
│           ├── narradolma_base_sample.parquet
│           ├── narradolma_agency_setting_preds.parquet
│           ├── narradolma_finegrained_event_labels.parquet
│           └── narradolma_global_event_scores.parquet
├── models/
│   ├── agency_setting_narrabert/
│   │   ├── config.json                       # Hyperparameters and training metadata
│   │   └── tokenizer/                        # RoBERTa tokenizer files
│   ├── events_narrabert/
│   │   ├── config.json
│   │   └── tokenizer/
│   ├── llm_api_calls/
│   │   ├── likert/                           # Likert annotation scripts
│   │   │   ├── claude_sonnet_4_6_api.py
│   │   │   ├── gemma_api.py
│   │   │   └── qwen_api.py
│   │   └── events/                           # Event relation annotation scripts
│   │       ├── claude_sonnet_4_6_events_api.py
│   │       └── gemma_events_api.py
│   └── narrabert_training_files/
│       ├── train_likert.py                   # Trains agency_setting_narrabert
│       └── train_event_relation.py           # Trains events_narrabert
└── README.md
```

---

## Dataset Overview

### NarraDolma

**NarraDolma** is a labeled subset of the [Dolma](https://huggingface.co/datasets/allenai/dolma) pretraining corpus annotated for narrative properties. Texts were sampled from Dolma and filtered for narrativity/topical content before large-scale annotation.

| Property | Value |
|---|---|
| Source corpus | Dolma |
| Dolma version | 1.7 |
| Total NarraDolma rows provided | 209,708 passages (out of ~3M) |
| Human gold set size | 456 instances (456 event labels, 400 agency and setting labels) |

### Gold Annotation

Human annotations were collected using **Potato**. Each instance was annotated by **2-3** annotators. Inter-annotator agreement (IAA) statistics are reported in the paper.

---

## Annotation Schema

We annotate three conceptually distinct aspects of narrative structure: **agency and setting dimensions** (Likert-scale ratings) and **event relations** (categorical labels between event span pairs). See paper for more details.

### Agency and Setting Dimensions (Likert 1–5)

Each text is rated on 9 dimensions on a 1–5 Likert scale:

**Agency dimensions** (how vividly characters are portrayed as agents):

| Dimension | Description |
|---|---|
| `focalization` | Degree to which the text is focalized through a specific character's perspective |
| `emotion` | Degree to which characters' emotional states are explicitly described |
| `cognition` | Degree to which characters' thoughts, beliefs, or mental states are described |
| `change_of_state` | Degree to which characters undergo a meaningful change over the narrative |
| `conflict` | Degree to which characters are in conflict with each other or external forces |

**Setting dimensions** (how concretely the narrative world is situated):

| Dimension | Description |
|---|---|
| `concreteness` | Degree to which the setting is described in concrete, specific terms |
| `temporal_grounding` | Degree to which events are anchored to specific times or a clear time sequence |
| `spatial_grounding` | Degree to which events are anchored to specific physical locations |
| `sensory` | Degree to which the text contains sensory descriptions (sight, sound, touch, etc.) |

### Event Relations (Categorical)

For each neighboring pair of event spans within a text, annotators judged:

| Label type | Values |
|---|---|
| `span1_is_event` | `true` / `false` — whether the first span is a genuine event trigger |
| `span2_is_event` | `true` / `false` — whether the second span is a genuine event trigger |
| `temporal_order` | `span1_first`, `span2_first`, `simultaneous`, `same_event`, `too_hard_to_tell` (or `not_applicable` if either span is not an event) |
| `causality_rating` | `direct_cause`, `enables`, `not_related` (or `not_applicable` if either span is not an event) |

Event spans were identified using a verb-extraction pipeline applied to the `sampled_text` field. Neighboring pairs are consecutive spans in the extracted span sequence.

For model training, event relations are binarized:
- **temporal_sequential**: `span1_first` or `span2_first` → 1; all others → 0
- **causal**: `direct_cause` or `enables` → 1; `not_related` → 0

---

## Data Files

### `data/human_gold_labels/gold_annotations.parquet`

The primary human-annotated dataset (456 total instances). Key columns:

| Column | Description |
|---|---|
| `safe_instance_id` | Anonymized instance identifier (format: `inst_<hash>_<idx>`) |
| `dolma_id` | Original Dolma document URL/identifier |
| `dolma_shard` | Dolma shard the document came from |
| `dolma_source` | Dolma domain (e.g., `common-crawl`, `reddit`, `gutenberg`) |
| `sampled_text` | The text passage that was annotated (a passage sampled from `full_text`) |
| `full_text` | Full Dolma document text |
| `narrative_label` | `Narrative` or `Non-narrative` from DeBERTa narrative classifier, see paper for more |
| `narrative_confidence` | Classifier confidence |
| `topic_classification` | Topic classification from WebOrganizer |
| `llm_summary` | LLM-generated summary used during annotation |
| `verb_tokens` / `verb_spans` / `verb_count` | Spacy verb extraction outputs |
| `event_tokens` / `event_spans` / `event_count` | LitBank event span extraction outputs |
| `assigned_span1` / `assigned_span2` | The two event spans selected for relation annotation (JSON: `[start, end, text, type]`) |
| `agency_{dim}_gold` | Human gold Likert rating (1–5) for each agency dimension |
| `setting_{dim}_gold` | Human gold Likert rating (1–5) for each setting dimension |
| `span1_is_event_gold` / `span2_is_event_gold` | Boolean event trigger judgments |
| `temporal_order_gold` | Categorical temporal relation label |
| `causality_rating_gold` | Categorical causal relation label |

---

### `data/llm_labels/llm_gold_preds/`

LLM predictions on the 456 gold instances. Used to benchmark LLM performance against human gold labels.

| File | Model | Task | Key columns |
|---|---|---|---|
| `claude_sonnet_4_6_agency_setting_gold_preds.csv` | Claude Sonnet 4.6 | Likert (9 dims) | `safe_instance_id`, `sampled_text`, `pred_{dim}` (1–5) for all 9 dims |
| `claude_sonnet_4_6_events_gold_preds.csv` | Claude Sonnet 4.6 | Event relations | `safe_instance_id`, `pred_span1_is_event`, `pred_span2_is_event`, `pred_temporal_order`, `pred_causality_rating` |
| `gemma_agency_setting_gold_preds.csv` | Gemma 4-31B-it | Likert (9 dims) | same as Claude Likert format |
| `gemma_events_gold_preds.csv` | Gemma 4-31B-it | Event relations | same as Claude events format |
| `qwen_agency_setting_gold_preds.csv` | Qwen 3.6-35B-A3B-FP8 | Likert (9 dims) | same as Claude Likert format |

Claude predictions were generated via the [Anthropic Message Batches API](https://docs.anthropic.com/en/api/creating-message-batches) with prompt caching. Gemma and Qwen predictions were generated via the Doubleword AI API.

---

### `data/llm_labels/scaled_preds/`

Large-scale Gemma 4-31B-it predictions over 5k instances.

| File | Description |
|---|---|
| `gemma_event_scores.parquet` | Text-level Likert scores (columns: `safe_instance_id` or `id`, `pred_{dim}` for all 9 dims) |
| `gemma_global_event_scores.parquet` | Text-level aggregated event scores: `n_event_spans`, `temporal_sequencing` (fraction of sequential pairs), `causal_density` (fraction of causal pairs) |

These pseudo-labels were used to train NarraBERT.

---

### `data/narradolma/narrabert_gold_preds/`

NarraBERT predictions on the 456 gold instances. Used to evaluate trained model performance.

| File | Model | Key columns |
|---|---|---|
| `narrarbert_agency_setting_gold_preds.csv` | agency_setting_narrabert | `safe_instance_id`, `pred_{dim}` (continuous, 1–5 scale) for all 9 dims |
| `narrabert_finegrained_events_gold_preds.csv` | events_narrabert | `safe_instance_id`, `pred_temporal_sequential` (binary), `pred_causal` (binary), `temporal_logit`, `causal_logit` |

---

### `data/narradolma/scaled_preds_subset/`

NarraBERT predictions over the NarraDolma subset. The primary release of narrative labels at scale.

| File | Description |
|---|---|
| `narradolma_base_sample.parquet` | Base sample of texts from Dolma (includes `safe_instance_id`, `sampled_text`, and metadata) |
| `narradolma_agency_setting_preds.parquet` | agency_setting_narrabert predictions over the full subset |
| `narradolma_finegrained_event_labels.parquet` | events_narrabert predictions at the event-pair level |
| `narradolma_global_event_scores.parquet` | Text-level aggregated event scores (temporal_sequencing, causal_density) |

---

## Models

Both NarraBERT models are fine-tuned from `roberta-base` ([Liu et al., 2019](https://arxiv.org/abs/1907.11692)). Model weights are not included in this release, they will be provided upon paper acceptance.

### `agency_setting_narrabert`

Multi-task Likert regression model predicting all 9 agency and setting dimensions simultaneously.

| Property | Value |
|---|---|
| Base model | `roberta-base` |
| Task | Multi-task regression (9 Likert dimensions, 1–5 scale) |
| Architecture | RoBERTa + 9 independent linear regression heads on `[CLS]` token |
| Loss | Masked MSE (handles missing labels per instance) |
| Max sequence length | 200 tokens |
| Training pseudo-labels | Gemma 4-31B-it predictions |
| Train/val split | 90/10 of pseudo-label data |
| Best epoch | 13 |
| Optimizer | AdamW (lr=2e-5, weight_decay=0.01) |
| Gold test MAE | **0.581** |

### `events_narrabert`

Binary classification model predicting temporal sequencing and causality of neighboring event span pairs.

| Property | Value |
|---|---|
| Base model | `roberta-base` |
| Task | Binary classification: `temporal_sequential`, `causal` |
| Architecture | RoBERTa (with `[E1]`/`[E2]` entity markers) + 2 linear classification heads on `[CLS]` |
| Loss | Masked BCE (handles `not_applicable` pairs) |
| Max sequence length | 256 tokens |
| Input format | Text with span markers inserted: `[E1]span1[/E1] ... [E2]span2[/E2]` |
| Training pseudo-labels | Gemma 4-31B-it event relation predictions |
| Train/val split | 90/10 of pseudo-label data |
| Best epoch | 4 |
| Optimizer | AdamW (lr=2e-5, weight_decay=0.01) |
| Gold test F1 (macro avg) | **0.805** |

---

## LLM Annotation Scripts

Located in `models/llm_api_calls/`. These scripts were used to generate pseudo-labels and to benchmark LLMs on the gold set.

### Likert Annotation

**`models/llm_api_calls/likert/claude_sonnet_4_6_api.py`**

Annotates texts on the 9 Likert narrative dimensions using Claude Sonnet 4.6 via the Anthropic Message Batches API with prompt caching. Supports batch submission and resumption by batch ID.

```bash
export ANTHROPIC_API_KEY=<your_key>
python models/llm_api_calls/likert/claude_sonnet_4_6_api.py --n 100
python models/llm_api_calls/likert/claude_sonnet_4_6_api.py --batch-id <existing_batch_id>
```

**`models/llm_api_calls/likert/gemma_api.py`**

Annotates texts on the 9 Likert dimensions using Gemma 4-31B-it via the Doubleword AI API. Supports parallel requests and checkpointing for large-scale runs.

```bash
export DOUBLEWORD_API_KEY=<your_key>
python models/llm_api_calls/likert/gemma_api.py --n 5000 --workers 20
python models/llm_api_calls/likert/gemma_api.py --output-dir outputs/scale/google_gemma-4-31B-it/<timestamp>  # resume
```

**`models/llm_api_calls/likert/qwen_api.py`**

Annotates texts on the 9 Likert dimensions using Qwen 3.6-35B-A3B-FP8 via the Doubleword AI API.

```bash
export DOUBLEWORD_API_KEY=<your_key>
python models/llm_api_calls/likert/qwen_api.py --n 100
```

### Event Relation Annotation

**`models/llm_api_calls/events/claude_sonnet_4_6_events_api.py`**

Annotates neighboring event span pairs for temporal order and causality using Claude Sonnet 4.6. Supports both the Batches API (default) and real-time single requests (`--realtime`). Also aggregates pair-level predictions to text-level `temporal_sequencing` and `causal_density` scores.

```bash
export ANTHROPIC_API_KEY=<your_key>
python models/llm_api_calls/events/claude_sonnet_4_6_events_api.py
python models/llm_api_calls/events/claude_sonnet_4_6_events_api.py --realtime --limit 10
python models/llm_api_calls/events/claude_sonnet_4_6_events_api.py --batch-id <existing_batch_id>
```

**`models/llm_api_calls/events/gemma_events_api.py`**

Annotates event span pairs using Gemma 4-31B-it via the Doubleword AI API with concurrent requests.

```bash
export DOUBLEWORD_API_KEY=<your_key>
python models/llm_api_calls/events/gemma_events_api.py --n 193 --workers 10
```

---

## NarraBERT Training Scripts

Located in `models/narrabert_training_files/`. Both scripts support two modes: **gold-only** (5-fold cross-validation on human labels) and **pseudo-label** (train on LLM-generated labels, evaluate on gold).

### `train_likert.py`

Fine-tunes `roberta-base` for multi-task Likert regression.

```bash
# Gold-only mode: 5-fold CV on human annotations
python models/narrabert_training_files/train_likert.py

# Pseudo-label mode: train on Gemma predictions, evaluate on gold
python models/narrabert_training_files/train_likert.py \
    --pseudo-labels <path_to_scale_outputs>/google_gemma-4-31B-it/<timestamp>/predictions.csv
```

**Outputs** (in `bert_modeling/checkpoints/<timestamp>/`):
- `model.pt` — model weights
- `tokenizer/` — saved tokenizer
- `config.json` — hyperparameters and training metadata
- `cv_results.csv` — per-fold metrics (gold-only mode)
- `test_results.csv` — gold set evaluation metrics (pseudo-label mode)

### `train_event_relation.py`

Fine-tunes `roberta-base` for binary event relation classification. Adds `[E1]`/`[/E1]`/`[E2]`/`[/E2]` entity markers to the tokenizer vocabulary.

```bash
# Gold-only mode: 5-fold CV on human annotations
python models/narrabert_training_files/train_event_relation.py

# Pseudo-label mode
python models/narrabert_training_files/train_event_relation.py \
    --scale-up-dir <path_to_event_relation_outputs>/google_gemma-4-31B-it/<timestamp>
```

**Outputs** (in `bert_modeling/checkpoints/event_relation_<timestamp>/`):
- `model.pt`, `tokenizer/`, `config.json`
- `cv_results.csv`, `best_fold_predictions.csv` (gold-only mode)

---

## Reproducing the Pipeline

The full labelling pipeline proceeds in the following order:

1. **Sample texts from Dolma**

2. **Run LLM Likert annotation at scale** (generates pseudo-labels):
   ```bash
   python models/llm_api_calls/likert/gemma_api.py --n <N>
   ```

3. **Run LLM event relation annotation at scale**:
   ```bash
   python models/llm_api_calls/events/gemma_events_api.py --n <N>
   ```

4. **Train NarraBERT models on pseudo-labels**:
   ```bash
   python models/narrabert_training_files/train_likert.py \
       --pseudo-labels <scale_output_path>/predictions.csv

   python models/narrabert_training_files/train_event_relation.py \
       --scale-up-dir <event_relation_output_path>
   ```

5. **Run NarraBERT inference on NarraDolma**

6. **Evaluate on gold set**

### Dependencies

```
anthropic
openai
pandas
pyarrow
torch
transformers
scikit-learn
scipy
numpy
```

Install with:
```bash
pip install anthropic openai pandas pyarrow torch transformers scikit-learn scipy numpy
```

### API Keys

| Service | Environment Variable | Used by |
|---|---|---|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `claude_sonnet_4_6_api.py`, `claude_sonnet_4_6_events_api.py` |
| Doubleword AI (Gemma, Qwen) | `DOUBLEWORD_API_KEY` | `gemma_api.py`, `qwen_api.py`, `gemma_events_api.py` |
