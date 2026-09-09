"""
Fine-tune RoBERTa-base for event relation identification.
Binary: sequential vs non-sequential (temporal), causal vs not (causal).
Uses LLM pseudo-labels from the event_relation scale-up pipeline.

Trains on a 90/10 split of pseudo-labels (early stopping on the 10% val set),
then evaluates on held-out gold annotations.

Usage:
    python src/narrabert/train_event_relation.py
    python src/narrabert/train_event_relation.py \
        --scale-up-dir llm_labels/event_relation_llm_labels

Outputs checkpoints/event_relation_{timestamp}/
    model.pt, tokenizer/, test_results.csv, config.json
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_annotation"))

import data  # noqa: E402  — needs the path insert above
from narrabert import training  # noqa: E402

warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME   = "roberta-base"
MAX_LEN      = 256
BATCH_SIZE   = 16
LR           = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS   = 20
PATIENCE     = 3
N_FOLDS      = 5
VAL_FRAC     = 0.1   # fraction of pseudo-labels held out for early stopping
SEED         = 42

SCALE_UP_MODEL = "google_gemma-4-31B-it"
SCALE_UP_DIR   = None   # None = auto-detect the latest local run under outputs/

ENTITY_MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]

TEMPORAL_MAP = {
    "span1_first": 1, "span2_first": 1,
    "simultaneous": 0, "too_hard_to_tell": 0, "same_event": 0,
}
CAUSAL_MAP = {
    "direct_cause": 1, "enables": 1,
    "not_related": 0,
}

DIMS = ["temporal_sequential", "causal"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def find_latest_scale_up(model_name: str) -> Path:
    base = data.REPO_ROOT / "outputs" / "events_scale" / model_name
    runs = sorted(
        d for d in (base.iterdir() if base.is_dir() else ())
        if d.is_dir() and (d / "pair_predictions.csv").exists()
    )
    if not runs:
        raise FileNotFoundError(
            f"No pair_predictions.csv found under {base}. Either run the "
            f"scale-up annotator first:\n"
            f"    python src/llm_annotation/annotate_events.py --model gemma --split scale\n"
            f"or train against the published pseudo-labels with "
            f"--scale-up-dir llm_labels/{data.PAIRS_CONFIG}"
        )
    return runs[-1]


def insert_markers(text: str, span1: list, span2: list) -> str:
    """Insert [E1]/[E2] markers at character offsets, right-to-left to preserve positions."""
    insertions = sorted([
        (span1[0], "[E1]"), (span1[1], "[/E1]"),
        (span2[0], "[E2]"), (span2[1], "[/E2]"),
    ], key=lambda x: -x[0])
    for pos, marker in insertions:
        text = text[:pos] + marker + text[pos:]
    return text

# ── Data ───────────────────────────────────────────────────────────────────────

def load_pairs(source: Path) -> pd.DataFrame:
    """Pair predictions from an annotator run directory or a parquet source.

    Three shapes reach here: a run directory holding `pair_predictions.csv`
    (what the scale-up annotators write), a directory of parquet shards (what
    a Hugging Face snapshot of the published pseudo-labels looks like), and a
    single csv/parquet file.
    """
    source = Path(source)
    if not source.is_absolute():
        source = data.REPO_ROOT / source

    if source.is_dir():
        csv = source / "pair_predictions.csv"
        if csv.exists():
            return pd.read_csv(csv)
        shards = sorted(source.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"{source} holds neither pair_predictions.csv nor parquet shards."
            )
        return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    return pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)


def load_data(scale_up_dir: Path) -> tuple[list[str], np.ndarray, list[str]]:
    df = load_pairs(scale_up_dir)

    texts = [
        insert_markers(row["sampled_text"], json.loads(row["span1"]), json.loads(row["span2"]))
        for _, row in df.iterrows()
    ]

    temporal = df["pred_temporal_order"].map(TEMPORAL_MAP).values.astype(float)
    causal   = df["pred_causality_rating"].map(CAUSAL_MAP).values.astype(float)
    labels   = np.stack([temporal, causal], axis=1)

    # Pair-grain ids: (id, chunk_index, pair_idx). Dropping chunk_index would
    # make two passages of the same document share ids.
    key = data.passage_key_cols(df)
    inst_ids = (
        df[key].astype(str).agg("::".join, axis=1) + "__pair_" + df["pair_idx"].astype(str)
    ).tolist()

    print(f"Loaded {len(df)} pairs from {Path(scale_up_dir).name}")
    for name, col in [("temporal", temporal), ("causal", causal)]:
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        n_nan = int(np.isnan(col).sum())
        pos_rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
        print(f"  {name}: {n_pos} pos, {n_neg} neg, {n_nan} skipped  (pos_rate={pos_rate:.2f})")

    return texts, labels, inst_ids


def load_gold_data() -> tuple[list[str], np.ndarray, list[str]]:
    df = data.load_gold("event_relation")
    df = df[df["span1_is_event_gold"] & df["span2_is_event_gold"]].reset_index(drop=True)

    texts = [
        insert_markers(row["sampled_text"], json.loads(row["assigned_span1"]), json.loads(row["assigned_span2"]))
        for _, row in df.iterrows()
    ]

    temporal = df["temporal_order_gold"].map(TEMPORAL_MAP).values.astype(float)
    causal   = df["causality_rating_gold"].map(CAUSAL_MAP).values.astype(float)
    causal[df["temporal_order_gold"].values == "simultaneous"] = np.nan

    labels   = np.stack([temporal, causal], axis=1)
    inst_ids = df["safe_instance_id"].tolist()

    print(f"Loaded {len(df)} gold pairs (both spans confirmed events)")
    for name, col in [("temporal", temporal), ("causal", causal)]:
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        n_nan = int(np.isnan(col).sum())
        pos_rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
        print(f"  {name}: {n_pos} pos, {n_neg} neg, {n_nan} skipped  (pos_rate={pos_rate:.2f})")

    return texts, labels, inst_ids


class EventRelationDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

# ── Model ──────────────────────────────────────────────────────────────────────

class EventRelationRoBERTa(nn.Module):
    def __init__(self, model_name: str, n_new_tokens: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.resize_token_embeddings(self.backbone.config.vocab_size + n_new_tokens)
        hidden = self.backbone.config.hidden_size
        self.temporal_head = nn.Linear(hidden, 1)
        self.causal_head   = nn.Linear(hidden, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return self.temporal_head(cls).squeeze(-1), self.causal_head(cls).squeeze(-1)

# ── Training ───────────────────────────────────────────────────────────────────

def build_model() -> "EventRelationRoBERTa":
    return EventRelationRoBERTa(MODEL_NAME, len(ENTITY_MARKERS))


def combined_bce(outputs, labels: torch.Tensor) -> torch.Tensor:
    """Sum the masked BCE of both heads. `outputs` is (t_logits, c_logits)."""
    t_logits, c_logits = outputs
    return masked_bce(t_logits, labels[:, 0]) + masked_bce(c_logits, labels[:, 1])


def make_score_val(device):
    """Early-stopping criterion for this model: validation BCE, not F1.

    F1 is reported alongside it but not used to stop: it moves in coarse steps
    on a two-class problem, so it plateaus while the loss is still improving.
    """
    def score_val(model, loader) -> tuple[float, str]:
        t_log, c_log, labs = predict(model, loader, device)
        vloss = val_loss(t_log, c_log, labs)
        f1_avg = mean_f1(compute_metrics(t_log, c_log, labs))
        return vloss, f"val_loss={vloss:.4f}  val_F1={f1_avg:.4f}"
    return score_val


#: Hyperparameters the shared loop needs, fixed for this model.
LOOP = dict(batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            lr=LR, weight_decay=WEIGHT_DECAY)


def masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(labels)
    if not mask.any():
        return (logits * 0).sum()   # zero loss, preserves computation graph
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask])


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_t, all_c, all_labs = [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        t_logits, c_logits = model(ids, mask)
        all_t.append(t_logits.cpu().numpy())
        all_c.append(c_logits.cpu().numpy())
        all_labs.append(batch["labels"].numpy())
    return np.concatenate(all_t), np.concatenate(all_c), np.vstack(all_labs)


def val_loss(t_logits: np.ndarray, c_logits: np.ndarray, labs: np.ndarray) -> float:
    total = 0.0
    for i, logits in enumerate([t_logits, c_logits]):
        valid = ~np.isnan(labs[:, i])
        if valid.sum() > 0:
            t = torch.tensor(logits[valid], dtype=torch.float32)
            l = torch.tensor(labs[valid, i], dtype=torch.float32)
            total += nn.functional.binary_cross_entropy_with_logits(t, l).item()
    return total

# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(t_logits: np.ndarray, c_logits: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for i, (name, logits) in enumerate(zip(DIMS, [t_logits, c_logits])):
        valid = ~np.isnan(labels[:, i])
        if valid.sum() < 2:
            rows.append({"dim": name, "N": int(valid.sum()), "Accuracy": float("nan"),
                         "Kappa": float("nan"), "Precision": float("nan"),
                         "Recall": float("nan"), "F1": float("nan")})
            continue
        g = labels[valid, i].astype(int)
        p = (logits[valid] > 0).astype(int)
        rows.append({
            "dim":       name,
            "N":         int(valid.sum()),
            "Accuracy":  round(float((p == g).mean()), 3),
            "Kappa":     round(float(cohen_kappa_score(g, p)), 3),
            "Precision": round(float(precision_score(g, p, zero_division=0)), 3),
            "Recall":    round(float(recall_score(g, p, zero_division=0)), 3),
            "F1":        round(float(f1_score(g, p, zero_division=0)), 3),
        })
    return rows


def mean_f1(metrics: list[dict]) -> float:
    vals = [m["F1"] for m in metrics if not np.isnan(m["F1"])]
    return float(np.mean(vals)) if vals else 0.0

# ── CV + final training ────────────────────────────────────────────────────────

def run_cv(
    texts, labels, inst_ids, tokenizer, device
) -> tuple[pd.DataFrame, int, list[int], pd.DataFrame]:
    kf           = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_rows    = []
    best_epochs  = []
    best_fold_f1 = -1.0
    best_fold_df = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} {'─'*50}")

        train_ds = EventRelationDataset(
            [texts[i] for i in train_idx], labels[train_idx], tokenizer, MAX_LEN
        )
        val_ds = EventRelationDataset(
            [texts[i] for i in val_idx], labels[val_idx], tokenizer, MAX_LEN
        )
        model, best_epoch = training.train_with_early_stopping(
            build_model, train_ds, val_ds, device,
            loss_fn=combined_bce, score_fn=make_score_val(device), **LOOP,
        )
        best_epochs.append(best_epoch)

        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        t_log, c_log, labs = predict(model, val_loader, device)
        metrics  = compute_metrics(t_log, c_log, labs)
        fold_f1  = mean_f1(metrics)
        print(f"\n  Results (best epoch {best_epoch}):")
        for m in metrics:
            print(f"    {m['dim']:25s}  Acc={m['Accuracy']:.3f}  P={m['Precision']:.3f}  R={m['Recall']:.3f}  F1={m['F1']:.3f}")
        fold_rows.extend([{"fold": fold, **m} for m in metrics])

        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_fold_df = pd.DataFrame({
                "inst_id":                            [inst_ids[i] for i in val_idx],
                "gold_temporal_sequential":           labs[:, 0],
                "gold_causal":                        labs[:, 1],
                "pred_temporal_sequential_logit":     t_log,
                "pred_causal_logit":                  c_log,
                "pred_temporal_sequential":           (t_log > 0).astype(int),
                "pred_causal":                        (c_log > 0).astype(int),
            })

    cv_df       = pd.DataFrame(fold_rows)
    final_epoch = int(np.median(best_epochs))
    return cv_df, final_epoch, best_epochs, best_fold_df


def train_final(texts, labels, tokenizer, device, n_epochs: int) -> "EventRelationRoBERTa":
    print(f"\n── Final model — {n_epochs} epochs on all {len(texts)} pairs {'─'*30}")
    return training.train_fixed_epochs(
        build_model, EventRelationDataset(texts, labels, tokenizer, MAX_LEN), device,
        loss_fn=combined_bce, n_epochs=n_epochs,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for event relation classification.")
    parser.add_argument(
        "--scale-up-dir", type=str, default=None,
        help="Path to a scale-up output dir containing pair_predictions.csv. "
             "Trains on a 90/10 split of pseudo-labels and evaluates on gold annotations.",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = training.select_device()
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(ENTITY_MARKERS, special_tokens=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = data.REPO_ROOT / "checkpoints" / f"event_relation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pseudo-label mode ──────────────────────────────────────────────────────
    if args.scale_up_dir:
        scale_dir = args.scale_up_dir
        texts_pl, labels_pl, ids_pl = load_data(scale_dir)

        n_total = len(texts_pl)
        n_val   = max(1, int(n_total * VAL_FRAC))
        rng     = np.random.RandomState(SEED)
        idx     = rng.permutation(n_total)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        print(f"\nSplit: {len(train_idx)} train  /  {n_val} val  (VAL_FRAC={VAL_FRAC})")

        train_ds = EventRelationDataset(
            [texts_pl[i] for i in train_idx], labels_pl[train_idx], tokenizer, MAX_LEN
        )
        val_ds = EventRelationDataset(
            [texts_pl[i] for i in val_idx], labels_pl[val_idx], tokenizer, MAX_LEN
        )

        print("\n── Training (early stopping on pseudo-label val set) " + "─" * 25)
        model, best_epoch = training.train_with_early_stopping(
            build_model, train_ds, val_ds, device,
            loss_fn=combined_bce, score_fn=make_score_val(device), **LOOP,
        )

        # ── Held-out gold evaluation ──────────────────────────────────────────
        # The 10% val split is pseudo-labels too, so it measures agreement with
        # the LLM annotator, not accuracy. The human gold pairs are the only
        # held-out signal — they were excluded from the scale-up draw, and only
        # pairs whose spans both humans confirmed as events are scored.
        texts_g, labels_g, ids_g = load_gold_data()
        gold_loader = DataLoader(
            EventRelationDataset(texts_g, labels_g, tokenizer, MAX_LEN), batch_size=BATCH_SIZE
        )
        t_logits, c_logits, labs_g = predict(model, gold_loader, device)
        test_metrics = compute_metrics(t_logits, c_logits, labs_g)

        print("\n── Gold evaluation " + "─" * 55)
        for m in test_metrics:
            print(f"    {m['dim']:22s}  N={m['N']:4d}  Acc={m['Accuracy']:.3f}  "
                  f"F1={m['F1']:.3f}  kappa={m['Kappa']:.3f}")

        pd.DataFrame(test_metrics).to_csv(out_dir / "test_results.csv", index=False)

        pd.DataFrame({
            "instance_id":         ids_g,
            "gold_temporal":       labs_g[:, 0],
            "pred_temporal":       (t_logits > 0).astype(int),
            "gold_causal":         labs_g[:, 1],
            "pred_causal":         (c_logits > 0).astype(int),
        }).to_csv(out_dir / "test_predictions.csv", index=False)

        torch.save(model.state_dict(), out_dir / "model.pt")
        tokenizer.save_pretrained(out_dir / "tokenizer")
        (out_dir / "config.json").write_text(json.dumps({
            "model_name":  MODEL_NAME,
            "max_len":     MAX_LEN,
            "dims":        DIMS,
            "data_source": str(scale_dir),
            "n_train":     len(train_idx),
            "n_val":       n_val,
            "val_frac":    VAL_FRAC,
            "best_epoch":  best_epoch,
            "seed":        SEED,
            "n_gold_test": len(texts_g),
            "gold_mean_f1": round(mean_f1(test_metrics), 4),
        }, indent=2))

    # ── Cross-validation mode ─────────────────────────────────────────────────
    # Note this cross-validates the *pseudo-labels*, not the gold pairs — there
    # are only 236 gold pairs with both spans confirmed as events, too few to
    # fold. train_likert.py's no-flag mode does use gold, so the two scripts
    # differ here.
    else:
        scale_dir = Path(SCALE_UP_DIR) if SCALE_UP_DIR else find_latest_scale_up(SCALE_UP_MODEL)
        print(f"Scale-up dir: {scale_dir}\n")

        texts, labels, inst_ids = load_data(scale_dir)

        cv_df, final_epoch, best_epochs, best_fold_df = run_cv(
            texts, labels, inst_ids, tokenizer, device
        )

        summary = cv_df.groupby("dim")[["Accuracy", "F1"]].agg(["mean", "std"]).round(3)
        print("\n── CV Summary " + "─" * 60)
        print(summary.to_string())
        print(f"\nBest epochs / fold: {best_epochs}  → median {final_epoch}")

        final_model = train_final(texts, labels, tokenizer, device, final_epoch)

        torch.save(final_model.state_dict(), out_dir / "model.pt")
        tokenizer.save_pretrained(out_dir / "tokenizer")
        cv_df.to_csv(out_dir / "cv_results.csv", index=False)
        best_fold_df.to_csv(out_dir / "best_fold_predictions.csv", index=False)
        (out_dir / "config.json").write_text(json.dumps({
            "model_name":   MODEL_NAME,
            "max_len":      MAX_LEN,
            "dims":         DIMS,
            "n_folds":      N_FOLDS,
            "final_epochs": final_epoch,
            "seed":         SEED,
            "scale_up_dir": str(scale_dir),
        }, indent=2))

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
