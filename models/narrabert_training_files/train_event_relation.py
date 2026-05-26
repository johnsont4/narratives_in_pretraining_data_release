"""
Fine-tune RoBERTa-base for event relation identification.
Binary: sequential vs non-sequential (temporal), causal vs not (causal).
Uses LLM pseudo-labels from the event_relation scale-up pipeline.

Trains on a 90/10 split of pseudo-labels (early stopping on the 10% val set),
then evaluates on held-out gold annotations.

Usage:
    python bert_modeling/train_event_relation.py
    python bert_modeling/train_event_relation.py --scale-up-dir event_relation/outputs/google_gemma-4-31B-it/<ts>

Outputs bert_modeling/checkpoints/event_relation_{timestamp}/
    model.pt, tokenizer/, test_results.csv, config.json
"""

import argparse
import json
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
SCALE_UP_DIR   = None   # None = auto-detect latest run; or Path("event_relation/outputs/model/ts")

GOLD_PARQUET = "annotated_data/event_relation_annotations.parquet"

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
    base = Path("event_relation/outputs") / model_name
    runs = sorted(
        d for d in base.iterdir()
        if d.is_dir() and (d / "pair_predictions.csv").exists()
    )
    if not runs:
        raise FileNotFoundError(f"No pair_predictions.csv found under {base}")
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

def load_data(scale_up_dir: Path) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(scale_up_dir / "pair_predictions.csv")

    texts = [
        insert_markers(row["sampled_text"], json.loads(row["span1"]), json.loads(row["span2"]))
        for _, row in df.iterrows()
    ]

    temporal = df["pred_temporal_order"].map(TEMPORAL_MAP).values.astype(float)
    causal   = df["pred_causality_rating"].map(CAUSAL_MAP).values.astype(float)
    labels   = np.stack([temporal, causal], axis=1)

    inst_ids = (df["safe_instance_id"] + "__pair_" + df["pair_idx"].astype(str)).tolist()

    print(f"Loaded {len(df)} pairs from {scale_up_dir.name}")
    for name, col in [("temporal", temporal), ("causal", causal)]:
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        n_nan = int(np.isnan(col).sum())
        pos_rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
        print(f"  {name}: {n_pos} pos, {n_neg} neg, {n_nan} skipped  (pos_rate={pos_rate:.2f})")

    return texts, labels, inst_ids


def load_gold_data() -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_parquet(GOLD_PARQUET)
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

def masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(labels)
    if not mask.any():
        return (logits * 0).sum()   # zero loss, preserves computation graph
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def make_optimizer_scheduler(model: nn.Module, steps_per_epoch: int, n_epochs: int):
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = steps_per_epoch * n_epochs
    warmup_steps = total_steps // 10
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["labels"].to(device)
        optimizer.zero_grad()
        t_logits, c_logits = model(ids, mask)
        loss = masked_bce(t_logits, labs[:, 0]) + masked_bce(c_logits, labs[:, 1])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


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
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

        model     = EventRelationRoBERTa(MODEL_NAME, len(ENTITY_MARKERS)).to(device)
        optimizer, scheduler = make_optimizer_scheduler(model, len(train_loader), MAX_EPOCHS)

        best_vloss, best_state, best_epoch, patience = float("inf"), None, 0, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            t_log, c_log, labs = predict(model, val_loader, device)
            vloss   = val_loss(t_log, c_log, labs)
            metrics = compute_metrics(t_log, c_log, labs)
            f1_avg  = mean_f1(metrics)
            print(f"  epoch {epoch:2d}  train={train_loss:.4f}  val_loss={vloss:.4f}  val_F1={f1_avg:.4f}")
            if vloss < best_vloss:
                best_vloss, best_epoch, patience = vloss, epoch, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f"  early stop at epoch {epoch}")
                    break

        best_epochs.append(best_epoch)
        model.load_state_dict(best_state)
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


def train_final(texts, labels, tokenizer, device, n_epochs: int) -> EventRelationRoBERTa:
    print(f"\n── Final model — {n_epochs} epochs on all {len(texts)} pairs {'─'*30}")
    full_ds     = EventRelationDataset(texts, labels, tokenizer, MAX_LEN)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
    model       = EventRelationRoBERTa(MODEL_NAME, len(ENTITY_MARKERS)).to(device)
    optimizer, scheduler = make_optimizer_scheduler(model, len(full_loader), n_epochs)
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(model, full_loader, optimizer, scheduler, device)
        print(f"  epoch {epoch:2d}  train_loss={loss:.4f}")
    return model


def train_with_early_stopping(
    train_ds: Dataset, val_ds: Dataset, device
) -> tuple:
    """Train a single model with early stopping on val_loss (BCE). Returns (model, best_epoch)."""
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    model        = EventRelationRoBERTa(MODEL_NAME, len(ENTITY_MARKERS)).to(device)
    optimizer, scheduler = make_optimizer_scheduler(model, len(train_loader), MAX_EPOCHS)

    best_vloss, best_state, best_epoch, patience = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        t_log, c_log, labs = predict(model, val_loader, device)
        vloss  = val_loss(t_log, c_log, labs)
        f1_avg = mean_f1(compute_metrics(t_log, c_log, labs))
        print(f"  epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={vloss:.4f}  val_F1={f1_avg:.4f}")
        if vloss < best_vloss:
            best_vloss, best_epoch, patience = vloss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  early stop at epoch {epoch}  (best epoch {best_epoch}  val_loss={best_vloss:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_epoch

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

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(ENTITY_MARKERS, special_tokens=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path("bert_modeling/checkpoints") / f"event_relation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pseudo-label mode ──────────────────────────────────────────────────────
    if args.scale_up_dir:
        scale_dir = Path(args.scale_up_dir)
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
        model, best_epoch = train_with_early_stopping(train_ds, val_ds, device)

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
        }, indent=2))

    # ── Gold-only mode (CV) ────────────────────────────────────────────────────
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
