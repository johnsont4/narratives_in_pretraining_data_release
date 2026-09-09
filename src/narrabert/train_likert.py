"""
Fine-tune RoBERTa-base for multi-task Likert regression over all 9 narrative dimensions
(5 agency + 4 setting).

Two modes:
  Gold only (no flag):  5-fold CV on gold annotations, then final model on all gold.
  Pseudo-labels:        90/10 split of pseudo-label data, early stopping on the 10% val
                        set, final evaluation on held-out gold annotations.

Usage:
    python src/narrabert/train_likert.py
    python src/narrabert/train_likert.py \
        --pseudo-labels outputs/scale/google_gemma-4-31B-it/<ts>/predictions.csv

Outputs  checkpoints/{timestamp}/
    model.pt           — model weights
    tokenizer/         — saved tokenizer
    config.json        — hyperparameters and summary
    test_results.csv   — metrics on gold annotations  (pseudo-label mode)
    cv_results.csv     — per-fold metrics             (gold-only mode)
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
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_annotation"))

import data  # noqa: E402  — needs the path insert above
from narrabert import training  # noqa: E402

warnings.filterwarnings("ignore", message="An input array is constant")
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME   = "roberta-base"
MAX_LEN      = 200
BATCH_SIZE   = 16
LR           = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS   = 20
PATIENCE     = 3
N_FOLDS      = 5
VAL_FRAC     = 0.1   # fraction of pseudo-labels held out for early stopping
SEED         = 42

AGENCY_DIMS  = ["focalization", "emotion", "cognition", "change_of_state", "conflict"]
SETTING_DIMS = ["concreteness", "temporal_grounding", "spatial_grounding", "sensory"]
ALL_DIMS     = AGENCY_DIMS + SETTING_DIMS

GOLD_COLS = {
    **{dim: f"agency_{dim}_gold"  for dim in AGENCY_DIMS},
    **{dim: f"setting_{dim}_gold" for dim in SETTING_DIMS},
}

# ── Data ───────────────────────────────────────────────────────────────────────

def load_data() -> tuple[list[str], np.ndarray, list[str]]:
    agency  = data.load_gold("agency")
    setting = data.load_gold("setting")[
        ["safe_instance_id"] + [GOLD_COLS[d] for d in SETTING_DIMS]
    ]
    df = agency.merge(setting, on="safe_instance_id", how="inner")
    texts      = df["sampled_text"].tolist()
    labels     = df[[GOLD_COLS[d] for d in ALL_DIMS]].values.astype(float)
    inst_ids   = df["safe_instance_id"].tolist()
    print(f"Loaded {len(df)} instances, {labels.shape[1]} dimensions")
    for dim, n in zip(ALL_DIMS, np.isnan(labels).sum(axis=0)):
        if n:
            print(f"  Warning: {n} null(s) in {dim} — will be masked in loss")
    return texts, labels, inst_ids


def load_pseudo_label_data(path: str) -> tuple[list[str], np.ndarray, list[str]]:
    # Accepts both shapes the pseudo-labels arrive in: the annotator writes
    # predictions.csv, while the published Hugging Face split is parquet.
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    pred_cols = [f"pred_{d}" for d in ALL_DIMS]
    before = len(df)
    df = df.dropna(subset=pred_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} instances with null predictions")
    texts    = df["sampled_text"].tolist()
    labels   = df[pred_cols].values.astype(float)
    # Passage-grain ids: `id` alone is a document id and repeats across the
    # passages drawn from one document.
    key = data.passage_key_cols(df)
    inst_ids = df[key].astype(str).agg("::".join, axis=1).tolist()
    print(f"Loaded {len(df)} instances, {labels.shape[1]} dimensions (pseudo-labels from {path})")
    return texts, labels, inst_ids


class NarrativeDataset(Dataset):
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

class NarrativeRoBERTa(nn.Module):
    def __init__(self, model_name: str, n_dims: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_dims)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]                           # (B, H)
        return torch.cat([h(cls) for h in self.heads], dim=1)  # (B, n_dims)

# ── Training ───────────────────────────────────────────────────────────────────

def build_model() -> "NarrativeRoBERTa":
    return NarrativeRoBERTa(MODEL_NAME, len(ALL_DIMS))


def make_score_val(device):
    """Early-stopping criterion for this model: mean MAE across the nine dimensions."""
    def score_val(model, loader) -> tuple[float, str]:
        preds, labs = predict(model, loader, device)
        val_mae = mean_mae(compute_metrics(preds, labs))
        return val_mae, f"val_MAE={val_mae:.4f}"
    return score_val


#: Hyperparameters the shared loop needs, fixed for this model.
LOOP = dict(batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            lr=LR, weight_decay=WEIGHT_DECAY)


def masked_mse(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(labels)
    return ((preds[mask] - labels[mask]) ** 2).mean()


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labs = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        all_preds.append(model(ids, mask).cpu().numpy())
        all_labs.append(batch["labels"].numpy())
    return np.vstack(all_preds), np.vstack(all_labs)

# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> list[dict]:
    preds = np.clip(preds, 1.0, 5.0)
    rows  = []
    for i, dim in enumerate(ALL_DIMS):
        valid = ~np.isnan(labels[:, i])
        p, g  = preds[valid, i], labels[valid, i]
        if len(p) < 2:
            rows.append({"dim": dim, "N": int(valid.sum()), "MAE": float("nan"),
                         "Pearson_r": float("nan"), "Precision": float("nan"),
                         "Recall": float("nan"), "F1": float("nan")})
            continue
        r, _   = pearsonr(p, g)
        p_int  = np.round(p).astype(int)
        g_int  = g.astype(int)
        rows.append({
            "dim":       dim,
            "N":         int(valid.sum()),
            "MAE":       round(float(np.abs(p - g).mean()), 3),
            "Pearson_r": round(float(r), 3),
            "Precision": round(float(precision_score(g_int, p_int, average="macro", zero_division=0)), 3),
            "Recall":    round(float(recall_score(g_int, p_int, average="macro", zero_division=0)), 3),
            "F1":        round(float(f1_score(g_int, p_int, average="macro", zero_division=0)), 3),
        })
    return rows


def mean_mae(metrics: list[dict]) -> float:
    vals = [m["MAE"] for m in metrics if not np.isnan(m["MAE"])]
    return float(np.mean(vals))

# ── CV + final training ────────────────────────────────────────────────────────

def run_cv(
    texts, labels, inst_ids, tokenizer, device
) -> tuple[pd.DataFrame, int, list[int], pd.DataFrame]:
    kf          = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_rows   = []
    best_epochs = []
    best_fold_mae   = float("inf")
    best_fold_df    = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} {'─'*50}")

        train_ds = NarrativeDataset(
            [texts[i] for i in train_idx], labels[train_idx], tokenizer, MAX_LEN
        )
        val_ds = NarrativeDataset(
            [texts[i] for i in val_idx], labels[val_idx], tokenizer, MAX_LEN
        )
        model, best_epoch = training.train_with_early_stopping(
            build_model, train_ds, val_ds, device,
            loss_fn=masked_mse, score_fn=make_score_val(device), **LOOP,
        )
        best_epochs.append(best_epoch)

        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        preds, labs = predict(model, val_loader, device)
        preds_clipped = np.clip(preds, 1.0, 5.0)
        metrics     = compute_metrics(preds_clipped, labs)
        print(f"\n  Results (best epoch {best_epoch}):")
        for m in metrics:
            print(f"    {m['dim']:25s}  MAE={m['MAE']:.3f}  r={m['Pearson_r']:.3f}")
        fold_rows.extend([{"fold": fold, **m} for m in metrics])

        fold_mean_mae = mean_mae(metrics)
        if fold_mean_mae < best_fold_mae:
            best_fold_mae = fold_mean_mae
            rows = {"instance_id": [inst_ids[i] for i in val_idx]}
            for d, dim in enumerate(ALL_DIMS):
                rows[f"gold_{dim}"]  = labs[:, d]
                rows[f"pred_{dim}"]  = preds_clipped[:, d]
            best_fold_df = pd.DataFrame(rows)

    cv_df       = pd.DataFrame(fold_rows)
    final_epoch = int(np.median(best_epochs))
    return cv_df, final_epoch, best_epochs, best_fold_df


def train_final(texts, labels, tokenizer, device, n_epochs: int) -> "NarrativeRoBERTa":
    print(f"\n── Final model — {n_epochs} epochs on all {len(texts)} instances {'─'*30}")
    return training.train_fixed_epochs(
        build_model, NarrativeDataset(texts, labels, tokenizer, MAX_LEN), device,
        loss_fn=masked_mse, n_epochs=n_epochs,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for multi-task Likert regression.")
    parser.add_argument(
        "--pseudo-labels", type=str, default=None,
        help="Path to a predictions CSV from gemma/annotate_scale.py. "
             "Trains on a 90/10 split of pseudo-labels and evaluates on gold annotations.",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = training.select_device()
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = data.REPO_ROOT / "checkpoints" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pseudo-label mode ──────────────────────────────────────────────────────
    if args.pseudo_labels:
        texts_pl, labels_pl, ids_pl = load_pseudo_label_data(args.pseudo_labels)

        # 90/10 split for early stopping
        n_total = len(texts_pl)
        n_val   = max(1, int(n_total * VAL_FRAC))
        rng     = np.random.RandomState(SEED)
        idx     = rng.permutation(n_total)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        print(f"\nSplit: {len(train_idx)} train  /  {n_val} val  (VAL_FRAC={VAL_FRAC})")

        train_ds = NarrativeDataset(
            [texts_pl[i] for i in train_idx], labels_pl[train_idx], tokenizer, MAX_LEN
        )
        val_ds = NarrativeDataset(
            [texts_pl[i] for i in val_idx], labels_pl[val_idx], tokenizer, MAX_LEN
        )

        print("\n── Training (early stopping on pseudo-label val set) " + "─" * 25)
        model, best_epoch = training.train_with_early_stopping(
            build_model, train_ds, val_ds, device,
            loss_fn=masked_mse, score_fn=make_score_val(device), **LOOP,
        )

        # ── Held-out gold evaluation ──────────────────────────────────────────
        # The 10% val split is pseudo-labels too, so it measures agreement with
        # the LLM annotator, not accuracy. The human gold set is the only
        # held-out signal here — it was excluded from the scale-up draw, so no
        # gold passage appears in the pseudo-labels.
        texts_g, labels_g, ids_g = load_data()
        gold_loader = DataLoader(
            NarrativeDataset(texts_g, labels_g, tokenizer, MAX_LEN), batch_size=BATCH_SIZE
        )
        preds_g, labs_g = predict(model, gold_loader, device)
        preds_g = np.clip(preds_g, 1.0, 5.0)
        test_metrics = compute_metrics(preds_g, labs_g)

        print("\n── Gold evaluation " + "─" * 55)
        for m in test_metrics:
            print(f"    {m['dim']:25s}  MAE={m['MAE']:.3f}  r={m['Pearson_r']:.3f}")
        print(f"    {'mean':25s}  MAE={mean_mae(test_metrics):.3f}")

        pd.DataFrame(test_metrics).to_csv(out_dir / "test_results.csv", index=False)

        pred_rows = {"instance_id": ids_g}
        for d, dim in enumerate(ALL_DIMS):
            pred_rows[f"gold_{dim}"] = labs_g[:, d]
            pred_rows[f"pred_{dim}"] = preds_g[:, d]
        pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions.csv", index=False)

        # Save
        torch.save(model.state_dict(), out_dir / "model.pt")
        tokenizer.save_pretrained(out_dir / "tokenizer")
        (out_dir / "config.json").write_text(json.dumps({
            "model_name":  MODEL_NAME,
            "max_len":     MAX_LEN,
            "dims":        ALL_DIMS,
            "data_source": args.pseudo_labels,
            "n_train":     len(train_idx),
            "n_val":       n_val,
            "val_frac":    VAL_FRAC,
            "best_epoch":  best_epoch,
            "seed":        SEED,
            "n_gold_test": len(texts_g),
            "gold_mean_mae": round(mean_mae(test_metrics), 4),
        }, indent=2))

    # ── Gold-only mode (CV) ────────────────────────────────────────────────────
    else:
        texts, labels, inst_ids = load_data()

        cv_df, final_epoch, best_epochs, best_fold_df = run_cv(
            texts, labels, inst_ids, tokenizer, device
        )

        summary     = cv_df.groupby("dim")[["MAE", "Pearson_r"]].agg(["mean", "std"]).round(3)
        overall_mae = round(float(cv_df["MAE"].mean()), 3)
        print("\n── CV Summary " + "─" * 60)
        print(summary.to_string())
        print(f"\nOverall mean MAE:    {overall_mae}")
        print(f"Best epochs / fold:  {best_epochs}  → median {final_epoch}")

        final_model = train_final(texts, labels, tokenizer, device, final_epoch)

        torch.save(final_model.state_dict(), out_dir / "model.pt")
        tokenizer.save_pretrained(out_dir / "tokenizer")
        cv_df.to_csv(out_dir / "cv_results.csv", index=False)
        best_fold_df.to_csv(out_dir / "best_fold_predictions.csv", index=False)
        (out_dir / "config.json").write_text(json.dumps({
            "model_name":   MODEL_NAME,
            "max_len":      MAX_LEN,
            "dims":         ALL_DIMS,
            "data_source":  "gold_annotations",
            "n_folds":      N_FOLDS,
            "final_epochs": final_epoch,
            "seed":         SEED,
            "cv_mean_mae":  overall_mae,
        }, indent=2))

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
