"""
NER Model Training Script using BiLSTM-CRF
Dataset: CoNLL-2003 (loaded via datasets library)
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN     = 128
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.33
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# CoNLL-2003 label list (index → tag)
LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
PAD_TAG  = -100          # ignored by CrossEntropy
UNK      = "<UNK>"
PAD      = "<PAD>"


# ── Vocabulary ───────────────────────────────────────────────────────────────
def build_vocab(dataset_split, min_freq: int = 1):
    counter = Counter()
    for example in dataset_split:
        counter.update(t.lower() for t in example["tokens"])
    vocab = [PAD, UNK] + [w for w, c in counter.items() if c >= min_freq]
    word2id = {w: i for i, w in enumerate(vocab)}
    return vocab, word2id


# ── Dataset ──────────────────────────────────────────────────────────────────
class NERDataset(Dataset):
    def __init__(self, examples, word2id: dict, max_len: int = MAX_LEN):
        self.examples = examples
        self.word2id  = word2id
        self.max_len  = max_len

    def __len__(self):
        return len(self.examples)

    def _encode(self, tokens, labels):
        ids  = [self.word2id.get(t.lower(), self.word2id[UNK]) for t in tokens]
        ids  = ids[:self.max_len]
        lbls = labels[:self.max_len]
        pad_len = self.max_len - len(ids)
        ids  += [self.word2id[PAD]] * pad_len
        lbls += [PAD_TAG] * pad_len
        mask = [1] * (self.max_len - pad_len) + [0] * pad_len
        return ids, lbls, mask

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids, lbls, mask = self._encode(ex["tokens"], ex["ner_tags"])
        return (
            torch.tensor(ids,  dtype=torch.long),
            torch.tensor(lbls, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
        )


# ── Model ─────────────────────────────────────────────────────────────────────
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, num_tags, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        emb  = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        out  = self.dropout(out)
        return self.fc(out)          # (B, L, num_tags)


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_metrics(preds_flat, labels_flat):
    """Ignore PAD_TAG entries."""
    valid = [(p, l) for p, l in zip(preds_flat, labels_flat) if l != PAD_TAG]
    if not valid:
        return {}
    ps, ls = zip(*valid)
    tag_names = [ID2LABEL[i] for i in sorted(set(ls))]
    return classification_report(
        ls, ps,
        labels=sorted(set(ls)),
        target_names=tag_names,
        output_dict=True,
        zero_division=0,
    )


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion  = nn.CrossEntropyLoss(ignore_index=PAD_TAG)
    with torch.no_grad():
        for ids, lbls, _ in loader:
            ids, lbls = ids.to(DEVICE), lbls.to(DEVICE)
            logits = model(ids)
            loss   = criterion(logits.view(-1, logits.size(-1)), lbls.view(-1))
            total_loss += loss.item()
            preds = logits.argmax(-1).cpu().numpy().tolist()
            lbls_ = lbls.cpu().numpy().tolist()
            for p_seq, l_seq in zip(preds, lbls_):
                all_preds.extend(p_seq)
                all_labels.extend(l_seq)
    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    print(f"Using device: {DEVICE}")

    # 1. Load CoNLL-2003
    print("Loading CoNLL-2003 dataset …")
    raw = load_dataset("conll2003", trust_remote_code=True)

    # 2. Vocabulary
    vocab, word2id = build_vocab(raw["train"])
    print(f"Vocab size: {len(vocab)}")

    # 3. Datasets & loaders
    train_ds = NERDataset(raw["train"],      word2id)
    val_ds   = NERDataset(raw["validation"], word2id)
    test_ds  = NERDataset(raw["test"],       word2id)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # 4. Model
    model = BiLSTM_NER(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_tags=len(LABEL_LIST),
        dropout=DROPOUT,
        pad_idx=word2id[PAD],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, verbose=True
    )

    best_val_f1 = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    # 5. Epoch loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for ids, lbls, _ in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            ids, lbls = ids.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids)
            loss   = criterion(logits.view(-1, logits.size(-1)), lbls.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_dl)
        val_loss, val_metrics = evaluate(model, val_dl)
        val_f1 = val_metrics.get("weighted avg", {}).get("f1-score", 0)
        scheduler.step(val_loss)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
            print("  ✓ New best model saved")

    # 6. Test evaluation
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
    _, test_metrics = evaluate(model, test_dl)
    print("\n=== Test Results ===")
    for tag, scores in test_metrics.items():
        if isinstance(scores, dict):
            print(f"  {tag}: P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1-score']:.3f}")

    # 7. Save artefacts
    with open(os.path.join(MODEL_DIR, "word2id.pkl"), "wb") as f:
        pickle.dump(word2id, f)
    with open(os.path.join(MODEL_DIR, "label_info.json"), "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)
    config = {
        "vocab_size": len(vocab),
        "embed_dim":  EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_tags":   len(LABEL_LIST),
        "dropout":    DROPOUT,
        "pad_idx":    word2id[PAD],
        "max_len":    MAX_LEN,
    }
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nAll artefacts saved to models/")
    print(f"Best validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()
