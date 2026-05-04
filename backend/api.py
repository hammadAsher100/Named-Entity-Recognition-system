"""
FastAPI backend — Named Entity Recognition API
Run: uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import json
import pickle
import re
from typing import List

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# ── Model definition (must match train.py) ────────────────────────────────────
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
        self.fc = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        return self.fc(out)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NER API",
    description="Named Entity Recognition using BiLSTM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (loaded once at startup) ─────────────────────────────────────
_model   = None
_word2id = None
_id2label = None
_config  = None
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global _model, _word2id, _id2label, _config

    cfg_path   = os.path.join(MODEL_DIR, "model_config.json")
    w2id_path  = os.path.join(MODEL_DIR, "word2id.pkl")
    lbl_path   = os.path.join(MODEL_DIR, "label_info.json")
    model_path = os.path.join(MODEL_DIR, "best_model.pt")

    for p in [cfg_path, w2id_path, lbl_path, model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing artefact: {p}. "
                "Please run `python -m backend.train` first."
            )

    with open(cfg_path)  as f: _config   = json.load(f)
    with open(lbl_path)  as f: lbl_info  = json.load(f)
    with open(w2id_path, "rb") as f: _word2id = pickle.load(f)

    _id2label = {int(k): v for k, v in lbl_info["id2label"].items()}

    # Remove max_len from config as it's not a model parameter
    model_config = {k: v for k, v in _config.items() if k != "max_len"}
    _model = BiLSTM_NER(**model_config).to(DEVICE)
    _model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    _model.eval()
    print("✓ Model loaded.")


@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"WARNING: {e}")


# ── Tokeniser (simple whitespace + punctuation split) ─────────────────────────
def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(tokens: List[str]) -> List[str]:
    if _model is None:
        raise RuntimeError("Model not loaded. Run training first.")

    max_len = _config["max_len"]
    unk_id  = _word2id.get("<UNK>", 1)
    pad_id  = _config["pad_idx"]

    ids = [_word2id.get(t.lower(), unk_id) for t in tokens]
    ids = ids[:max_len]
    pad = [pad_id] * (max_len - len(ids))
    ids_tensor = torch.tensor([ids + pad], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = _model(ids_tensor)                # (1, L, num_tags)
        preds  = logits.argmax(-1)[0].cpu().tolist()

    return [_id2label.get(p, "O") for p in preds[:len(tokens)]]


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str


class EntitySpan(BaseModel):
    text:  str
    label: str
    start: int
    end:   int


class TokenTag(BaseModel):
    token: str
    tag:   str


class PredictResponse(BaseModel):
    tokens:   List[TokenTag]
    entities: List[EntitySpan]
    raw_text: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "NER API is running", "status": "ok"}


@app.get("/health")
def health():
    return {"model_loaded": _model is not None, "device": str(DEVICE)}


@app.post("/predict", response_model=PredictResponse)
def predict_entities(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    tokens = simple_tokenize(req.text)
    if not tokens:
        raise HTTPException(status_code=400, detail="No tokens found in text.")

    tags = predict(tokens)

    # Build token-tag list
    token_tags = [TokenTag(token=t, tag=tg) for t, tg in zip(tokens, tags)]

    # Build entity spans using BIO scheme
    entities: List[EntitySpan] = []
    i = 0
    text_lower = req.text
    search_pos  = 0

    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):
            entity_type  = tag[2:]
            entity_tokens = [tokens[i]]
            j = i + 1
            while j < len(tokens) and tags[j] == f"I-{entity_type}":
                entity_tokens.append(tokens[j])
                j += 1
            entity_text = " ".join(entity_tokens)

            # Find position in original text (approximate)
            idx = req.text.lower().find(entity_tokens[0].lower(), search_pos)
            start = idx if idx != -1 else search_pos
            end   = start + len(entity_text)

            entities.append(EntitySpan(
                text=entity_text,
                label=entity_type,
                start=start,
                end=end,
            ))
            i = j
        else:
            i += 1

    return PredictResponse(
        tokens=token_tags,
        entities=entities,
        raw_text=req.text,
    )


@app.get("/labels")
def get_labels():
    if _id2label is None:
        return {"labels": []}
    return {"labels": list(set(_id2label.values()))}
