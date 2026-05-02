# 🔍 NER System — BiLSTM + FastAPI + Streamlit

A complete **Named Entity Recognition** pipeline using a **Bidirectional LSTM** model, trained on CoNLL-2003, served via **FastAPI**, and visualised with a **Streamlit** frontend.

```
ner_project/
├── backend/
│   ├── __init__.py
│   ├── train.py          ← BiLSTM model + training loop
│   └── api.py            ← FastAPI prediction server
├── frontend/
│   └── app.py            ← Streamlit UI
├── notebooks/
│   └── exploration.ipynb ← Dataset exploration & post-training analysis
├── models/               ← Created automatically after training
│   ├── best_model.pt
│   ├── word2id.pkl
│   ├── label_info.json
│   ├── model_config.json
│   └── training_history.json
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Local — No Docker)

### Step 1 — Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/ner-project.git
cd ner-project
pip install -r requirements.txt
```

> **Python 3.9 – 3.11 recommended**

---

### Step 2 — Train the model

```bash
python -m backend.train
```

This will:
- Download the **CoNLL-2003** dataset automatically (via HuggingFace `datasets`)
- Train a **BiLSTM** model for 10 epochs
- Save the best checkpoint to `models/best_model.pt`
- Print test-set F1 scores when done

Training takes **~5–15 minutes** on CPU, ~2 minutes on GPU.

---

### Step 3 — Start the FastAPI backend

Open a **new terminal**:

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Barack Obama was born in Hawaii."}'
```

Interactive docs: **http://localhost:8000/docs**

---

### Step 4 — Start the Streamlit frontend

Open another **new terminal**:

```bash
streamlit run frontend/app.py
```

Open your browser at **http://localhost:8501**

---

## 🐳 Docker Deployment

### Build & run everything with one command

```bash
docker compose up --build
```

| Service   | URL                        |
|-----------|----------------------------|
| FastAPI   | http://localhost:8000      |
| Streamlit | http://localhost:8501      |
| API Docs  | http://localhost:8000/docs |

> **Note:** Train the model locally first (`python -m backend.train`) so the `models/` folder is populated before starting Docker. The models folder is mounted as a volume.

### Build individual images

```bash
# API only
docker build --target api -t ner-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models ner-api

# Frontend only
docker build --target frontend -t ner-frontend .
docker run -p 8501:8501 -e API_URL=http://YOUR_API_HOST:8000 ner-frontend
```

### Push to Docker Hub

```bash
docker tag ner-api YOUR_DOCKERHUB_USERNAME/ner-api:latest
docker push YOUR_DOCKERHUB_USERNAME/ner-api:latest

docker tag ner-frontend YOUR_DOCKERHUB_USERNAME/ner-frontend:latest
docker push YOUR_DOCKERHUB_USERNAME/ner-frontend:latest
```

---

## ☁️ Streamlit Community Cloud Deployment

If Docker is not available, deploy the frontend on **Streamlit Community Cloud** (free):

1. Push your repo to GitHub (make sure `models/` files are included — remove `models/*.pt` from `.gitignore` after training)
2. Go to **https://share.streamlit.io** → New app
3. Set:
   - **Repository**: `your-username/ner-project`
   - **Main file path**: `frontend/app.py`
4. Add a secret in **Advanced settings**:
   ```
   API_URL = "https://your-fastapi-server.com"
   ```
5. Click **Deploy**

> For the API, deploy FastAPI on **Render** (free tier):
> - Build command: `pip install -r requirements.txt`
> - Start command: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`

---

## 📡 API Reference

### `POST /predict`

**Request:**
```json
{ "text": "Apple was founded by Steve Jobs in California." }
```

**Response:**
```json
{
  "raw_text": "Apple was founded by Steve Jobs in California.",
  "tokens": [
    { "token": "Apple", "tag": "B-ORG" },
    { "token": "was",   "tag": "O" },
    { "token": "Steve", "tag": "B-PER" },
    { "token": "Jobs",  "tag": "I-PER" },
    ...
  ],
  "entities": [
    { "text": "Apple",      "label": "ORG", "start": 0,  "end": 5  },
    { "text": "Steve Jobs", "label": "PER", "start": 21, "end": 31 },
    { "text": "California", "label": "LOC", "start": 35, "end": 45 }
  ]
}
```

### `GET /health`
```json
{ "model_loaded": true, "device": "cpu" }
```

### `GET /labels`
```json
{ "labels": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"] }
```

---

## 🧠 Model Architecture

| Component     | Detail                                 |
|---------------|----------------------------------------|
| Embedding     | Learnable, 128-dim                     |
| Encoder       | 2-layer **BiLSTM**, 256 hidden units   |
| Dropout       | 0.33 (after embedding & LSTM output)   |
| Classifier    | Linear → 9 tags (BIO scheme)           |
| Loss          | Cross-Entropy (PAD tokens ignored)     |
| Optimiser     | Adam, LR=1e-3 with ReduceLROnPlateau   |

**Entity types (CoNLL-2003 BIO scheme):**

| Label  | Meaning      |
|--------|--------------|
| PER    | Person       |
| ORG    | Organisation |
| LOC    | Location     |
| MISC   | Miscellaneous|

---

## 📊 Expected Performance

After 10 epochs on CoNLL-2003:

| Entity | Precision | Recall | F1    |
|--------|-----------|--------|-------|
| PER    | ~0.88     | ~0.90  | ~0.89 |
| ORG    | ~0.82     | ~0.80  | ~0.81 |
| LOC    | ~0.87     | ~0.88  | ~0.88 |
| MISC   | ~0.76     | ~0.74  | ~0.75 |

---

## 📁 GitHub Setup

```bash
git init
git add .
git commit -m "Initial commit: NER system with BiLSTM + FastAPI + Streamlit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ner-project.git
git push -u origin main
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Missing artefact` | Run `python -m backend.train` first |
| `Cannot connect to API` | Make sure FastAPI is running on port 8000 |
| CUDA out of memory | Reduce `BATCH_SIZE` in `backend/train.py` |
| Slow training | Set `EPOCHS = 5` for a faster run |
| Dataset download fails | Check internet connection; HuggingFace downloads CoNLL-2003 automatically |
