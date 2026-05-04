# Streamlit Deployment Guide

## Local Development

### Run Streamlit App
```bash
streamlit run frontend/app.py
```
Open your browser to `http://localhost:8501`

---

## Deployment on Streamlit Cloud (FREE)

### Prerequisites
- GitHub account
- Your project pushed to GitHub (public or private)

### Step 1: Push Code to GitHub
```bash
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click **"New app"**
3. Select:
   - **Repository**: `your-username/ner_project`
   - **Branch**: `main`
   - **Main file path**: `frontend/app.py`
4. Click **"Deploy!"**

### Step 3: Configure Backend URL
After deployment, you'll get a Streamlit Cloud URL like:
```
https://ner-project-abc123.streamlit.app
```

For the Streamlit frontend to connect to your FastAPI backend:

**Option A: Self-hosted FastAPI**
- Deploy your FastAPI backend separately (Render, Railway, Heroku, etc.)
- In Streamlit app settings, set environment variable:
  ```
  API_URL=https://your-api-backend.com
  ```

**Option B: Run both on Streamlit + FastAPI backend**
- You'll need to host the FastAPI separately since Streamlit Cloud doesn't support multiple services

### Step 4: Share Your Link
Your Streamlit app will be publicly available at:
```
https://ner-project-<random-id>.streamlit.app
```

People can access it by simply opening that URL in their browser - **no installation required!**

---

## Share Instructions with Users

Once deployed, share this link with people:
```
👉 https://ner-project-<random-id>.streamlit.app

Just open the link in your browser and start using the NER tool!
```

---

## Environment Variables (for deployment)

Create `.streamlit/secrets.toml` locally (NOT committed to GitHub):
```toml
api_url = "https://your-api-backend.com"
```

In Streamlit Cloud dashboard → App settings → Secrets:
```
api_url = "https://your-api-backend.com"
```

Then access in code:
```python
import streamlit as st
API_URL = st.secrets.get("api_url", "http://localhost:8000")
```

---

## Popular Backend Hosting Options

| Platform | Cost | Supports | Notes |
|----------|------|----------|-------|
| [Railway](https://railway.app) | $5-20/month | FastAPI, Python | Easy Flask/FastAPI deployment |
| [Render](https://render.com) | Free-$20/month | FastAPI, Python | Free tier available |
| [Fly.io](https://fly.io) | $5-20/month | Docker, Python | Global deployment |
| [PythonAnywhere](https://pythonanywhere.com) | $5-50/month | Python | Simple Python hosting |

---

## Troubleshooting

### API Connection Error
- Make sure backend is running and URL is correct
- Check firewall/CORS settings
- Test manually: `curl https://your-api-backend.com/health`

### Slow App
- Streamlit Cloud might be slower; consider upgrading
- Move heavy processing to backend API

### GitHub Connection Issues
- Check repository is public or you have access
- Verify `frontend/app.py` path is correct
