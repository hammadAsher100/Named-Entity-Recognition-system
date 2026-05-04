# 🚀 Complete Deployment Guide

## Overview
- **Frontend**: Streamlit app (deployed on Streamlit Cloud)
- **Backend**: FastAPI server (deploy separately)
- **Models**: Pre-trained NER model files

---

## Part 1: Frontend Deployment (Streamlit Cloud)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Deploy NER project"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repository
4. Select:
   - **Repository**: `your-username/ner_project`
   - **Branch**: `main`
   - **Main file path**: `frontend/app.py`
5. Click **"Deploy"**

✅ Your Streamlit app is now live at: `https://ner-project-xxx.streamlit.app`

---

## Part 2: Backend Deployment (FastAPI)

You have two options:

### Option A: Deploy on Render (Recommended - Free tier available)

#### Step 1: Create `render.yaml`
Create a file named `render.yaml` in your project root:
```yaml
services:
  - type: web
    name: ner-backend
    env: python
    plan: free
    buildCommand: pip install --upgrade pip && pip install -r requirements.txt && python -m backend.train
    startCommand: uvicorn backend.api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.10
```

#### Step 2: Deploy on Render
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click **"New +"** → **"Web Service"**
4. Connect your repository
5. Fill in:
   - **Name**: `ner-backend`
   - **Runtime**: `Python 3.10`
   - **Build Command**: `pip install -r requirements.txt && python -m backend.train`
   - **Start Command**: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
6. Click **"Deploy"**

✅ Your API is now live at: `https://ner-backend-xxxx.onrender.com`

---

### Option B: Deploy on Railway

1. Go to [railway.app](https://railway.app)
2. Click **"New Project"** → **"Deploy from GitHub"**
3. Select your repository
4. Railway auto-detects it's a Python project
5. Set environment variables (if needed)
6. It automatically deploys!

---

### Option C: Deploy on PythonAnywhere (Easiest)

1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. Sign up (free account available)
3. Upload your files or clone from GitHub
4. Go to "Web" → "Add a new web app"
5. Select FastAPI
6. Configure WSGI file to point to `backend/api:app`
7. Reload and go live!

---

## Part 3: Connect Frontend to Backend

Once your backend is deployed, you'll have a URL like:
```
https://ner-backend-xxxx.onrender.com
```

### Update Streamlit App Settings:

1. Go to your **Streamlit Cloud** app → **Settings** (gear icon)
2. Go to **Secrets**
3. Add:
```toml
api_url = "https://ner-backend-xxxx.onrender.com"
```

4. **Reboot** the app

---

## Part 4: Share Your Live App

Your full-stack NER app is now ready! Share this link:

```
🌐 https://ner-project-xxx.streamlit.app

👤 Users can open this in their browser with NO installation needed!
```

---

## Troubleshooting

### "Cannot reach API"
- Check that your backend is deployed and running
- Verify the API URL in Streamlit secrets
- Test manually: `curl https://your-backend-url/health`

### Model Training on Deployment
- The model training happens during **first deployment** (it takes a few minutes)
- Subsequent deployments will be faster
- If it times out, increase the timeout in your deploy settings

### API Takes Time to Respond
- First request after deployment might be slow (cold start)
- Subsequent requests will be faster
- Consider upgrading from free tier if latency is important

---

## Final Architecture

```
┌─────────────────────────────────┐
│   Streamlit Cloud               │
│   (Frontend - Public)           │
│   ner-project-xxx.streamlit.app │
└────────────────┬────────────────┘
                 │ HTTPS
                 ↓
┌─────────────────────────────────┐
│   Render / Railway / etc        │
│   (Backend - Public)            │
│   ner-backend-xxxx.onrender.com │
└─────────────────────────────────┘
```

Users can now access your NER tool without any setup! 🎉
