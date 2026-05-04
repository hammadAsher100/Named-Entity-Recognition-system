# 🌐 Live Deployment URLs

## Your NER System is Live! 🚀

### Frontend (Streamlit App)
```
https://ner-project-xxx.streamlit.app
```
👉 **Share this link** with anyone to use your NER tool!

### Backend (FastAPI)
```
https://named-entity-recognition-system.onrender.com
```
Status: ✅ Active & Running

---

## Quick Links

| Component | URL | Status |
|-----------|-----|--------|
| **Frontend** | https://ner-project-xxx.streamlit.app | 🟢 Live |
| **Backend API** | https://named-entity-recognition-system.onrender.com | 🟢 Live |
| **API Health Check** | https://named-entity-recognition-system.onrender.com/health | 🟢 Live |
| **API Docs** | https://named-entity-recognition-system.onrender.com/docs | 📖 Available |

---

## Testing

### Test the Frontend
1. Open: **https://ner-project-xxx.streamlit.app**
2. Click a sample text button (e.g., "Tech Companies")
3. Click "🔍 Analyse"
4. See entities highlighted!

### Test the Backend Directly
```bash
# Health check
curl https://named-entity-recognition-system.onrender.com/health

# Make prediction
curl -X POST https://named-entity-recognition-system.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Steve Jobs founded Apple in Cupertino."}'
```

### Test Debug Mode
1. Open the Streamlit app
2. Open sidebar (click hamburger menu)
3. Check **🔧 Debug Mode**
4. Click **API Configuration**
5. Click **🔌 Test API**

Should show:
```json
{
  "model_loaded": true,
  "device": "cpu"
}
```

---

## Sharing Instructions

Send your friends this simple message:

> 🔍 **Check out my NER AI tool!**
> 
> Open: **https://ner-project-xxx.streamlit.app**
> 
> No installation needed - just paste text and see entities detected by AI!

---

## Environment Variables

If you need to override the API URL:

### Local Development
```bash
export API_URL=https://named-entity-recognition-system.onrender.com
streamlit run frontend/app.py
```

### Streamlit Cloud Secrets
Go to: App Settings → Secrets
```toml
api_url = "https://named-entity-recognition-system.onrender.com"
```

---

## API Endpoints

### `/health` (GET)
Check if API and model are ready
```bash
curl https://named-entity-recognition-system.onrender.com/health
```

### `/predict` (POST)
Identify entities in text
```bash
curl -X POST https://named-entity-recognition-system.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple was founded by Steve Jobs in California"
  }'
```

**Response**:
```json
{
  "tokens": [
    {"token": "Apple", "tag": "B-ORG"},
    {"token": "founded", "tag": "O"},
    ...
  ],
  "entities": [
    {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
    {"text": "Steve Jobs", "label": "PER", "start": 24, "end": 34},
    {"text": "California", "label": "LOC", "start": 38, "end": 48}
  ],
  "raw_text": "Apple was founded by Steve Jobs in California"
}
```

---

## Monitoring

### Check Backend Status
- Go to: **Render Dashboard** → Your App
- Should show: "Deployed" ✅

### Check Frontend Status  
- Go to: **Streamlit Cloud** → Your App
- Should show: "Deployed" ✅

### If Something is Down
1. Check Render dashboard for errors
2. Use Debug Mode to test API URL
3. Check that training completed (model files exist)

---

## Next Steps

✅ Frontend is live  
✅ Backend is live  
✅ Models are trained  
✅ Ready to share!

**Send link to friends/team and get feedback!**
