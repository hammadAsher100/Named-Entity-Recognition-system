# 🧪 Testing Guide

## Sample Test Texts (Built-in)

The app now includes 6 sample texts organized by category:

### 1. **Tech Companies**
```
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California. Today, Tim Cook serves as CEO and the company is headquartered in Cupertino.
```
**Expected entities**: Steve Jobs (PER), Steve Wozniak (PER), Ronald Wayne (PER), Apple Inc. (ORG), Cupertino (LOC), Tim Cook (PER), California (LOC)

---

### 2. **News Article**
```
Elon Musk founded Tesla in 2003 and SpaceX in 2002. Both companies are headquartered in California. Musk is also the founder of The Boring Company.
```
**Expected entities**: Elon Musk (PER), Tesla (ORG), SpaceX (ORG), California (LOC), The Boring Company (ORG)

---

### 3. **Geography**
```
The United Nations headquarters is located in New York City. Paris is the capital of France. Mount Everest is in the Himalayas between Nepal and Tibet.
```
**Expected entities**: United Nations (ORG), New York City (LOC), Paris (LOC), France (LOC), Mount Everest (LOC), Himalayas (LOC), Nepal (LOC), Tibet (LOC)

---

### 4. **Politics**
```
Barack Obama served as the 44th President of the United States from 2009 to 2017. He was born in Honolulu, Hawaii. Michelle Obama was the First Lady.
```
**Expected entities**: Barack Obama (PER), United States (LOC), Honolulu (LOC), Hawaii (LOC), Michelle Obama (PER)

---

### 5. **Business**
```
Microsoft acquired LinkedIn for $26.2 billion in 2016. Bill Gates founded Microsoft in 1975 in Albuquerque, New Mexico. Satya Nadella is the current CEO.
```
**Expected entities**: Microsoft (ORG), LinkedIn (ORG), Bill Gates (PER), Albuquerque (LOC), New Mexico (LOC), Satya Nadella (PER)

---

### 6. **Entertainment**
```
Tom Hanks starred in Forrest Gump directed by Robert Zemeckis. The movie was filmed in various locations including Savannah, Georgia.
```
**Expected entities**: Tom Hanks (PER), Forrest Gump (MISC), Robert Zemeckis (PER), Savannah (LOC), Georgia (LOC)

---

## How to Test

### Local Testing:

1. **Start FastAPI backend**:
   ```bash
   uvicorn backend.api:app --reload
   ```

2. **Start Streamlit app** (in another terminal):
   ```bash
   streamlit run frontend/app.py
   ```

3. **Test using samples**:
   - Click any sample button to load the text
   - Click **🔍 Analyse**
   - Check if entities are correctly identified

### Production Testing:

1. **Enable Debug Mode** (bottom of sidebar):
   - Check the "🔧 Debug Mode" checkbox
   - Click "API Configuration" expander
   - Paste your Render/Railway API URL
   - Click **🔌 Test API** button

2. **Expected result**:
   ```json
   {
     "model_loaded": true,
     "device": "cpu"
   }
   ```

3. **If API is down**, you'll see:
   - ❌ Connection refused
   - ⏱️ Timeout
   - 🚫 API returned an error

---

## Troubleshooting

### ✗ "Cannot reach API"

**Step 1**: Check if backend is running
```bash
# Local: Should see "Uvicorn running on http://0.0.0.0:8000"
# Render: Check app dashboard for "Deployed" status

# Test manually:
curl http://localhost:8000/health
# or
curl https://your-render-api-url.onrender.com/health
```

**Step 2**: Verify API URL in Debug Mode
- Use the override field to test different URLs
- Try: `http://localhost:8000` (if local)
- Try: `https://ner-backend-xxxx.onrender.com` (if deployed)

**Step 3**: Check Streamlit Secrets (for production)
Go to Streamlit Cloud → App settings → Secrets, verify:
```toml
api_url = "https://your-backend-url.onrender.com"
```

---

## Performance Expectations

| Test Type | Expected Time |
|-----------|---|
| First request (cold start) | 5-10 seconds |
| Subsequent requests | 1-3 seconds |
| Short text (< 50 words) | 1-2 seconds |
| Long text (> 200 words) | 2-5 seconds |

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "API Offline" warning | Check backend is running with `uvicorn` |
| Timeout on first request | Normal - first request is slower. Try again. |
| Model not found error | Run `python -m backend.train` to train the model |
| 404 error at /predict | Check that backend is on latest version |
| CORS errors | Backend CORS should be enabled (already is in `api.py`) |

---

## Sample Queries for Quick Testing

Copy & paste these to test quickly:

**Short (Easy)**:
```
Bill Gates founded Microsoft.
```

**Medium (Normal)**:
```
Steve Jobs was the CEO of Apple and founded Pixar in Emeryville, California.
```

**Long (Advanced)**:
```
The International Olympic Committee is headquartered in Lausanne, Switzerland. The 2024 Summer Olympics were hosted by Paris, France. Athletes from around the world competed in various sports. Michael Phelps, the legendary swimmer from the United States, has won 28 Olympic medals.
```

---

## Next Steps

1. ✅ Test with sample texts
2. ✅ Verify API connection in Debug Mode
3. ✅ Try custom text input
4. ✅ Share link with others!

**Your public link**: `https://ner-project-xxx.streamlit.app`
