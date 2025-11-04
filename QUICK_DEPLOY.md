# Quick Deployment Checklist

## 📦 Pre-Deployment

```bash
# 1. Make sure everything is committed
cd /Users/tzhang/projects/A2I2
git add .
git commit -m "Ready for deployment"
git push origin main
```

## 🔧 Backend (Render)

1. **Sign up:** [render.com](https://render.com/) (use GitHub)
2. **New Web Service** → Connect GitHub repo
3. **Settings:**
   - Root Directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
4. **Environment Variables:**
   - `OPENAI_API_KEY` = your_api_key
5. **Deploy** → Copy URL (e.g., `https://your-app.onrender.com`)

## 🎨 Frontend (Netlify)

1. **Update** `frontend/config.js`:
   ```javascript
   : 'https://your-app.onrender.com';  // ⬅️ Paste your Render URL here
   ```

2. **Commit:**
   ```bash
   git add frontend/config.js
   git commit -m "Update backend URL"
   git push
   ```

3. **Sign up:** [netlify.com](https://www.netlify.com/) (use GitHub)
4. **New Site** → Import from GitHub
5. **Settings:**
   - Base directory: `frontend`
   - Publish directory: `.`
6. **Deploy** → Get your URL!

## ✅ Test

Visit your Netlify URL and try chatting with a character!

## ⚠️ Common Issues

**Backend 502 Error:** Wait 1-2 minutes, it's starting up

**CORS Error:** Add your Netlify URL to `backend/server.py`:
```python
allow_origins=["https://your-site.netlify.app"]
```

**Cold Start:** Free tier sleeps after 15 min. First request takes 30-60 sec.

## 🎉 Done!

Share your link: `https://your-site.netlify.app`

