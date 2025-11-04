# Deployment Guide - Emergency Response Chatbot

This guide will help you deploy your chatbot publicly using free hosting services.

## 🏗️ Architecture

Your app has two parts:
- **Frontend** (HTML/CSS/JS) → Deploy to **Netlify** (Free)
- **Backend** (FastAPI Python) → Deploy to **Render** (Free)

---

## Part 1: Deploy Backend to Render (Free)

### Step 1: Prepare Your Repository

1. Make sure your code is pushed to GitHub:
```bash
cd /Users/tzhang/projects/A2I2
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Sign Up for Render

1. Go to [render.com](https://render.com/)
2. Sign up with your GitHub account (free)

### Step 3: Create a New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Select the `A2I2` repository
4. Configure the service:

   **Settings:**
   - **Name:** `emergency-chatbot-backend` (or your choice)
   - **Region:** Select closest to you
   - **Branch:** `main`
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Plan:** `Free`

5. Click **"Advanced"** and add Environment Variables:
   - **Key:** `OPENAI_API_KEY`
   - **Value:** Your OpenAI API key (from your `.env` file)

6. Click **"Create Web Service"**

### Step 4: Wait for Deployment

- Render will build and deploy your backend
- This takes 5-10 minutes for the first deployment
- You'll get a URL like: `https://emergency-chatbot-backend.onrender.com`

⚠️ **Important:** Copy this URL - you'll need it for the frontend!

### Step 5: Test Your Backend

Once deployed, test it:
```bash
curl https://your-backend-app.onrender.com/
```

You should see: `{"status":"ok","message":"Emergency Response Chatbot Backend is running"}`

---

## Part 2: Deploy Frontend to Netlify (Free)

### Step 1: Update Frontend Config

1. Open `/Users/tzhang/projects/A2I2/frontend/config.js`
2. Replace the URL with your Render backend URL:

```javascript
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8001'  // Local development
    : 'https://emergency-chatbot-backend.onrender.com';  // ⬅️ UPDATE THIS
```

3. Commit the change:
```bash
git add frontend/config.js
git commit -m "Update backend URL for production"
git push origin main
```

### Step 2: Sign Up for Netlify

1. Go to [netlify.com](https://www.netlify.com/)
2. Sign up with your GitHub account (free)

### Step 3: Deploy to Netlify

**Option A: Drag and Drop (Easiest)**

1. In Netlify dashboard, click **"Add new site"** → **"Deploy manually"**
2. Drag the `/Users/tzhang/projects/A2I2/frontend` folder into the upload area
3. Wait for deployment (takes 1-2 minutes)
4. You'll get a URL like: `https://random-name-12345.netlify.app`

**Option B: Connect to GitHub (Recommended)**

1. Click **"Add new site"** → **"Import an existing project"**
2. Choose **GitHub**
3. Select your `A2I2` repository
4. Configure:
   - **Base directory:** `frontend`
   - **Build command:** (leave empty)
   - **Publish directory:** `.` (dot)
5. Click **"Deploy site"**

### Step 4: Custom Domain (Optional)

1. In Netlify, go to **Site settings** → **Domain management**
2. Click **"Add custom domain"**
3. You can use a free `.netlify.app` subdomain or your own domain

---

## 🔧 Configuration Summary

### Files Created for Deployment:

1. **`frontend/netlify.toml`** - Netlify configuration
2. **`frontend/config.js`** - API URL configuration
3. **`backend/render.yaml`** - Render configuration (optional)

### URLs After Deployment:

- **Backend:** `https://your-backend-app.onrender.com`
- **Frontend:** `https://your-site-name.netlify.app`

---

## 🔒 Security Considerations

### 1. CORS Configuration

Update your backend `server.py` CORS settings:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # Local development
        "https://your-site-name.netlify.app",  # ⬅️ Add your Netlify URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. Environment Variables

- **Never commit `.env` files** to GitHub
- Always use environment variables in Render for secrets
- OpenAI API key should only be in Render environment variables

### 3. API Key Protection

Your OpenAI API key is safe because:
- It's stored on the backend (Render)
- Frontend never sees it
- Users can't access it from the browser

---

## 💰 Free Tier Limitations

### Render Free Tier:
- ✅ 750 hours/month (enough for a demo)
- ⚠️ Spins down after 15 minutes of inactivity
- ⚠️ Cold start takes 30-60 seconds
- ✅ Automatic HTTPS
- ✅ Automatic deploys from GitHub

### Netlify Free Tier:
- ✅ 100 GB bandwidth/month
- ✅ Always online (no spin down)
- ✅ Automatic HTTPS
- ✅ Automatic deploys from GitHub
- ✅ Custom domains

---

## 🚀 Deployment Checklist

- [ ] Push code to GitHub
- [ ] Deploy backend to Render
- [ ] Add OPENAI_API_KEY to Render environment
- [ ] Copy backend URL
- [ ] Update `frontend/config.js` with backend URL
- [ ] Push updated config to GitHub
- [ ] Deploy frontend to Netlify
- [ ] Test the live app
- [ ] Update CORS in backend if needed

---

## 🐛 Troubleshooting

### Backend Issues:

**"Application failed to start"**
- Check Render logs
- Verify `requirements.txt` has all dependencies
- Make sure `OPENAI_API_KEY` is set in environment

**"502 Bad Gateway"**
- Backend is still starting (wait 1-2 minutes)
- Check Render logs for errors

### Frontend Issues:

**"Failed to fetch" or CORS errors**
- Update CORS settings in `backend/server.py`
- Make sure backend URL in `config.js` is correct
- Check if backend is running

**"API key not found"**
- Backend environment variable missing
- Check Render environment variables

### Cold Start (Render Free Tier):

The backend spins down after 15 minutes of inactivity. First request will take 30-60 seconds.

**Solutions:**
1. Use a ping service to keep it alive (e.g., UptimeRobot)
2. Upgrade to paid plan ($7/month) for always-on
3. Accept the cold start delay for demo purposes

---

## 🔄 Continuous Deployment

Both Netlify and Render support automatic deploys:

1. **Make changes** to your code
2. **Commit and push** to GitHub
3. **Automatic deployment** triggers
4. **Live site updates** in 2-5 minutes

---

## 💡 Alternative Hosting Options

### Backend Alternatives:
- **Railway** - Similar to Render, free tier available
- **Fly.io** - Free tier with better performance
- **Heroku** - No longer has free tier (paid only)
- **PythonAnywhere** - Free tier for Python apps

### Frontend Alternatives:
- **Vercel** - Similar to Netlify
- **GitHub Pages** - Free static hosting
- **Cloudflare Pages** - Free with excellent performance

---

## 📞 Support Resources

- **Render Docs:** https://render.com/docs
- **Netlify Docs:** https://docs.netlify.com/
- **OpenAI API:** https://platform.openai.com/docs

---

## 🎉 Next Steps After Deployment

1. **Share your URL** with others
2. **Monitor usage** in OpenAI dashboard
3. **Set up analytics** (Google Analytics, etc.)
4. **Add custom domain** for professional look
5. **Monitor costs** to stay within free tier

---

## ⚠️ Important Notes

1. **OpenAI Costs:** Even on free hosting, OpenAI API calls cost money. Monitor your usage at https://platform.openai.com/usage

2. **Free Tier Limits:** If you get significant traffic, you may need to upgrade hosting plans

3. **Cold Starts:** Free tier backend sleeps after inactivity. First user each day may wait 30-60 seconds

4. **Data Persistence:** Free tiers don't include databases. Conversations aren't saved permanently.

---

Good luck with your deployment! 🚀

