# 🍎 FoodSaver - AI Food Management App

Flask web application with AI-powered food recognition and smart inventory management.

## 🚀 Deploy to Render

1. **Connect Repository to Render**
2. **Set Environment Variables:**
   ```
   SECRET_KEY=your-secret-key
   HUGGINGFACE_API_KEY=your-hf-api-key
   FLASK_ENV=production
   PORT=10000
   ```
3. **Deploy** - Render auto-detects Procfile

## ✨ Features
- 🤖 AI Food Recognition with Hugging Face Vision API
- � Smart Learning Chatbot
- 📊 Food Inventory & Expiry Tracking  
- 📱 Mobile-Responsive Design

## 🛠️ Local Development
```bash
pip install -r requirements.txt
export SECRET_KEY="dev-key"
export HUGGINGFACE_API_KEY="your-api-key"
python app.py
```

## � Tech Stack
Flask • SQLite • Hugging Face API • Gunicorn
