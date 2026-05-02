# CardioScreen — Deployment Guide

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at http://localhost:8501

On first run it trains the model (~2 min for 630K rows). After that it loads instantly from cache.

## Deploy to Streamlit Cloud (free, public link in minutes)

1. Create a GitHub repository (public or private)
2. Upload these files:
   - app.py
   - requirements.txt
   - heart_disease.csv
3. Go to https://share.streamlit.io
4. Connect your GitHub repo
5. Set main file path: `app.py`
6. Click Deploy — you get a public URL like:
   `https://your-app-name.streamlit.app`

## Deploy to Hugging Face Spaces (alternative)

1. Go to https://huggingface.co/spaces
2. Create new Space → Streamlit
3. Upload app.py, requirements.txt, heart_disease.csv
4. Done — public URL automatically generated

## Include the link in your deck

Once deployed, paste the URL into slide 3 as the "MVP Link" the brief asks for.

## Files needed
- app.py — the Streamlit application
- requirements.txt — dependencies
- heart_disease.csv — training dataset (630,000 patients)
