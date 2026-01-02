# Elmos Yar - University Analysis Dashboard

## Overview
This project is a data mining dashboard and recommender system for university professors.

## Structure
- `data/`: Raw and processed data.
- `notebooks/`: Experiments.
- `src/`: Core logic.
- `app/`: Streamlit dashboard.


## How to Run the AI Model (Phase 3)
To run the Sentiment Analysis (ParsBERT), you need the model files.
The code is designed to **automatically download** them the first time you run it.

**Option A (Automatic):**
1. Just run the `04_sentiment_analysis.ipynb` notebook.
2. The script will automatically download the model (~450MB) from HuggingFace to `models/parsbert_sentiment`.

**Option B (Manual - If you have slow internet):**
1. Download the files manually from [HuggingFace HooshvareLab](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-digikala/tree/main).
2. Create a folder: `models/parsbert_sentiment`.
3. Place `pytorch_model.bin`, `config.json`, and `vocab.txt` inside.