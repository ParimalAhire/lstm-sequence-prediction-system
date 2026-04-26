# Lab Assignment 5: LSTM-Based AI Agent for Sequence Prediction

## Group Members
- **Mohit Patil** (202301040272)
- **Parimal Ahire** (202301040067)
- **Rajveersinh Kher** (202301040233)
- **Atharva Suryawanshi** (202301040283)

---

## Project Overview
This project implements an **LSTM-based Next Word Prediction System** as part of the Deep Learning Lab. The system is trained on the Simple English Wikipedia dataset and deployed using **FastAPI** to create an industry-relevant AI agent.

### Key Features
- **NLP Preprocessing Pipeline** — Lowercasing, special character removal, n-gram sequence generation, pre-padding
- **LSTM Sequence Learning** — Captures long-term dependencies in text for context-aware predictions
- **Memory-Efficient Training** — Uses sparse categorical crossentropy (avoids one-hot encoding RAM overhead)
- **Early Stopping + Checkpointing** — Saves best model weights, protects against Colab disconnects
- **FastAPI Deployment** — Two endpoints: `/predict` (next word) and `/generate` (sentence continuation)

---

## Project Structure

```
DeepLearning_Lab5/
│
├── LSTM_Text_Prediction.ipynb   # Main training notebook (run on Google Colab)
├── app.py                       # FastAPI deployment server
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render cloud deployment config
├── README.md                    # This file
│
├── dataset/                     # PUT YOUR DATASET FILE HERE
│   └── AllCombined.txt          # Simple English Wikipedia plain text
│
└── model/                       # Generated after training (do not edit)
    ├── lstm_model.keras         # Trained LSTM model
    ├── tokenizer.pkl            # Fitted Keras tokenizer
    └── max_seq_len.pkl          # Max sequence length (required by API)
```

---

## Dataset

**Dataset:** Plain Text Wikipedia (Simple English)
**Source:** https://www.kaggle.com/datasets/josephrmartinez/simple-english-wikipedia
**File to use:** AllCombined.txt
**Place it in:** dataset/AllCombined.txt inside this project folder before zipping

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| API Framework | FastAPI + Uvicorn |
| Data Analysis | NumPy, Matplotlib |
| Notebook | Jupyter / Google Colab |
| Deployment | Render (cloud) |

---

## Step 1 — Train the Model on Google Colab

1. Place AllCombined.txt inside the dataset/ folder
2. Zip the entire project folder → DeepLearning_Lab5.zip
3. Upload the zip to Google Drive
4. Open LSTM_Text_Prediction.ipynb in Google Colab
5. Run all cells — it will:
   - Mount your Drive and extract the zip automatically
   - Clean and preprocess the text
   - Train the LSTM model (up to 80 epochs with early stopping)
   - Save model/lstm_model.keras, model/tokenizer.pkl, model/max_seq_len.pkl
6. Download the 3 saved model files from Colab

---

## Step 2 — Run the API Locally

Place the downloaded model/ folder in the project root, then:

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

API will be live at http://localhost:8000

---

## Step 3 — Deploy Publicly on Render

1. Push the project (with model/ folder) to a GitHub repo
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects render.yaml and configures everything
5. Click Deploy — you will get a public URL like:
   https://lstm-next-word-api.onrender.com

---

## API Documentation

### Home
- URL: GET /
- Response: Status message and instructions

### Predict Next Word
- URL: POST /predict
- Request:  { "text": "the world is" }
- Response: { "input": "the world is", "predicted_next_word": "known" }

### Generate Sentence
- URL: POST /generate
- Request:  { "text": "the world is", "num_words": 4 }
- Response: { "input": "the world is", "generated_text": "the world is known as a" }

### Interactive Docs
- Swagger UI: http://localhost:8000/docs
- Redoc:      http://localhost:8000/redoc

---

## LSTM Mathematical Model Summary

| Gate | Equation | Purpose |
|------|----------|---------|
| Forget Gate  | f_t = sigmoid(W_f . [h_(t-1), x_t] + b_f)    | Discard irrelevant past info   |
| Input Gate   | i_t = sigmoid(W_i . [h_(t-1), x_t] + b_i)    | Select new info to store       |
| Cell State   | C_t = f_t * C_(t-1) + i_t * C~_t             | Long-term memory carrier       |
| Output Gate  | o_t = sigmoid(W_o . [h_(t-1), x_t] + b_o)    | Control what to output         |
| Hidden State | h_t = o_t * tanh(C_t)                         | Short-term output / next input |

---

## AI Acknowledgement (Mandatory as per assignment)

| Tool | Purpose | Sections Used |
|------|---------|---------------|
| Claude (Anthropic) | Code structuring, notebook layout, FastAPI scaffolding, README | Dataset loading, preprocessing, model architecture, API code, deployment config |

All LSTM concepts, mathematical understanding, and design decisions were reviewed and understood by the team. AI was used as a coding assistant, not as a replacement for learning.
